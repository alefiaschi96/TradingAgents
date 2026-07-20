"""Run the TradingAgents pipeline and return its decision.

Wraps ``TradingAgentsGraph.propagate(symbol, date, asset_type="crypto")`` with
our Gemini configuration. The crypto pipeline is used so fundamentals (which
make no sense for a coin) are skipped; by default we run only the market,
social and news analysts.

``propagate`` returns ``(full_state, rating)`` where ``rating`` is one of
``Buy / Overweight / Hold / Underweight / Sell``.

The whole analysis is retried on transient network drops (e.g. Gemini's
"Server disconnected without sending a response"), which otherwise abort an
entire multi-minute run over a single flaky HTTP round-trip.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)


def _transient_network_errors() -> tuple[type[BaseException], ...]:
    """Exception types worth retrying: transport-level drops/timeouts.

    Built defensively so a missing httpx/httpcore never breaks the import.
    """
    errs: list[type[BaseException]] = [ConnectionError, TimeoutError]
    for mod_name in ("httpx", "httpcore"):
        try:
            mod = __import__(mod_name)
        except ImportError:
            continue
        for name in (
            "RemoteProtocolError", "ReadError", "WriteError", "ConnectError",
            "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout",
        ):
            exc = getattr(mod, name, None)
            if isinstance(exc, type) and issubclass(exc, BaseException):
                errs.append(exc)
    return tuple(errs)


@contextmanager
def _time_limit(seconds: int):
    """Raise TimeoutError if the wrapped block runs longer than ``seconds``.

    Uses SIGALRM, so it only arms on Unix and only in the main thread; anywhere
    else (or seconds<=0) it is a no-op. This turns a silently *stalled* call —
    e.g. a Gemini stream that opens but never delivers tokens — into a
    TimeoutError, which run_analysis already treats as transient and retries.
    """
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"analysis exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def build_config(cfg) -> dict:
    """Copy DEFAULT_CONFIG and apply our Gemini / debate settings."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = cfg.llm_provider
    config["deep_think_llm"] = cfg.deep_model
    config["quick_think_llm"] = cfg.quick_model
    config["temperature"] = cfg.temperature
    config["max_debate_rounds"] = cfg.debate_rounds
    config["max_risk_discuss_rounds"] = cfg.debate_rounds
    if cfg.google_thinking_level:
        config["google_thinking_level"] = cfg.google_thinking_level

    # Early market gate: when the analyst gate is on, a market report that backs
    # no trade (HOLD/flat or no usable data) ends the graph right after the
    # market analyst — the end-of-run gate would veto those cases anyway, so the
    # remaining LLM stages (other analysts, debate, trader, risk, PM) are pure
    # cost. The run then returns a plain "Hold".
    config["market_gate_short_circuit"] = bool(cfg.analyst_gate)

    # Prediction markets: set the vendor to "none" when disabled so the news
    # analyst neither queries Polymarket nor logs its failures (it is blocked at
    # the network level in some jurisdictions). Build a fresh data_vendors dict —
    # DEFAULT_CONFIG.copy() is shallow, so never mutate the shared one in place.
    if not cfg.prediction_markets:
        config["data_vendors"] = {
            **config.get("data_vendors", {}),
            "prediction_markets": "none",
        }

    # Intraday mode (Path 1): route market price + indicators + verified
    # snapshot to the Kraken intraday vendor so the whole analysis sees the
    # same intraday bars instead of daily candles.
    if cfg.intraday:
        config["intraday"] = True
        config["intraday_timeframe"] = cfg.intraday_timeframe
        config["intraday_lookback_bars"] = cfg.intraday_bars
        config["data_vendors"] = {
            **config.get("data_vendors", {}),
            "core_stock_apis": "kraken",
            "technical_indicators": "kraken",
        }
        # Surface the live trading scenario into the graph config so agents can
        # read it via get_config() (see get_scenario_instruction): instrument,
        # leverage, take-profit ratio, short permission and stop mode.
        config["leverage"] = cfg.leverage
        config["take_profit_rr"] = cfg.take_profit_rr
        config["allow_short"] = cfg.allow_short
        config["symbol"] = cfg.symbol
        config["analysis_symbol"] = cfg.analysis_symbol
        config["stop_mode"] = cfg.stop_mode
        # Paper-sim risk features (env-gated, 0/off = absent): surfaced to the
        # agents via get_scenario_instruction so their mental model of the
        # execution matches what the simulator will actually do (#Paul: agents
        # must know the mechanics they trade under).
        config["risk_pct_per_trade"] = float(os.environ.get("RISK_PCT_PER_TRADE", "0"))
        config["time_stop_hours"] = float(os.environ.get("TIME_STOP_HOURS", "0"))
        config["regime_exit_check"] = os.environ.get("REGIME_EXIT_CHECK", "0") == "1"
        # Placeholder; run_analysis fills this with the live regime read just
        # before building the graph (kept here so the key always exists).
        config["regime_context"] = ""
    return config


def run_analysis(cfg, trade_date: str | None = None) -> tuple[str, dict]:
    """Run the agents on the configured instrument, retrying transient drops.

    Returns ``(rating, full_state)``. ``rating`` is the 5-tier Portfolio
    Manager decision string. ``trade_date`` defaults to today (UTC).
    """
    trade_date = trade_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    config = build_config(cfg)

    # Higher-timeframe regime read, injected as INFORMATION into the agents'
    # prompts (via get_scenario_instruction). Never vetoes anything; fails open
    # to "" so a regime-fetch hiccup can't abort the analysis.
    if cfg.intraday:
        try:
            from live.regime import get_regime_context

            config["regime_context"] = get_regime_context(
                cfg.symbol,
                timeframe=cfg.regime_timeframe,
                ema_period=cfg.regime_ema_period,
            )
            if config["regime_context"]:
                logger.info("Regime context: %s", config["regime_context"])
        except Exception as e:  # noqa: BLE001 - informational only, never block
            logger.warning("Could not compute regime context (%s); continuing", e)
            config["regime_context"] = ""

    logger.info(
        "Running analysis: %s on %s (provider=%s deep=%s quick=%s analysts=%s)",
        cfg.analysis_symbol, trade_date, cfg.llm_provider,
        cfg.deep_model, cfg.quick_model, list(cfg.analysts),
    )

    graph = TradingAgentsGraph(
        selected_analysts=cfg.analysts,
        debug=cfg.debug,
        config=config,
    )

    # Long non-streaming Gemini calls (e.g. the market analyst's detailed
    # report) get their idle connection cut at ~60s on some networks, surfacing
    # as "Server disconnected without sending a response". Streaming keeps the
    # connection active (tokens flow continuously) so long calls survive. We
    # flip the flag on the already-built LLMs — no change to the upstream repo.
    if cfg.stream_llm and cfg.llm_provider.lower() == "google":
        enabled = []
        for name in ("deep_thinking_llm", "quick_thinking_llm"):
            llm = getattr(graph, name, None)
            if llm is not None and hasattr(llm, "streaming"):
                llm.streaming = True
                enabled.append(name)
        logger.info("Streaming enabled on Gemini clients: %s", enabled)

    transient = _transient_network_errors()
    attempts = max(1, cfg.analysis_max_attempts)
    last_err: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            with _time_limit(max(0, cfg.analysis_timeout_sec)):
                full_state, rating = graph.propagate(
                    cfg.analysis_symbol, trade_date, asset_type="crypto"
                )
            logger.info("Analysis decision for %s: %s", cfg.analysis_symbol, rating)
            return rating, full_state
        except transient as e:
            last_err = e
            if attempt >= attempts:
                break
            backoff = min(5 * attempt, 20)
            logger.warning(
                "Transient network error on attempt %d/%d (%s: %s). "
                "Retrying in %ds...",
                attempt, attempts, type(e).__name__, e, backoff,
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"Analysis failed after {attempts} attempt(s) due to transient "
        f"network errors; last was {type(last_err).__name__}: {last_err}"
    ) from last_err


def reasoning_from_state(state: dict | None) -> dict:
    """Pull every agent's reasoning out of a final graph state, flattened for
    structured (JSONL) logging.

    The full chain is otherwise discarded after ``run_analysis`` returns and
    only printed to stdout, so old runs aren't analysable. This captures, per
    decision: the market + news reports, the bull/bear case and the research
    manager's verdict, the trader's proposal, the three risk debaters, and the
    Portfolio Manager's final decision. Every field is best-effort — a partial
    state (e.g. after an error) just yields empty strings, never raises.
    """
    state = state or {}
    debate = state.get("investment_debate_state") or {}
    risk = state.get("risk_debate_state") or {}
    return {
        "market_report": state.get("market_report", ""),
        "news_report": state.get("news_report", ""),
        "bull_case": debate.get("bull_history", ""),
        "bear_case": debate.get("bear_history", ""),
        "research_manager_plan": state.get("investment_plan", ""),
        "trader_proposal": state.get("trader_investment_plan", ""),
        "risk_aggressive": risk.get("aggressive_history", ""),
        "risk_conservative": risk.get("conservative_history", ""),
        "risk_neutral": risk.get("neutral_history", ""),
        "pm_decision": state.get("final_trade_decision", ""),
    }
