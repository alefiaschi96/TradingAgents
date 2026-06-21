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
import time
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
    return config


def run_analysis(cfg, trade_date: str | None = None) -> tuple[str, dict]:
    """Run the agents on the configured instrument, retrying transient drops.

    Returns ``(rating, full_state)``. ``rating`` is the 5-tier Portfolio
    Manager decision string. ``trade_date`` defaults to today (UTC).
    """
    trade_date = trade_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    config = build_config(cfg)

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
