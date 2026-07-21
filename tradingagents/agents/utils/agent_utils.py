import functools
import logging
from collections.abc import Mapping
from typing import Any

import yfinance as yf
from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators
from tradingagents.agents.utils.market_data_validation_tools import get_verified_market_snapshot
from tradingagents.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from tradingagents.agents.utils.prediction_markets_tools import get_prediction_markets
from tradingagents.agents.utils.technical_indicators_tools import get_indicators

# Public surface: the data tools are imported here so agents and the graph
# import them from one place, plus the instrument/language helpers defined below.
__all__ = [
    "get_stock_data",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_macro_indicators",
    "get_prediction_markets",
    "get_verified_market_snapshot",
    "build_instrument_context",
    "resolve_instrument_identity",
    "get_instrument_context_from_state",
    "get_language_instruction",
    "get_horizon_instruction",
    "get_scenario_instruction",
    "create_msg_delete",
]

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def get_horizon_instruction() -> str:
    """Return a prompt directive that pins the trade horizon.

    In intraday mode (config ``intraday``) it forces every agent to reason for a
    position held over the next 1-2 hours on intraday bars, and away from
    fundamentals / multi-day theses. Returns empty string in daily mode so the
    default behavior is unchanged. Applied to researchers, debaters, research
    manager, trader, and portfolio manager.
    """
    from tradingagents.dataflows.config import get_config
    config = get_config()
    if not config.get("intraday"):
        return ""
    tf = config.get("intraday_timeframe", "15m")
    return (
        f" HORIZON: This is an INTRADAY trade. Evaluate strictly for a position "
        f"held over the next 1-2 hours on {tf} bars. Ground every conclusion in "
        f"intraday price action — momentum, trend (EMA/VWAP), key intraday "
        f"levels, breakout vs mean-reversion, and volatility (ATR) for stops. Do "
        f"NOT reason from fundamentals, valuation, earnings, revenue, scalability "
        f"or multi-day/multi-week theses; they are irrelevant on this horizon. It "
        f"is a short-term trade, not an investment."
    )


def get_scenario_instruction() -> str:
    """Return a prompt directive describing the concrete trading scenario.

    In intraday mode (config ``intraday``) it spells out the mechanics the agents
    are actually trading: a crypto perpetual future, the configured leverage,
    that LONG / SHORT / FLAT are all available with FLAT as the default, and how
    each position is sized and bracketed by an automatic stop and take-profit.
    The text is built dynamically from the live config so it always matches the
    deployed instrument and risk settings, and it prescribes NO strategy.
    Returns empty string in daily mode so the default behavior is unchanged.
    """
    from tradingagents.dataflows.config import get_config
    config = get_config()
    if not config.get("intraday"):
        return ""

    symbol = config.get("symbol") or config.get("analysis_symbol")
    instrument = f" ({symbol})" if symbol else ""

    leverage = config.get("leverage")
    try:
        lev_txt = f" at {leverage:g}x leverage" if leverage else ""
    except (TypeError, ValueError):
        lev_txt = ""

    rr = config.get("take_profit_rr")
    try:
        rr_txt = f"~{rr:g}x" if rr else "a multiple of"
    except (TypeError, ValueError):
        rr_txt = "a multiple of"

    tf = config.get("intraday_timeframe", "15m")

    if config.get("allow_short", True):
        direction = (
            "You may take a LONG or a SHORT with equal ease, or stay FLAT."
        )
    else:
        direction = "You may take a LONG, or stay FLAT (shorts are disabled)."

    risk_pct = config.get("risk_pct_per_trade") or 0
    if risk_pct:
        sizing_txt = (
            f"Each position is sized so a stop-out risks ~{risk_pct:g}% of equity "
            f"(leverage is a cap, not a target)"
        )
    else:
        sizing_txt = "Each position is sized automatically"

    # Describe early exits only when they are actually armed, so the agents'
    # mental model of the bracket matches the simulator's real behaviour.
    early_exits = []
    time_stop = config.get("time_stop_hours") or 0
    if time_stop:
        early_exits.append(
            f"a time-stop closes stagnant positions after ~{time_stop:g}h"
        )
    if config.get("regime_exit_check"):
        early_exits.append(
            "a clean opposite higher-timeframe regime closes the position early"
        )
    if early_exits:
        exit_txt = "held until stop/target unless " + " or ".join(early_exits)
    else:
        exit_txt = "then left until one is hit"

    min_rr = config.get("min_analyst_rr") or 0
    target_rule = (
        f" Trades whose stated price target lies nearer than ~{min_rr:g}x the stop "
        f"distance are SKIPPED automatically — if you see real edge, state a price "
        f"target with room to run; a timid target equals no trade."
        if min_rr
        else ""
    )

    # Structural-SL: the PM declares an execution_plan stop contract and must
    # know EXACTLY how each field executes — the whole point is moving intent
    # out of prose into fields the executor honours literally.
    structural_rule = (
        " STOP CONTRACT: your execution_plan is executed exactly as declared. "
        "'touch' exits the instant the 1-minute range trades through your "
        "invalidation level; 'close'/'hold' exit only after your confirm_bars "
        "consecutive 15m CLOSES beyond it — intrabar wicks do NOT trigger them. "
        "Your hard_level always executes on touch and its distance sizes the "
        "position (it IS your risk budget), so place it beyond wick noise. "
        "With 'touch' the invalidation level itself IS the stop and any "
        "hard_level is IGNORED — the wick protection is off. Declare 'touch' "
        "only when the mere trade-through of the level kills the thesis on "
        "the spot; if your reasoning mentions wick noise, failed reclaims or "
        "waiting for a close, that is 'close'/'hold', not 'touch'. "
        "Give concrete price levels — freeze VWAP/EMA at their current value. "
        "If price has already crossed your invalidation level when the order "
        "would execute, the trade is skipped entirely."
        if config.get("structural_sl")
        else ""
    )

    scenario = (
        f" SCENARIO: you trade a crypto PERPETUAL FUTURE{instrument}{lev_txt}, "
        f"intraday on {tf} bars, holding ~1-2 hours. {direction} FLAT is the "
        f"default and the correct choice whenever there is no clear directional "
        f"edge — never take a position just to be active. {sizing_txt} and "
        f"bracketed by a volatility-based stop and a take-profit "
        f"at {rr_txt} the stop distance, {exit_txt}; one position "
        f"at a time.{target_rule}{structural_rule} Reason only from intraday "
        f"price action and fresh catalysts."
    )

    # Optional higher-timeframe regime read, injected as INFORMATION only. It
    # prescribes nothing and vetoes nothing — the agent decides what to make of
    # it. Empty when unavailable (fail-open).
    regime_context = config.get("regime_context", "")
    if regime_context:
        scenario += (
            f" CURRENT REGIME (informational, you decide what to do with it): "
            f"{regime_context}"
        )

    return scenario


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. Cached so
    the lookup happens at most once per ticker per process.

    The symbol is normalized first (e.g. ``XAUUSD`` -> ``GC=F``) so identity
    resolves for the same instrument the price path actually fetches (#983).
    """
    from tradingagents.dataflows.symbol_utils import normalize_symbol

    try:
        info = yf.Ticker(normalize_symbol(ticker)).info or {}
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    if identity:
        name = identity.get("company_name") or identity.get("name")
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``TradingAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
    )


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-anchored placeholder.

        The placeholder must not be a bare ``"Continue"``: some
        OpenAI-compatible providers interpret that literally as the user task
        and produce output about the word "continue" instead of analysing the
        instrument (#888). Anchoring it to the resolved instrument context and
        date keeps the next analyst on-task even if the provider treats the
        placeholder as a standalone request.
        """
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        instrument_context = get_instrument_context_from_state(state)
        trade_date = state.get("trade_date", "the requested date")
        placeholder = HumanMessage(
            content=(
                f"Proceed with your assigned analysis for this workflow. "
                f"{instrument_context} The analysis date is {trade_date}."
            )
        )
        return {"messages": removal_operations + [placeholder]}

    return delete_messages



