"""Decision mapping and pre-trade safety checks.

These are the hard rails that sit between the LLM decision and a real order:
nothing reaches Kraken without passing through here.
"""

from __future__ import annotations

import logging

from tradingagents.agents.utils.market_lean import (  # noqa: F401 - re-exported
    market_analyst_lean,
    market_report_has_no_data,
)

logger = logging.getLogger(__name__)


def analyst_gate(market_report: str, side: str, *, enabled: bool = True) -> tuple[bool, str]:
    """Pre-trade veto: block entries the market analyst's report doesn't back.

    Detects the two anomaly patterns found in the paper-sim logs (Jul 2026):
    positions opened while the analyst declared its data feed unusable, and
    positions opened while the analyst's own stance was HOLD/flat/neutral.
    Both cohorts lost money net; trades backed by a clear, agreeing analyst
    read were the only profitable subset. The report classifier itself lives in
    ``tradingagents.agents.utils.market_lean`` — shared with the in-graph
    market gate that short-circuits the pipeline early on the same signals.

    Vetoes when the analyst had no usable market data, or when its stance is
    HOLD/neutral or points the other way. Can only VETO a decision, never
    create one. An empty report (market analyst not enabled) fails open.
    Returns ``(allowed, reason)``.
    """
    if not enabled:
        return True, "analyst gate off"
    text = (market_report or "").strip()
    if not text:
        return True, "no market report (allow)"
    if market_report_has_no_data(text):
        return False, "market analyst had no reliable market data"
    lean = market_analyst_lean(text)
    wanted = "bull" if side == "buy" else "bear"
    if lean == wanted:
        return True, f"analyst {lean} read backs {side}"
    if lean in ("bull", "bear"):
        return False, f"{side} against market analyst {lean} read"
    return False, "market analyst recommends HOLD/flat (no directional signal)"


def map_decision_to_side(rating: str, cfg) -> str | None:
    """Map a 5-tier rating to an order side, honouring ALLOW_SHORT.

    Returns ``"buy"`` (long), ``"sell"`` (short), or ``None`` (stay flat).
    """
    rating_norm = (rating or "").strip()
    if rating_norm in cfg.long_signals:
        return "buy"
    if rating_norm in cfg.short_signals:
        if not cfg.allow_short:
            logger.info("Decision %s is short but ALLOW_SHORT=false -> flat", rating_norm)
            return None
        return "sell"
    return None


def check_pre_trade(kraken, cfg) -> tuple[bool, str]:
    """Run the safety checks that must pass before opening a position.

    Returns ``(ok, reason)``. ``reason`` is empty when ok.
    """
    # 1. Leverage cap — never exceed the configured hard ceiling.
    if cfg.leverage > cfg.max_leverage:
        return False, f"leverage {cfg.leverage}x exceeds MAX_LEVERAGE {cfg.max_leverage}x"

    # 2. Equity kill-switch — stop opening new trades below the floor.
    equity = kraken.get_equity_usd()
    if cfg.equity_floor_usd > 0 and equity < cfg.equity_floor_usd:
        return False, f"equity {equity:.2f} below floor {cfg.equity_floor_usd:.2f}"
    if equity <= 0:
        return False, "no readable equity on the futures wallet"

    return True, ""
