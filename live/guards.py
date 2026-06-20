"""Decision mapping and pre-trade safety checks.

These are the hard rails that sit between the LLM decision and a real order:
nothing reaches Kraken without passing through here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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
