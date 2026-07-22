"""Translate an order side into a Kraken Futures position with SL + TP.

Sizing: margin = balance * BALANCE_PCT, notional = margin * leverage,
amount = notional / price. Every entry is immediately bracketed by a
stop-loss and a take-profit at +/- STOP_PCT of the fill price.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _bracket_prices(
    side: str, entry_price: float, stop_pct: float, rr: float = 1.0
) -> tuple[float, float, str]:
    """Return (stop_loss, take_profit, close_side) for an entry.

    For a long: SL below, TP above, closed by a sell.
    For a short: SL above, TP below, closed by a buy.

    ``rr`` is the reward:risk ratio: the take-profit distance is ``rr`` times
    the stop distance. ``rr=1.0`` (default) keeps the original symmetric
    bracket, so existing callers are unchanged.
    """
    frac = stop_pct / 100.0
    tp_frac = frac * rr
    if side == "buy":  # long
        return entry_price * (1 - frac), entry_price * (1 + tp_frac), "sell"
    # short
    return entry_price * (1 + frac), entry_price * (1 - tp_frac), "buy"


def open_position(kraken, side: str, cfg) -> dict:
    """Open a position sized to the full balance, bracketed by SL/TP.

    Returns a structured result dict (also useful for logging / future DB).
    Honours dry-run via the KrakenClient write gates.
    """
    equity = kraken.get_equity_usd()
    price = kraken.get_last_price()

    margin = equity * cfg.balance_pct
    notional = margin * cfg.leverage
    amount = kraken.amount_for_notional(notional, price)

    min_amount = kraken.min_amount()
    if amount <= 0 or (min_amount and amount < min_amount):
        reason = (
            f"computed size {amount} below Kraken minimum {min_amount} "
            f"(equity={equity:.2f}, notional={notional:.2f}, price={price:.4f})"
        )
        logger.warning("Skipping entry: %s", reason)
        return {"opened": False, "reason": reason}

    # Leverage + isolated margin first, then the market entry.
    kraken.set_leverage_isolated(cfg.leverage)
    entry = kraken.create_market_entry(side, amount)

    # Use the real average fill price when available; fall back to last price
    # (always the case in dry-run, where no order is actually sent).
    entry_price = float(entry.get("average") or entry.get("price") or price)

    sl_price, tp_price, close_side = _bracket_prices(side, entry_price, cfg.stop_pct)
    sl = kraken.create_stop_loss(close_side, amount, sl_price)
    tp = kraken.create_take_profit(close_side, amount, tp_price)

    result = {
        "opened": True,
        "live": cfg.live,
        "symbol": cfg.symbol,
        "side": side,
        "amount": amount,
        "leverage": cfg.leverage,
        "equity_usd": equity,
        "notional_usd": notional,
        "entry_price": entry_price,
        "stop_loss": sl_price,
        "take_profit": tp_price,
        "orders": {"entry": entry, "stop_loss": sl, "take_profit": tp},
    }
    logger.info(
        "%s position %s: %s %s @ %.4f | SL %.4f TP %.4f (%.1f%%) lev %sx notional %.2f",
        "DRY-RUN" if not cfg.live else "LIVE",
        cfg.symbol, side, amount, entry_price, sl_price, tp_price,
        cfg.stop_pct, cfg.leverage, notional,
    )
    return result


def ensure_protective_orders(kraken, position: dict, cfg) -> dict:
    """Re-attach SL/TP to an existing position if they went missing.

    Safety net: a live position with no stops is the worst state to be in. If
    the open orders don't already include both a stop-loss and a take-profit,
    we recreate them from the position's entry price.
    """
    if not cfg.reattach_stops:
        return {"reattached": False, "reason": "REATTACH_STOPS=false"}

    open_orders = kraken.get_open_orders()
    has_sl = any(o.get("stopLossPrice") or o.get("triggerPrice") for o in open_orders)
    has_tp = any(o.get("takeProfitPrice") for o in open_orders)
    if has_sl and has_tp:
        return {"reattached": False, "reason": "stops already present"}

    side = "buy" if (position.get("side") == "long") else "sell"
    entry_price = float(position.get("entryPrice") or position.get("markPrice") or kraken.get_last_price())
    amount = abs(float(position.get("contracts") or 0.0))
    sl_price, tp_price, close_side = _bracket_prices(side, entry_price, cfg.stop_pct)

    placed = {}
    if not has_sl:
        placed["stop_loss"] = kraken.create_stop_loss(close_side, amount, sl_price)
    if not has_tp:
        placed["take_profit"] = kraken.create_take_profit(close_side, amount, tp_price)
    logger.warning(
        "Re-attached missing protective orders on %s (SL=%s TP=%s)",
        cfg.symbol, not has_sl, not has_tp,
    )
    return {"reattached": True, "orders": placed}
