"""Public Kraken Futures market-structure tools — funding, open interest,
order-book imbalance.

These read Kraken's PUBLIC futures endpoints via ccxt (no API keys). They give
the intraday analyst a feel for derivatives positioning and short-term order
flow, which spot OHLCV alone can't show. Every tool returns a short, readable
string and degrades to "data unavailable" on any error — a missing or flaky
futures feed must never crash the analysis.
"""

from typing import Annotated

from langchain_core.tools import tool

_exchange = None  # public ccxt krakenfutures singleton (no API keys needed)


def _get_exchange():
    """Lazily build a public krakenfutures ccxt client (local import so
    non-intraday runs never require ccxt)."""
    global _exchange
    if _exchange is None:
        import ccxt  # local import: keep ccxt out of the daily/import path

        _exchange = ccxt.krakenfutures({"enableRateLimit": True})
        _exchange.load_markets()
    return _exchange


def _futures_symbol(symbol: str) -> str:
    """Map a framework symbol (``SOL-USD``) to a ccxt krakenfutures perpetual
    symbol (``SOL/USD:USD``). If the caller already passes a ccxt-style symbol,
    leave it as-is."""
    s = symbol.upper()
    if "/" in s:
        return s
    base = s.replace("-USD", "").replace("-", "/")
    return f"{base}/USD:USD"


@tool
def get_funding_rate(
    symbol: Annotated[
        str,
        "Instrument symbol, e.g. 'SOL-USD'. Mapped to the Kraken Futures "
        "perpetual automatically.",
    ],
) -> str:
    """
    Retrieve the current perpetual funding rate and its recent trend from
    Kraken Futures (public data, no API key). Funding is the periodic payment
    between longs and shorts: a large POSITIVE rate means longs pay shorts
    (crowded-long, squeeze risk to the downside); a large NEGATIVE rate means
    shorts pay longs (crowded-short, squeeze risk to the upside).

    Args:
        symbol (str): Instrument symbol such as 'SOL-USD'

    Returns:
        str: A short, readable summary of current funding and recent trend,
        or 'data unavailable' on any error.
    """
    try:
        ex = _get_exchange()
        fsym = _futures_symbol(symbol)
        current = ex.fetch_funding_rate(fsym)
        rate = current.get("fundingRate")
        if rate is None:
            return "data unavailable"

        # Recent trend, if the history endpoint is available.
        trend = ""
        try:
            hist = ex.fetch_funding_rate_history(fsym, limit=8)
            rates = [h.get("fundingRate") for h in hist if h.get("fundingRate") is not None]
            if len(rates) >= 2:
                avg = sum(rates) / len(rates)
                direction = "rising" if rates[-1] > rates[0] else "falling"
                trend = (
                    f" Recent trend: {direction} "
                    f"(last {len(rates)} points avg {avg * 100:.4f}%)."
                )
        except Exception:  # noqa: BLE001 - history is best-effort
            trend = ""

        side = "longs pay shorts (crowded long)" if rate > 0 else (
            "shorts pay longs (crowded short)" if rate < 0 else "neutral"
        )
        return f"Funding rate for {fsym}: {rate * 100:.4f}% — {side}.{trend}"
    except Exception:  # noqa: BLE001 - never propagate
        return "data unavailable"


@tool
def get_open_interest(
    symbol: Annotated[
        str,
        "Instrument symbol, e.g. 'SOL-USD'. Mapped to the Kraken Futures "
        "perpetual automatically.",
    ],
) -> str:
    """
    Retrieve current open interest (OI) and its recent change from Kraken
    Futures (public data, no API key). Open interest is the total number of
    outstanding contracts: RISING OI alongside a price move signals fresh
    money / a 'real' move; FALLING OI during a move signals position-closing
    (a move that may fade).

    Args:
        symbol (str): Instrument symbol such as 'SOL-USD'

    Returns:
        str: A short, readable summary of current OI and recent change, or
        'data unavailable' on any error.
    """
    try:
        ex = _get_exchange()
        fsym = _futures_symbol(symbol)
        current = ex.fetch_open_interest(fsym)
        oi = current.get("openInterestAmount")
        if oi is None:
            oi = current.get("openInterestValue")
        if oi is None:
            return "data unavailable"

        # Recent change, if the history endpoint is available.
        change = ""
        try:
            hist = ex.fetch_open_interest_history(fsym, timeframe="1h", limit=8)
            vals = [
                h.get("openInterestAmount") or h.get("openInterestValue")
                for h in hist
            ]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2 and vals[0]:
                pct = (vals[-1] - vals[0]) / vals[0] * 100
                direction = "rising" if pct > 0 else "falling"
                change = f" Recent OI {direction} {pct:+.2f}% over last {len(vals)} points."
        except Exception:  # noqa: BLE001 - history is best-effort
            change = ""

        return f"Open interest for {fsym}: {oi:,.0f}.{change}"
    except Exception:  # noqa: BLE001 - never propagate
        return "data unavailable"


@tool
def get_orderbook_imbalance(
    symbol: Annotated[
        str,
        "Instrument symbol, e.g. 'SOL-USD'. Mapped to the Kraken Futures "
        "perpetual automatically.",
    ],
) -> str:
    """
    Measure the order-book imbalance at the top of the Kraken Futures book
    (public data, no API key): the bid vs ask volume summed over the first ~10
    levels. A book skewed toward BIDS = near-term buying pressure (support);
    skewed toward ASKS = near-term selling pressure (resistance). This is a
    short-lived, tactical signal only.

    Args:
        symbol (str): Instrument symbol such as 'SOL-USD'

    Returns:
        str: A short, readable summary of top-of-book imbalance, or
        'data unavailable' on any error.
    """
    try:
        ex = _get_exchange()
        fsym = _futures_symbol(symbol)
        book = ex.fetch_order_book(fsym, limit=10)
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            return "data unavailable"

        bid_vol = sum(level[1] for level in bids[:10] if len(level) > 1)
        ask_vol = sum(level[1] for level in asks[:10] if len(level) > 1)
        total = bid_vol + ask_vol
        if total <= 0:
            return "data unavailable"

        bid_pct = bid_vol / total * 100
        ask_pct = ask_vol / total * 100
        if bid_pct >= 60:
            read = "bid-heavy (near-term buying pressure)"
        elif ask_pct >= 60:
            read = "ask-heavy (near-term selling pressure)"
        else:
            read = "roughly balanced"
        return (
            f"Order-book imbalance for {fsym} (top 10 levels): "
            f"bids {bid_pct:.1f}% vs asks {ask_pct:.1f}% — {read}."
        )
    except Exception:  # noqa: BLE001 - never propagate
        return "data unavailable"
