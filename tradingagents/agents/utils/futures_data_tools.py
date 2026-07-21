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

        # --- Historical trend (last 6–8 periods) with acceleration ---
        trend = ""
        hist_table = ""
        try:
            hist = ex.fetch_funding_rate_history(fsym, limit=8)
            entries = []
            for h in hist:
                r = h.get("fundingRate")
                ts = h.get("timestamp") or h.get("datetime")
                if r is not None:
                    entries.append((ts, r))
            rates = [e[1] for e in entries]
            if len(rates) >= 2:
                avg = sum(rates) / len(rates)
                direction = "rising" if rates[-1] > rates[0] else "falling"

                # Acceleration: is the rate change accelerating?
                deltas = [rates[i] - rates[i - 1] for i in range(1, len(rates))]
                accel = ""
                if len(deltas) >= 2:
                    second_deltas = [deltas[i] - deltas[i - 1] for i in range(1, len(deltas))]
                    avg_accel = sum(second_deltas) / len(second_deltas)
                    if abs(avg_accel) > 1e-6:
                        accel_dir = "accelerating" if (
                            (direction == "rising" and avg_accel > 0) or
                            (direction == "falling" and avg_accel < 0)
                        ) else "decelerating"
                        accel = f" Rate change is **{accel_dir}**."

                # Crowding trade flag
                crowding = ""
                if all(rates[i] >= rates[i - 1] for i in range(1, len(rates))):
                    crowding = " ⚠ Monotonically rising — signals a crowding long trade."
                elif all(rates[i] <= rates[i - 1] for i in range(1, len(rates))):
                    crowding = " ⚠ Monotonically falling — signals a crowding short trade."

                trend = (
                    f" Trend: {direction} over {len(rates)} periods "
                    f"(avg {avg * 100:.4f}%).{accel}{crowding}"
                )

                # Build individual-rate table with human-readable UTC timestamps
                lines = ["\n| Period (UTC) | Rate (%) |", "|---|---:|"]
                for ts_val, r_val in entries:
                    if isinstance(ts_val, (int, float)):
                        from datetime import datetime, timezone
                        ts_str = datetime.fromtimestamp(
                            ts_val / 1000, tz=timezone.utc
                        ).strftime("%Y-%m-%d %H:%M")
                    else:
                        ts_str = str(ts_val)[:16] if ts_val else "?"
                    lines.append(f"| {ts_str} | {r_val * 100:.4f}% |")
                hist_table = "\n".join(lines)
        except Exception:  # noqa: BLE001 - history is best-effort
            trend = ""

        side = "longs pay shorts (crowded long)" if rate > 0 else (
            "shorts pay longs (crowded short)" if rate < 0 else "neutral"
        )
        return f"Funding rate for {fsym}: {rate * 100:.4f}% — {side}.{trend}{hist_table}"
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


@tool
def get_btc_trend(
    symbol: Annotated[
        str,
        "The ticker you are currently analyzing (NOT BTC). This is used "
        "only for labelling; BTC data is fetched automatically.",
    ],
) -> str:
    """Determine whether BTC is bullish or bearish for the current day.

    Fetches BTC/USD 1h bars for the last 24h from Kraken spot, computes:
    - price vs today's UTC open
    - 9-bar EMA vs 21-bar EMA alignment

    Returns a one-line classification: BULLISH / BEARISH / NEUTRAL plus
    supporting data. Use this to understand the macro crypto direction when
    analyzing a non-BTC asset.

    Args:
        symbol: The current ticker being analyzed (for labelling only).

    Returns:
        str: BTC intraday trend summary, or 'data unavailable' on error.
    """
    try:
        # Use the spot Kraken exchange (same singleton as crypto_intraday)
        from tradingagents.dataflows.crypto_intraday import (
            _ccxt_symbol,
            _fetch_ohlcv_with_retry,
        )

        btc_csym = _ccxt_symbol("BTC-USD")
        raw = _fetch_ohlcv_with_retry("BTC-USD", btc_csym, "1h", 24)
        if not raw or len(raw) < 3:
            return "BTC trend data unavailable"

        closes = [float(c[4]) for c in raw]
        opens = [float(c[1]) for c in raw]
        current_price = closes[-1]

        # Today's open: first bar whose date matches UTC today
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date()
        today_open = None
        for bar in raw:
            bar_date = datetime.fromtimestamp(bar[0] / 1000, tz=timezone.utc).date()
            if bar_date == today:
                today_open = float(bar[1])
                break
        if today_open is None:
            today_open = opens[0]

        # EMA9 / EMA21
        def _ema(values, period):
            k = 2.0 / (period + 1)
            ema = values[0]
            for v in values[1:]:
                ema = v * k + ema * (1.0 - k)
            return ema

        ema9 = _ema(closes, 9) if len(closes) >= 9 else closes[-1]
        ema21 = _ema(closes, 21) if len(closes) >= 21 else closes[-1]

        above_open = current_price > today_open
        ema_bullish = ema9 > ema21

        if above_open and ema_bullish:
            label = "BULLISH"
        elif not above_open and not ema_bullish:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        pct = (current_price - today_open) / today_open * 100 if today_open else 0
        return (
            f"BTC intraday trend: **{label}** — "
            f"price {current_price:.2f} vs today's open {today_open:.2f} "
            f"({pct:+.2f}%), EMA9({ema9:.2f}) {'>' if ema_bullish else '<='} "
            f"EMA21({ema21:.2f}) on 1h bars."
        )
    except Exception:  # noqa: BLE001 — never propagate
        return "BTC trend data unavailable"
