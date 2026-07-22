"""Public Kraken Futures market-structure tools — funding, open interest,
order-book imbalance.

These read Kraken's PUBLIC futures endpoints via ccxt (no API keys). They give
the intraday analyst a feel for derivatives positioning and short-term order
flow, which spot OHLCV alone can't show. Every tool returns a short, readable
string and degrades to "data unavailable" on any error — a missing or flaky
futures feed must never crash the analysis.

Transient endpoint drops are retried (same 3-attempts-with-backoff policy as
the core intraday OHLCV fetch in ``dataflows.crypto_intraday``) so a single
network blip doesn't blind the analyst to positioning for the whole cycle.
Structural failures (ccxt ``NotSupported``, bad symbols) fail fast: retrying
them would only add dead seconds before the fallback path.
"""

import json
import logging
import os
import time
from typing import Annotated

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_exchange = None  # public ccxt krakenfutures singleton (no API keys needed)

# --- local open-interest series --------------------------------------------
# ccxt's krakenfutures exposes neither fetchOpenInterest nor its history, so
# the analyst only ever saw a bare OI level — which it (rightly) refused to
# read directionally: OI is informative as a CHANGE (rising = fresh money,
# falling = position-closing), not as a level. Every analysis cycle calls the
# tool anyway, so we record the readings ourselves and rebuild the change
# series the endpoint doesn't provide. Persisted so daemon restarts keep it.
_OI_HISTORY_PATH = os.path.expanduser("~/.tradingagents/oi_history.json")
_OI_HISTORY_WINDOW_SEC = 48 * 3600  # drop readings older than 2 days
_OI_HISTORY_MAX_POINTS = 50         # per symbol
_OI_MIN_SPAN_SEC = 300              # need >=5 min of span to state a trend


def _oi_history_update(symbol: str, oi: float, now: float | None = None,
                       path: str | None = None) -> list:
    """Append this OI reading to the per-symbol local series; return the PRIOR
    readings (oldest first, excluding the one just added).

    Best-effort by design: a missing/corrupt file resets the series and an
    unwritable disk loses one point — never worth failing the tool over.
    """
    now = time.time() if now is None else now
    path = path or _OI_HISTORY_PATH
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:  # noqa: BLE001 - missing/corrupt store: start fresh
        data = {}
    series = [
        p for p in data.get(symbol, [])
        if isinstance(p, list) and len(p) == 2
        and isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))
        and now - p[0] <= _OI_HISTORY_WINDOW_SEC
    ]
    prior = list(series)
    series.append([now, float(oi)])
    data[symbol] = series[-_OI_HISTORY_MAX_POINTS:]
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:  # noqa: BLE001 - losing one point is fine
        pass
    return prior


def _oi_change_text(prior: list, oi: float, now: float | None = None) -> str:
    """One readable clause on how OI moved vs the recorded series, or ''.

    Compares against the OLDEST prior reading that is at least
    ``_OI_MIN_SPAN_SEC`` old, so back-to-back calls inside one analysis never
    produce a meaningless "+0.00% over 0 min".
    """
    now = time.time() if now is None else now
    old = [p for p in prior if now - p[0] >= _OI_MIN_SPAN_SEC and p[1]]
    if not old:
        return ""
    t0, v0 = old[0]
    pct = (oi - v0) / v0 * 100.0
    direction = "rising" if pct > 0 else ("falling" if pct < 0 else "flat")
    mins = (now - t0) / 60.0
    return (
        f" Recent OI {direction} {pct:+.2f}% over the last {mins:.0f} min "
        f"({len(prior) + 1} readings)."
    )

_ATTEMPTS = 3
_BASE_SLEEP = 1.0  # seconds; grows linearly per attempt (1s, 2s)
_sleep = time.sleep  # indirection so tests can stub the backoff

# Exception class names that no amount of retrying will fix — fail fast so
# e.g. an unsupported endpoint drops straight to its fallback path.
_NON_RETRYABLE = {"NotSupported", "BadSymbol", "BadRequest", "ArgumentsRequired"}


def _call_with_retry(fn, *args, **kwargs):
    """Run one public-endpoint call, retrying transient drops before giving up.

    Re-raises the last error after ``_ATTEMPTS`` failures (the calling tool's
    own try/except turns that into "data unavailable"). Non-retryable ccxt
    errors are re-raised immediately.
    """
    last_err = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - every ccxt/network error lands here
            if type(e).__name__ in _NON_RETRYABLE:
                raise
            last_err = e
            if attempt < _ATTEMPTS:
                logger.warning(
                    "krakenfutures %s attempt %d/%d failed (%s); retrying",
                    getattr(fn, "__name__", fn), attempt, _ATTEMPTS, e,
                )
                _sleep(_BASE_SLEEP * attempt)
    raise last_err


def _get_exchange():
    """Lazily build a public krakenfutures ccxt client (local import so
    non-intraday runs never require ccxt).

    The singleton is only stored once ``load_markets`` has succeeded —
    assigning it earlier would cache a half-initialised client after a
    transient failure and poison every later call.
    """
    global _exchange
    if _exchange is None:
        import ccxt  # local import: keep ccxt out of the daily/import path

        ex = ccxt.krakenfutures({"enableRateLimit": True})
        _call_with_retry(ex.load_markets)
        _exchange = ex
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
        current = _call_with_retry(ex.fetch_funding_rate, fsym)
        rate = current.get("fundingRate")
        if rate is None:
            return "data unavailable"

        # Recent trend, if the history endpoint is available.
        trend = ""
        try:
            hist = _call_with_retry(ex.fetch_funding_rate_history, fsym, limit=8)
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
    (a move that may fade). The change clause compares against readings
    recorded at previous analysis cycles (typically 30-60 min apart); when it
    is absent this is the first reading of the session, not missing data.

    Args:
        symbol (str): Instrument symbol such as 'SOL-USD'

    Returns:
        str: A short, readable summary of current OI and recent change, or
        'data unavailable' on any error.
    """
    try:
        ex = _get_exchange()
        fsym = _futures_symbol(symbol)
        oi = None
        try:
            current = _call_with_retry(ex.fetch_open_interest, fsym)
            oi = current.get("openInterestAmount")
            if oi is None:
                oi = current.get("openInterestValue")
        except Exception:  # noqa: BLE001 - ccxt krakenfutures lacks this endpoint
            oi = None
        if oi is None:
            # ccxt's krakenfutures has no fetchOpenInterest, but the public
            # ticker carries the live OI — use it as the primary fallback.
            ticker = _call_with_retry(ex.fetch_ticker, fsym)
            raw = (ticker.get("info") or {}).get("openInterest")
            oi = float(raw) if raw is not None else None
        if oi is None:
            return "data unavailable"

        # Record the reading in the local series regardless of what the
        # history endpoint can do — this is what makes the change clause
        # possible on krakenfutures at all.
        try:
            prior = _oi_history_update(fsym, float(oi))
        except Exception:  # noqa: BLE001 - the series is best-effort context
            prior = []

        # Recent change: the exchange history endpoint when available,
        # else the locally recorded series.
        change = ""
        try:
            hist = _call_with_retry(
                ex.fetch_open_interest_history, fsym, timeframe="1h", limit=8
            )
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
        if not change:
            try:
                change = _oi_change_text(prior, float(oi))
            except Exception:  # noqa: BLE001 - never fail the tool over context
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
        book = _call_with_retry(ex.fetch_order_book, fsym, limit=10)
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
