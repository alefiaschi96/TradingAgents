"""Higher-timeframe regime read, shared between the paper-sim and live analysis.

The pure math (EMA series, ATR from OHLCV) lives here so both the paper-sim
gate (``paper_sim/simulator.py``) and the live analysis context can reuse it
instead of duplicating it.

``get_regime_context`` produces a *single English sentence* describing the
higher-timeframe trend (up / down / range), the price's position relative to a
moving EMA, and how far it sits from that EMA in ATR units. It is injected into
the agents' prompts purely as INFORMATION — it never blocks a trade. On ANY
error it returns "" (fail-open), so it can never break an analysis run.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def ema_series(closes: list[float], period: int) -> list[float]:
    """Exponential moving average series over ``closes`` (same recurrence as the
    paper-sim's ``_ema_series``)."""
    if not closes:
        return []
    k = 2.0 / (period + 1)
    ema = closes[0]
    out = [ema]
    for c in closes[1:]:
        ema = c * k + ema * (1.0 - k)
        out.append(ema)
    return out


def atr_from_ohlcv(ohlcv: list, period: int) -> float | None:
    """Average True Range over the last ``period`` bars of an OHLCV list
    (same calculation as the paper-sim's ``_atr_from_ohlcv``)."""
    if len(ohlcv) < period + 1:
        return None
    trs = []
    for i in range(1, len(ohlcv)):
        high, low, prev_close = (
            float(ohlcv[i][2]),
            float(ohlcv[i][3]),
            float(ohlcv[i - 1][4]),
        )
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    trs = trs[-period:]
    return sum(trs) / len(trs) if trs else None


def _fetch_ohlcv_public(symbol: str, timeframe: str, limit: int) -> list:
    """Fetch candles from a PUBLIC krakenfutures ccxt client (no keys), with a
    small retry — the charts endpoint drops often. Raises on persistent failure.
    """
    import ccxt  # imported lazily so config/tests don't require ccxt installed

    exchange = ccxt.krakenfutures({"enableRateLimit": True})
    exchange.load_markets()
    # Resolve the unified symbol from a native id (e.g. "PF_SOLUSD") or accept
    # an already-unified symbol ("SOL/USD:USD").
    ccxt_symbol = symbol
    for m in exchange.markets.values():
        if symbol in (m.get("id"), m.get("symbol")):
            ccxt_symbol = m["symbol"]
            break

    last_err = None
    for attempt in range(3):
        try:
            return exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)
        except Exception as e:  # noqa: BLE001 - transient charts-endpoint blips
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise last_err


def get_regime_context(
    symbol: str,
    timeframe: str = "1h",
    ema_period: int = 20,
    *,
    fetch_ohlcv=None,
) -> str:
    """Build a one-sentence higher-timeframe regime description for ``symbol``.

    Downloads ``ema_period + 5`` candles of ``timeframe`` from a public
    krakenfutures client, computes the EMA, its slope (a few bars back), and the
    price's distance from the EMA in ATR units, then returns e.g.::

        "Higher-timeframe (1h) regime: UPTREND - price above a rising 20-bar
         EMA, ~1.8 ATR from it."

    Returns "" on ANY error (no data, ccxt missing, network drop, bad math):
    this is purely informational context and must never break the analysis.

    ``fetch_ohlcv`` is an injection point for tests: a callable
    ``(symbol, timeframe, limit) -> ohlcv`` used instead of the real fetch.
    """
    try:
        period = int(ema_period) if ema_period else 20
        tf = timeframe or "1h"
        need = period + 5  # extra bars for the slope lookback and ATR
        fetch = fetch_ohlcv or _fetch_ohlcv_public
        ohlcv = fetch(symbol, tf, need)
        if not ohlcv or len(ohlcv) < period + 2:
            return ""

        closes = [float(c[4]) for c in ohlcv]
        ema = ema_series(closes, period)
        if not ema:
            return ""
        price = closes[-1]
        ema_now = ema[-1]
        ema_prev = ema[-4] if len(ema) >= 4 else ema[0]
        slope = ema_now - ema_prev
        atr = atr_from_ohlcv(ohlcv, min(period, len(ohlcv) - 1))

        if price > ema_now and slope > 0:
            label = "UPTREND"
            desc = f"price above a rising {period}-bar EMA"
        elif price < ema_now and slope < 0:
            label = "DOWNTREND"
            desc = f"price below a falling {period}-bar EMA"
        else:
            label = "RANGE"
            desc = f"price oscillating around a flat {period}-bar EMA"

        stretch_txt = ""
        if atr and atr > 0:
            stretch = abs(price - ema_now) / atr
            stretch_txt = f", ~{stretch:.1f} ATR from it"

        return (
            f"Higher-timeframe ({tf}) regime: {label} - {desc}{stretch_txt}."
        )
    except Exception as e:  # noqa: BLE001 - fail open, never block the analysis
        logger.warning("regime context unavailable (%s); continuing without it", e)
        return ""
