"""Intraday OHLCV + indicators from Kraken via ccxt — the ``kraken`` vendor.

Lets the framework analyse intraday bars (e.g. 15m) instead of daily candles.
Same OHLCV source as execution (Kraken), so analysis and orders see consistent
prices. Timeframe and lookback come from the active config:

    config["intraday"]                = True
    config["intraday_timeframe"]      = "15m"      # any ccxt timeframe
    config["intraday_lookback_bars"]  = 300        # bars to fetch

Wired in via ``VENDOR_METHODS`` (interface.py) for get_stock_data /
get_indicators, and via a branch in the verified-snapshot path
(market_data_validator.py). The daily yfinance path is untouched.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
from stockstats import wrap

from .config import get_config
from .symbol_utils import NoMarketDataError

logger = logging.getLogger(__name__)

_DEFAULT_TIMEFRAME = "15m"
_DEFAULT_BARS = 300
_DEFAULT_VWAP_BARS = 96  # rolling VWAP window: 96 x 15m = 24h of "fair value"
_INDICATOR_DISPLAY_BARS = 40  # how many recent bars to show in an indicator dump

_exchange = None  # public ccxt singleton (no API keys needed for OHLCV)


def _get_exchange():
    global _exchange
    if _exchange is None:
        import ccxt  # local import so non-intraday runs never require ccxt
        _exchange = ccxt.kraken({"enableRateLimit": True})
        _exchange.load_markets()
    return _exchange


def _ccxt_symbol(symbol: str) -> str:
    """Map a framework symbol (``SOL-USD``) to ccxt Kraken (``SOL/USD``)."""
    return symbol.upper().replace("-", "/")


def _timeframe() -> str:
    return get_config().get("intraday_timeframe", _DEFAULT_TIMEFRAME)


def _lookback_bars() -> int:
    return int(get_config().get("intraday_lookback_bars", _DEFAULT_BARS))


def _vwap_window() -> int:
    return int(get_config().get("intraday_vwap_bars", _DEFAULT_VWAP_BARS))


def _add_vwap(df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Add a rolling VWAP column — the intraday 'fair value' anchor.

    stockstats has no session VWAP, and 24/7 crypto has no session anyway, so
    we use a rolling volume-weighted average price over ``window`` bars
    (default 24h). Typical price = (H+L+C)/3.
    """
    window = window or _vwap_window()
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = (typical * df["Volume"]).rolling(window, min_periods=1).sum()
    vol = df["Volume"].rolling(window, min_periods=1).sum()
    df = df.copy()
    df["vwap"] = (pv / vol.replace(0, pd.NA)).astype(float)
    return df


def fetch_intraday_ohlcv(symbol: str, timeframe: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Return a DataFrame[Date, Open, High, Low, Close, Volume] of intraday bars.

    Bars run up to the most recent (still-forming) one, so the latest Close is
    effectively the current price. ``Date`` is tz-naive UTC, matching the rest
    of the dataflows.
    """
    timeframe = timeframe or _timeframe()
    limit = limit or _lookback_bars()
    csym = _ccxt_symbol(symbol)
    try:
        raw = _get_exchange().fetch_ohlcv(csym, timeframe=timeframe, limit=limit)
    except NoMarketDataError:
        raise
    except Exception as e:  # noqa: BLE001 - turn any ccxt error into the typed sentinel
        raise NoMarketDataError(symbol, csym, f"kraken fetch_ohlcv failed: {e}")
    if not raw:
        raise NoMarketDataError(symbol, csym, "no intraday bars returned")

    df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    # VWAP is added here so every consumer (get_stock_data, get_indicators, the
    # verified snapshot) sees the same 'vwap' column with no extra plumbing.
    return _add_vwap(df)


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """``kraken`` vendor for get_stock_data: intraday OHLCV as a CSV string.

    ``start_date``/``end_date`` are accepted for signature compatibility with
    the router; live intraday always returns the most recent ``lookback`` bars
    up to now (the analysis is always "as of now").
    """
    df = fetch_intraday_ohlcv(symbol)
    tf = _timeframe()
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d %H:%M")
    for col in ("Open", "High", "Low", "Close", "vwap"):
        if col in out.columns:
            out[col] = out[col].round(4)
    header = (
        f"# Intraday OHLCV for {symbol.upper()} — {tf} bars (Kraken)\n"
        f"# Bars: {len(out)} | latest bar: {out['Date'].iloc[-1]} UTC\n"
        f"# Retrieved: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
    )
    return header + out.to_csv(index=False)


def get_indicators(symbol: str, indicator: str, curr_date: str, look_back_days=None) -> str:
    """``kraken`` vendor for get_indicators: indicator computed per intraday bar.

    Returns the indicator value for the most recent bars (not per calendar day).
    ``curr_date``/``look_back_days`` are accepted for signature compatibility.
    """
    indicator = indicator.strip().lower()
    df = fetch_intraday_ohlcv(symbol)
    dates = df["Date"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    sdf = wrap(df.copy())
    try:
        sdf[indicator]  # triggers stockstats computation
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"Indicator {indicator} is not supported on intraday bars: {e}"
        )
    values = list(sdf[indicator].values)

    tf = _timeframe()
    pairs = list(zip(dates, values))[-_INDICATOR_DISPLAY_BARS:]
    lines = [f"## {indicator} on {tf} bars (last {len(pairs)}) for {symbol.upper()}:", ""]
    for ts, val in pairs:
        lines.append(f"{ts}: {'N/A' if pd.isna(val) else val}")
    lines.append("")
    lines.append(
        f"(Values are on {tf} intraday bars — periods are in bars, not days: "
        f"e.g. a 50-period SMA spans 50×{tf}.)"
    )
    return "\n".join(lines)
