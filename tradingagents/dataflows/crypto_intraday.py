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

Multi-timeframe support
~~~~~~~~~~~~~~~~~~~~~~~
``fetch_multi_timeframe_ohlcv`` returns four labelled OHLCV series
(10m×12, 1h×6, 3h×6, 6h×18) so the market analyst can reason across
horizons.  The 10m series is the "primary" for indicators and the
verified snapshot.  Kraken doesn't offer native 3h candles, so we
resample from 1h×18.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from stockstats import wrap

from .config import get_config
from .symbol_utils import NoMarketDataError

logger = logging.getLogger(__name__)

_DEFAULT_TIMEFRAME = "15m"
_DEFAULT_BARS = 300
_DEFAULT_VWAP_BARS = 96  # rolling VWAP window: 96 x 15m = 24h of "fair value"
_INDICATOR_DISPLAY_BARS = 40  # how many recent bars to show in an indicator dump

# Multi-timeframe definitions: (timeframe_label, ccxt_timeframe, bars_to_fetch)
# Kraken doesn't support 3h natively — we fetch 1h×18 and resample.
_MULTI_TF_SPECS: list[tuple[str, str, int]] = [
    ("10m", "10m", 12),   # ~2h lookback — the primary / fastest
    ("1h",  "1h",   6),   # ~6h lookback
    ("3h",  "1h",  18),   # resampled from 1h×18 → 3h×6
    ("6h",  "6h",  18),   # ~108h (4.5 days) lookback
]

# Kraken's charts endpoint drops often (503s, empty responses). A single failed
# attempt used to blind the whole market analyst for the cycle ("data
# unavailable") while the paper-sim regime filter — which retries — kept seeing
# the same candles fine. Retry with the same backoff the simulator uses.
_OHLCV_ATTEMPTS = 3
_OHLCV_BASE_SLEEP = 1.0  # seconds; grows linearly per attempt (1s, 2s)
_sleep = time.sleep  # indirection so tests can stub the backoff

_exchange = None  # public ccxt singleton (no API keys needed for OHLCV)


def _get_exchange():
    global _exchange
    if _exchange is None:
        import ccxt  # local import so non-intraday runs never require ccxt
        exchange = ccxt.kraken({"enableRateLimit": True})
        # Load markets BEFORE publishing the singleton: assigning first would
        # leave a half-initialised exchange (no markets) permanently cached if
        # this network call fails once.
        exchange.load_markets()
        _exchange = exchange
    return _exchange


def _fetch_ohlcv_with_retry(symbol: str, csym: str, timeframe: str, limit: int) -> list:
    """Fetch OHLCV bars, retrying transient Kraken drops before giving up.

    An empty bar list counts as a failure too (Kraken occasionally returns 200
    with no data). Only after every attempt fails does this raise the typed
    ``NoMarketDataError`` that the tool layer reports as "data unavailable".
    """
    last_err: Exception | None = None
    for attempt in range(1, _OHLCV_ATTEMPTS + 1):
        try:
            raw = _get_exchange().fetch_ohlcv(csym, timeframe=timeframe, limit=limit)
            if raw:
                return raw
            last_err = ValueError("no intraday bars returned")
        except Exception as e:  # noqa: BLE001 - every ccxt/network error is retryable here
            last_err = e
        if attempt < _OHLCV_ATTEMPTS:
            logger.warning(
                "kraken fetch_ohlcv %s %s attempt %d/%d failed (%s); retrying",
                csym, timeframe, attempt, _OHLCV_ATTEMPTS, last_err,
            )
            _sleep(_OHLCV_BASE_SLEEP * attempt)
    raise NoMarketDataError(
        symbol, csym,
        f"kraken fetch_ohlcv failed after {_OHLCV_ATTEMPTS} attempts: {last_err}",
    )


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


def _add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add session-anchored VWAP (reset at 00:00 UTC) with ±1σ/2σ bands.

    Columns added: session_vwap, svwap_1s_upper, svwap_1s_lower,
    svwap_2s_upper, svwap_2s_lower.

    Within each UTC day the cumulative VWAP is:
        session_vwap = Σ(typical * volume) / Σ(volume)
    and the bands are based on the running volume-weighted variance:
        σ = sqrt( Σ(volume * (typical - session_vwap)²) / Σ(volume) )
    """
    df = df.copy()
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    dates = df["Date"].dt.date

    svwap = np.full(len(df), np.nan)
    sigma = np.full(len(df), np.nan)

    for _day, idx in df.groupby(dates).groups.items():
        idx_arr = idx.sort_values() if hasattr(idx, "sort_values") else sorted(idx)
        cum_pv = 0.0
        cum_vol = 0.0
        cum_var = 0.0
        for i in idx_arr:
            tp = typical.iloc[i]
            v = df["Volume"].iloc[i]
            cum_pv += tp * v
            cum_vol += v
            if cum_vol > 0:
                vw = cum_pv / cum_vol
                cum_var += v * (tp - vw) ** 2
                svwap[i] = vw
                sigma[i] = math.sqrt(cum_var / cum_vol)

    df["session_vwap"] = svwap
    s = pd.Series(sigma, index=df.index)
    df["svwap_1s_upper"] = df["session_vwap"] + s
    df["svwap_1s_lower"] = df["session_vwap"] - s
    df["svwap_2s_upper"] = df["session_vwap"] + 2 * s
    df["svwap_2s_lower"] = df["session_vwap"] - 2 * s
    return df


def _resample_to_3h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a 1h DataFrame to 3h bars."""
    df = df.set_index("Date")
    resampled = df.resample("3h", origin="start_day").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Open"])
    resampled = resampled.reset_index()
    return resampled


def fetch_intraday_ohlcv(symbol: str, timeframe: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Return a DataFrame[Date, Open, High, Low, Close, Volume] of intraday bars.

    Bars run up to the most recent (still-forming) one, so the latest Close is
    effectively the current price. ``Date`` is tz-naive UTC, matching the rest
    of the dataflows.
    """
    timeframe = timeframe or _timeframe()
    limit = limit or _lookback_bars()
    csym = _ccxt_symbol(symbol)
    raw = _fetch_ohlcv_with_retry(symbol, csym, timeframe, limit)

    df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    # VWAP is added here so every consumer (get_stock_data, get_indicators, the
    # verified snapshot) sees the same 'vwap' column with no extra plumbing.
    df = _add_vwap(df)
    df = _add_session_vwap(df)
    return df


def fetch_multi_timeframe_ohlcv(symbol: str) -> dict[str, pd.DataFrame]:
    """Fetch four labelled OHLCV series for multi-timeframe analysis.

    Returns ``{"10m": df, "1h": df, "3h": df, "6h": df}`` where each DataFrame
    has columns [Date, Open, High, Low, Close, Volume, vwap, session_vwap, ...].
    The 3h series is resampled from 1h bars because Kraken lacks a native 3h tf.
    """
    csym = _ccxt_symbol(symbol)
    result: dict[str, pd.DataFrame] = {}

    for label, ccxt_tf, bars in _MULTI_TF_SPECS:
        raw = _fetch_ohlcv_with_retry(symbol, csym, ccxt_tf, bars)
        df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        if label == "3h":
            df = _resample_to_3h(df)

        # Add both rolling and session-anchored VWAP
        df = _add_vwap(df, window=min(len(df), _vwap_window()))
        df = _add_session_vwap(df)
        result[label] = df

    return result


def _format_ohlcv_csv(df: pd.DataFrame, symbol: str, label: str) -> str:
    """Format a single-timeframe DataFrame as a labelled CSV section."""
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d %H:%M")
    round_cols = [c for c in ("Open", "High", "Low", "Close", "vwap",
                               "session_vwap", "svwap_1s_upper", "svwap_1s_lower",
                               "svwap_2s_upper", "svwap_2s_lower") if c in out.columns]
    for col in round_cols:
        out[col] = out[col].round(4)
    header = (
        f"## {label} bars (last {len(out)}) for {symbol.upper()}\n"
        f"## Latest bar: {out['Date'].iloc[-1]} UTC\n\n"
    )
    return header + out.to_csv(index=False) + "\n"


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """``kraken`` vendor for get_stock_data: multi-timeframe intraday OHLCV.

    ``start_date``/``end_date`` are accepted for signature compatibility with
    the router; live intraday always returns the most recent bars up to now.

    Returns four clearly separated CSV sections (10m, 1h, 3h, 6h) so the
    analyst can reason across multiple horizons.
    """
    mtf = fetch_multi_timeframe_ohlcv(symbol)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        f"# Multi-timeframe intraday OHLCV for {symbol.upper()} (Kraken)\n"
        f"# Retrieved: {now_str} UTC\n"
        f"# Series: 10m×{len(mtf['10m'])}, 1h×{len(mtf['1h'])}, "
        f"3h×{len(mtf['3h'])}, 6h×{len(mtf['6h'])}\n"
        f"# Columns include rolling 24h VWAP and session-anchored VWAP "
        f"(reset 00:00 UTC) with ±1σ/2σ bands.\n\n",
    ]
    for label in ("10m", "1h", "3h", "6h"):
        parts.append(_format_ohlcv_csv(mtf[label], symbol, label))
    return "".join(parts)


def get_indicators(symbol: str, indicator: str, curr_date: str, look_back_days=None) -> str:
    """``kraken`` vendor for get_indicators: indicator computed per intraday bar.

    Returns the indicator value for the most recent bars (not per calendar day).
    ``curr_date``/``look_back_days`` are accepted for signature compatibility.

    When the indicator name contains a ``@<timeframe>`` suffix (e.g.
    ``rsi@1h``), the indicator is computed on bars of that timeframe instead
    of the default 10m series.
    """
    raw_indicator = indicator.strip().lower()

    # Parse optional @timeframe suffix
    if "@" in raw_indicator:
        ind_name, tf_override = raw_indicator.rsplit("@", 1)
    else:
        ind_name, tf_override = raw_indicator, None

    # Session-VWAP columns are already on the DataFrame; return directly
    if ind_name.startswith("session_vwap") or ind_name.startswith("svwap_"):
        df = fetch_intraday_ohlcv(symbol, timeframe=tf_override or "10m",
                                  limit={"10m": 12, "1h": 6, "6h": 18}.get(tf_override, 12))
        if ind_name not in df.columns:
            raise ValueError(f"Column {ind_name} not found. "
                             f"Available session VWAP cols: session_vwap, "
                             f"svwap_1s_upper, svwap_1s_lower, svwap_2s_upper, svwap_2s_lower")
        dates = df["Date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        vals = df[ind_name].tolist()
        tf_label = tf_override or "10m"
        pairs = list(zip(dates, vals))[-_INDICATOR_DISPLAY_BARS:]
        lines = [f"## {ind_name} on {tf_label} bars (last {len(pairs)}) for {symbol.upper()}:", ""]
        for ts, val in pairs:
            lines.append(f"{ts}: {'N/A' if pd.isna(val) else round(val, 4)}")
        return "\n".join(lines)

    # Determine which timeframe to use
    if tf_override:
        tf_label = tf_override
        limit_map = {"10m": 12, "1h": 6, "6h": 18}
        limit = limit_map.get(tf_override, 12)
        if tf_override == "3h":
            # Fetch 1h and resample
            df = fetch_intraday_ohlcv(symbol, timeframe="1h", limit=18)
            df = _resample_to_3h(df)
            df = _add_vwap(df, window=min(len(df), _vwap_window()))
            df = _add_session_vwap(df)
        else:
            df = fetch_intraday_ohlcv(symbol, timeframe=tf_override, limit=limit)
    else:
        # Default: 10m series (primary)
        tf_label = "10m"
        df = fetch_intraday_ohlcv(symbol, timeframe="10m", limit=12)

    dates = df["Date"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    sdf = wrap(df.copy())
    try:
        sdf[ind_name]  # triggers stockstats computation
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"Indicator {ind_name} is not supported on intraday bars: {e}"
        )
    values = list(sdf[ind_name].values)

    pairs = list(zip(dates, values))[-_INDICATOR_DISPLAY_BARS:]
    lines = [f"## {ind_name} on {tf_label} bars (last {len(pairs)}) for {symbol.upper()}:", ""]
    for ts, val in pairs:
        lines.append(f"{ts}: {'N/A' if pd.isna(val) else val}")
    lines.append("")
    lines.append(
        f"(Values are on {tf_label} intraday bars — periods are in bars, not days: "
        f"e.g. a 50-period SMA spans 50×{tf_label}.)"
    )
    return "\n".join(lines)
