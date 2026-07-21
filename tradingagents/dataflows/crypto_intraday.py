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
verified snapshot.

Kraken's spot API only supports these native candle sizes: 1m, 5m, 15m,
30m, 1h, 4h, 1d, 1w, 2w (verified via ``ccxt.kraken().timeframes``).
There is NO native 10m, 3h, or 6h — requesting them directly causes an
"Invalid arguments" error. So "10m", "3h", and "6h" are *synthetic*
timeframes here: we fetch the nearest valid base timeframe (5m or 1h)
and resample with pandas. See ``_SYNTHETIC_TF_MAP``.
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

# Multi-timeframe definitions: (timeframe_label, bars_to_fetch). The label is
# resolved to a real ccxt call (native or resampled) by fetch_intraday_ohlcv.
_MULTI_TF_SPECS: list[tuple[str, int]] = [
    ("10m", 12),   # ~2h lookback — the primary / fastest
    ("1h",   6),   # ~6h lookback
    ("3h",   6),   # ~18h lookback
    ("6h",  18),   # ~108h (4.5 days) lookback
]

# Kraken has no native 10m/3h/6h candles. Map each synthetic label to a
# native base timeframe + the pandas resample rule + how many base bars make
# one synthetic bar (used to size the base fetch with margin for warmup).
_SYNTHETIC_TF_MAP: dict[str, tuple[str, str, int]] = {
    "10m": ("5m", "10min", 2),
    "3h": ("1h", "3h", 3),
    "6h": ("1h", "6h", 6),
}

# Kraken's charts endpoint drops often (503s, empty responses), and is prone
# to rate-limiting when the analyst issues many tool calls in one cycle. A
# single failed attempt used to blind the whole market analyst for the cycle
# ("data unavailable") while the paper-sim regime filter — which retries —
# kept seeing the same candles fine. Retry with backoff, same as the simulator.
_OHLCV_ATTEMPTS = 4
_OHLCV_BASE_SLEEP = 2.0  # seconds; grows linearly per attempt (2s, 4s, 6s, 8s)
_sleep = time.sleep  # indirection so tests can stub the backoff

# Short-TTL cache for raw OHLCV fetches. A single analyst turn can call
# get_stock_data, get_indicators (several times), get_verified_market_snapshot,
# and get_sl_tp_levels — each of which used to re-fetch the same bars from
# Kraken independently, multiplying API calls and tripping "Too many
# requests". Bars for a given (symbol, base timeframe) don't change within a
# few seconds, so we cache the raw ccxt response briefly and reuse it.
_OHLCV_CACHE_TTL = 20.0  # seconds
_ohlcv_cache: dict[tuple[str, str, int], tuple[float, list]] = {}

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

    Results are cached briefly (``_OHLCV_CACHE_TTL``) per (csym, timeframe,
    limit) so multiple tool calls within one analyst turn reuse the same
    fetch instead of re-hitting Kraken and tripping its rate limiter.
    """
    cache_key = (csym, timeframe, limit)
    now = time.time()
    cached = _ohlcv_cache.get(cache_key)
    if cached is not None and (now - cached[0]) < _OHLCV_CACHE_TTL:
        return cached[1]

    last_err: Exception | None = None
    for attempt in range(1, _OHLCV_ATTEMPTS + 1):
        try:
            raw = _get_exchange().fetch_ohlcv(csym, timeframe=timeframe, limit=limit)
            if raw:
                _ohlcv_cache[cache_key] = (now, raw)
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


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a base-timeframe OHLCV DataFrame to a coarser ``rule``
    (e.g. "10min", "3h", "6h"), aligned to UTC midnight."""
    df = df.set_index("Date")
    resampled = df.resample(rule, origin="start_day").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Open"])
    return resampled.reset_index()


def _fetch_base_df(symbol: str, ccxt_timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch a native-timeframe OHLCV DataFrame (no VWAP columns yet)."""
    csym = _ccxt_symbol(symbol)
    raw = _fetch_ohlcv_with_retry(symbol, csym, ccxt_timeframe, limit)
    df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def fetch_intraday_ohlcv(symbol: str, timeframe: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Return a DataFrame[Date, Open, High, Low, Close, Volume, vwap, ...] of
    intraday bars for ``timeframe``.

    ``timeframe`` may be a native Kraken interval (1m/5m/15m/30m/1h/4h/1d/...)
    or one of the synthetic labels "10m"/"3h"/"6h", which are resampled from
    a native base timeframe (see ``_SYNTHETIC_TF_MAP``) since Kraken has no
    native candle for them.

    ``limit`` is the number of OUTPUT bars (post-resample) desired.

    Bars run up to the most recent (still-forming) one, so the latest Close is
    effectively the current price. ``Date`` is tz-naive UTC, matching the rest
    of the dataflows.
    """
    timeframe = timeframe or _timeframe()
    limit = limit or _lookback_bars()

    if timeframe in _SYNTHETIC_TF_MAP:
        base_tf, rule, per_bar = _SYNTHETIC_TF_MAP[timeframe]
        # Fetch extra base bars so resampling yields at least `limit` full
        # bins even after dropping a leading partial bin.
        base_limit = limit * per_bar + per_bar * 2
        base_df = _fetch_base_df(symbol, base_tf, base_limit)
        df = _resample_ohlcv(base_df, rule)
        df = df.tail(limit).reset_index(drop=True)
    else:
        df = _fetch_base_df(symbol, timeframe, limit)

    # VWAP is added here so every consumer (get_stock_data, get_indicators, the
    # verified snapshot) sees the same 'vwap' column with no extra plumbing.
    df = _add_vwap(df, window=min(len(df), _vwap_window()) or None)
    df = _add_session_vwap(df)
    return df


def fetch_multi_timeframe_ohlcv(symbol: str) -> dict[str, pd.DataFrame]:
    """Fetch four labelled OHLCV series for multi-timeframe analysis.

    Returns ``{"10m": df, "1h": df, "3h": df, "6h": df}`` where each DataFrame
    has columns [Date, Open, High, Low, Close, Volume, vwap, session_vwap, ...].
    """
    return {
        label: fetch_intraday_ohlcv(symbol, timeframe=label, limit=bars)
        for label, bars in _MULTI_TF_SPECS
    }


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

    # Bars-per-label lookup, kept in sync with _MULTI_TF_SPECS
    _LABEL_BARS = dict(_MULTI_TF_SPECS)
    tf_label = tf_override or "10m"
    limit = _LABEL_BARS.get(tf_label, 12)
    df = fetch_intraday_ohlcv(symbol, timeframe=tf_label, limit=limit)

    # Session-VWAP columns are already on the DataFrame; return directly
    if ind_name.startswith("session_vwap") or ind_name.startswith("svwap_"):
        if ind_name not in df.columns:
            raise ValueError(f"Column {ind_name} not found. "
                             f"Available session VWAP cols: session_vwap, "
                             f"svwap_1s_upper, svwap_1s_lower, svwap_2s_upper, svwap_2s_lower")
        dates = df["Date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        vals = df[ind_name].tolist()
        pairs = list(zip(dates, vals))[-_INDICATOR_DISPLAY_BARS:]
        lines = [f"## {ind_name} on {tf_label} bars (last {len(pairs)}) for {symbol.upper()}:", ""]
        for ts, val in pairs:
            lines.append(f"{ts}: {'N/A' if pd.isna(val) else round(val, 4)}")
        return "\n".join(lines)

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
