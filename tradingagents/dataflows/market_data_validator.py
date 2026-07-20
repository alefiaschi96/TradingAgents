"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.stockstats_utils import load_ohlcv

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)

# Intraday snapshot set: fast EMAs, VWAP (the intraday fair-value anchor),
# session VWAP with bands, momentum, volatility — periods are in bars, not days.
INTRADAY_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_9_ema", "close_21_ema", "vwap",
    "session_vwap", "svwap_1s_upper", "svwap_1s_lower",
    "svwap_2s_upper", "svwap_2s_lower",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)


def _verified_rows(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV on or before curr_date, date-sorted. Raises if nothing usable.

    ``load_ohlcv`` already normalizes the Date column and filters out
    look-ahead rows, but we re-apply the cutoff defensively — this is a
    verification path, so it must not trust its input to be pre-filtered.

    In intraday mode the snapshot must match what the analyst sees, so we pull
    intraday bars from the same Kraken source as get_stock_data/get_indicators
    (the daily look-ahead cutoff does not apply: live analysis is "as of now").
    """
    from tradingagents.dataflows.config import get_config

    if get_config().get("intraday"):
        from tradingagents.dataflows.crypto_intraday import fetch_intraday_ohlcv

        # Use the 10m series (primary) for the verified snapshot —
        # it has the most recent price and matches the trade horizon.
        df = fetch_intraday_ohlcv(symbol, timeframe="10m", limit=12)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        if df.empty:
            raise ValueError(f"No intraday OHLCV rows for {symbol}.")
        return df

    data = load_ohlcv(symbol, curr_date)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {symbol}.")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(curr_date)].sort_values("Date")
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {curr_date} for {symbol}.")
    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_verified_market_snapshot(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes."""
    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    intraday = bool(get_config().get("intraday"))
    df = _verified_rows(symbol, curr_date)
    stock_df = wrap(df.copy())

    def _fmt_dt(value):
        """Show the bar time intraday; date-only for daily."""
        if intraday and isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d %H:%M")
        return _fmt(value)

    default_set = INTRADAY_SNAPSHOT_INDICATORS if intraday else DEFAULT_SNAPSHOT_INDICATORS
    selected = tuple(indicators or default_set)
    indicator_values: dict[str, str] = {}
    # Session-VWAP columns live on the raw df, not in stockstats
    _SVWAP_COLS = {"session_vwap", "svwap_1s_upper", "svwap_1s_lower",
                   "svwap_2s_upper", "svwap_2s_lower"}
    for name in selected:
        try:
            if name in _SVWAP_COLS and name in df.columns:
                indicator_values[name] = _fmt(df.iloc[-1][name])
            else:
                stock_df[name]  # triggers stockstats calculation
                indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt_dt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    if intraday:
        tf = get_config().get("intraday_timeframe", "15m")
        header_lines = [
            f"## Verified market data snapshot for {symbol.upper()} ({tf} intraday bars)",
            "",
            f"- Most recent bar (treat as the current moment): {latest_date} UTC",
            "- Bars run up to now; the latest bar is still forming.",
        ]
    else:
        header_lines = [
            f"## Verified market data snapshot for {symbol.upper()}",
            "",
            f"- Requested analysis date: {curr_date}",
            f"- Latest trading row used: {latest_date}",
            "- Rows after the requested analysis date are excluded before verification.",
        ]
    lines = header_lines + [
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt_dt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)
