"""Precomputed stop-loss / take-profit candidates for the intraday analyst.

Given a direction (long/short) and the current price, this tool computes
ATR-based SL and R-multiple TP levels so the LLM doesn't have to derive
them from raw ATR values.
"""

from typing import Annotated

from langchain_core.tools import tool


@tool
def get_sl_tp_levels(
    symbol: Annotated[str, "Instrument symbol, e.g. 'SOL-USD'"],
    direction: Annotated[str, "'long' or 'short' — the proposed trade direction"],
) -> str:
    """Compute ATR-based stop-loss and take-profit levels for the next 1–2h.

    Uses the 10m primary series (12 bars) to compute the 14-bar ATR, then
    returns:
    - SL at 1.5×ATR from entry (against the trade direction)
    - TP at 2R (risk × 2, in the trade direction)
    - TP at 3R (risk × 3, in the trade direction)
    along with the resulting R:R ratios.

    Args:
        symbol: Instrument symbol such as 'SOL-USD'
        direction: 'long' or 'short'

    Returns:
        str: Formatted SL/TP table, or 'data unavailable' on error.
    """
    try:
        from tradingagents.dataflows.crypto_intraday import fetch_intraday_ohlcv

        # Fetch enough 10m bars for a 14-period ATR calculation
        df = fetch_intraday_ohlcv(symbol, timeframe="10m", limit=30)
        if len(df) < 3:
            return "data unavailable — not enough bars for ATR"

        # Compute ATR manually (14-period, or as many as available)
        period = min(14, len(df) - 1)
        trs = []
        for i in range(1, len(df)):
            high = float(df.iloc[i]["High"])
            low = float(df.iloc[i]["Low"])
            prev_close = float(df.iloc[i - 1]["Close"])
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        atr = sum(trs[-period:]) / len(trs[-period:])

        entry = float(df.iloc[-1]["Close"])
        risk = 1.5 * atr  # SL distance

        d = direction.strip().lower()
        if d not in ("long", "short"):
            return f"Invalid direction '{direction}'. Use 'long' or 'short'."

        if d == "long":
            sl = entry - risk
            tp_2r = entry + 2 * risk
            tp_3r = entry + 3 * risk
        else:
            sl = entry + risk
            tp_2r = entry - 2 * risk
            tp_3r = entry - 3 * risk

        lines = [
            f"## SL/TP levels for {symbol.upper()} — {d.upper()} (10m ATR)",
            "",
            f"| Metric | Value |",
            f"|---|---:|",
            f"| Entry (latest close) | {entry:.4f} |",
            f"| ATR (14-bar, 10m) | {atr:.4f} |",
            f"| SL (1.5×ATR) | {sl:.4f} |",
            f"| Risk distance | {risk:.4f} |",
            f"| TP at 2R | {tp_2r:.4f} |",
            f"| TP at 3R | {tp_3r:.4f} |",
            f"| R:R (SL→TP 2R) | 1:{2.0:.1f} |",
            f"| R:R (SL→TP 3R) | 1:{3.0:.1f} |",
            "",
            f"Stop-loss is {risk:.4f} away from entry ({1.5:.1f}×ATR). "
            f"TP targets are 2× and 3× the risk distance.",
        ]
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — never propagate
        return "data unavailable"
