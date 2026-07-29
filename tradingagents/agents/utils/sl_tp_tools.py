"""Precomputed stop-loss / take-profit profiles for the Risk Manager.

Given a direction (long/short) and the current price, this tool computes
ATR-based SL and R-multiple TP levels deterministically.  Only the Risk
Manager calls this (via ``.invoke()``); analysts never touch it.

Three risk profiles are offered (Tight / Standard / Wide) so the Risk
Manager can pick the bracket that best fits current volatility and conviction.
"""

from typing import Annotated

from langchain_core.tools import tool

# Minimum risk distance as a fraction of entry price.  Prevents absurdly
# tight SL/TP when ATR is compressed (e.g. low-vol weekend candles).
_MIN_RISK_PCT = 0.015  # 1.5 %

# (ATR multiplier for stop distance, R-multiple for take-profit)
PROFILES: dict[str, tuple[float, float]] = {
    "Tight":    (1.0, 2.0),
    "Standard": (1.5, 2.0),
    "Wide":     (2.0, 3.0),
}


@tool
def get_sl_tp_levels(
    symbol: Annotated[str, "Instrument symbol, e.g. 'SOL-USD'"],
    direction: Annotated[str, "'long' or 'short' — the proposed trade direction"],
) -> str:
    """Compute ATR-based stop-loss and take-profit profiles for the next 1–2h.

    Uses the **4h** series (14-bar ATR) so the volatility estimate reflects
    full intraday swings rather than micro-noise.  A minimum risk floor of
    1.5 % of entry price prevents nonsensically tight levels during low-vol
    periods.

    Returns three profiles (Tight / Standard / Wide), each with a different
    ATR multiplier and R:R ratio.  The Risk Manager picks the one that best
    fits current conditions.
    """
    try:
        from tradingagents.dataflows.crypto_intraday import fetch_intraday_ohlcv

        # ── ATR on 4h bars ──────────────────────────────────────────
        df_atr = fetch_intraday_ohlcv(symbol, timeframe="4h", limit=30)
        if len(df_atr) < 3:
            return "data unavailable — not enough 4h bars for ATR"

        # Current price from the freshest 10m bar
        df_entry = fetch_intraday_ohlcv(symbol, timeframe="10m", limit=5)
        entry = (
            float(df_entry.iloc[-1]["Close"])
            if len(df_entry) >= 1
            else float(df_atr.iloc[-1]["Close"])
        )

        # 14-period ATR (or as many as available)
        period = min(14, len(df_atr) - 1)
        trs = []
        for i in range(1, len(df_atr)):
            high = float(df_atr.iloc[i]["High"])
            low = float(df_atr.iloc[i]["Low"])
            prev_close = float(df_atr.iloc[i - 1]["Close"])
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        atr = sum(trs[-period:]) / len(trs[-period:])

        d = direction.strip().lower()
        if d not in ("long", "short"):
            return f"Invalid direction '{direction}'. Use 'long' or 'short'."

        risk_floor = entry * _MIN_RISK_PCT

        lines = [
            f"## SL/TP profiles for {symbol.upper()} — {d.upper()} (4h ATR)",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Entry (latest close) | {entry:.4f} |",
            f"| ATR (14-bar, 4h) | {atr:.4f} |",
            f"| Min risk floor (1.5%) | {risk_floor:.4f} |",
            "",
            "| Profile | SL | TP | Risk Dist | R:R |",
            "|---|---:|---:|---:|---|",
        ]

        for name, (atr_mult, rr) in PROFILES.items():
            risk_raw = atr_mult * atr
            risk = max(risk_raw, risk_floor)
            floored = risk_raw < risk_floor
            flag = " *" if floored else ""

            if d == "long":
                sl = entry - risk
                tp = entry + rr * risk
            else:
                sl = entry + risk
                tp = entry - rr * risk

            lines.append(
                f"| {name} ({atr_mult:.1f}×ATR, {rr:.0f}R){flag} "
                f"| {sl:.4f} | {tp:.4f} | {risk:.4f} | 1:{rr:.1f} |"
            )

        lines.extend([
            "",
            "Pick the profile that best fits current volatility and conviction.",
            "* = risk distance was floored to 1.5% of entry.",
        ])
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — never propagate
        return "data unavailable"
