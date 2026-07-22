"""Analyst gate: veto entries the market analyst's report doesn't back.

The phrases below are taken verbatim from the paper-sim JSONL logs
(logs/paper_sim, July 2026 runs) where the two anomaly patterns were found:
positions opened while the analyst said HOLD/flat, and positions opened while
the analyst declared its market-data feed unusable. Note the typographic
apostrophe (can’t) — the LLM reports never use the ASCII one.
"""

import pytest

from live.guards import analyst_gate, market_analyst_lean

# --- real report snippets ----------------------------------------------------

FTP_HOLD = (
    "FINAL TRANSACTION PROPOSAL: **HOLD**\n\n"
    "ETH-USD is still in an intraday uptrend, but the better read for the "
    "next 1–2 hours is **wait / hold rather than chase**."
)

FTP_BUY = (
    "FINAL TRANSACTION PROPOSAL: **BUY**\n\n"
    "ETH-USD remains in a valid intraday uptrend: price is above VWAP and "
    "the 9/21 EMA structure is bullish."
)

FTP_SELL = (
    "FINAL TRANSACTION PROPOSAL: **SELL**\n\n"
    "Sellers are in control below VWAP; continuation lower is favoured."
)

NO_DATA_HEAD = (
    "Data unavailable for `ETH-USD` on the configured market-data source, so "
    "I can’t make a reliable 1–2 hour intraday call without "
    "fabricating values.\n\n- 15m OHLCV / VWAP data: unavailable"
)

NO_DATA_PRODUCE = (
    "I can’t produce a reliable intraday technical read for **ETH-USD** "
    "right now because the market-data source returned **no usable 15m OHLCV "
    "data** and the verified snapshot is empty."
)

# 07-08T10:20: no usable OHLCV but a bearish lean anyway — opened short, hit SL.
NO_DATA_WITH_LEAN = (
    "ETH-USD intraday data is unavailable from the market-data vendor right "
    "now, so I can’t make a grounded 1–2 hour price-level call from "
    "OHLCV/verified snapshot. That said, the indicators that did come "
    "through lean bearish.\n\n| 1–2h bias | Bearish lean | Favor "
    "short/avoid longs unless price reclaims strength |"
)

HOLD_INLINE = (
    "ETH-USD: **HOLD** for the next 1–2 hours.\n\nThe verified snapshot "
    "shows a **neutral-to-slightly-bullish intraday recovery**, but not "
    "enough confirmation to chase long here.\n"
    "| Trade stance | Consolidation | Wait for reclaim of 1797–1798 |"
)

BULL_PROSE = (
    "**ETH-USD 15m intraday read (next 1–2 hours): bullish continuation "
    "favoured.**\n\nPrice is above VWAP with a bullish EMA stack; momentum "
    "is bullish and dips are being bought.\n"
    "| 1–2 hour bias | Moderately bullish |"
)

BEAR_PROSE = (
    "The tape is bearish: price below VWAP, bearish EMA cross, sellers in "
    "control. Fade the pops.\n"
    "| 1–2 hour bias | Bearish lean | Favor short |"
)


# --- lean classification -------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    ("report", "lean"),
    [
        (FTP_HOLD, "neutral"),
        (FTP_BUY, "bull"),
        (FTP_SELL, "bear"),
        (NO_DATA_HEAD, "nodata"),
        (HOLD_INLINE, "neutral"),
        (BULL_PROSE, "bull"),
        (BEAR_PROSE, "bear"),
        ("", "neutral"),
    ],
)
def test_market_analyst_lean(report, lean):
    assert market_analyst_lean(report) == lean


# --- gate verdicts -------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("side", ["buy", "sell"])
def test_hold_proposal_vetoes_both_sides(side):
    allowed, reason = analyst_gate(FTP_HOLD, side)
    assert not allowed
    assert "HOLD" in reason


@pytest.mark.unit
@pytest.mark.parametrize("side", ["buy", "sell"])
def test_no_data_vetoes_both_sides(side):
    for report in (NO_DATA_HEAD, NO_DATA_PRODUCE):
        allowed, reason = analyst_gate(report, side)
        assert not allowed
        assert "no reliable market data" in reason


@pytest.mark.unit
def test_no_data_wins_even_with_directional_lean():
    # The one blind-but-directional trade in the logs still lost: data first.
    allowed, reason = analyst_gate(NO_DATA_WITH_LEAN, "sell")
    assert not allowed
    assert "no reliable market data" in reason


@pytest.mark.unit
def test_agreeing_direction_is_allowed():
    assert analyst_gate(FTP_BUY, "buy")[0]
    assert analyst_gate(FTP_SELL, "sell")[0]
    assert analyst_gate(BULL_PROSE, "buy")[0]
    assert analyst_gate(BEAR_PROSE, "sell")[0]


@pytest.mark.unit
def test_opposite_direction_is_vetoed():
    allowed, reason = analyst_gate(FTP_BUY, "sell")
    assert not allowed and "against" in reason
    allowed, reason = analyst_gate(BEAR_PROSE, "buy")
    assert not allowed and "against" in reason


@pytest.mark.unit
def test_inline_hold_is_vetoed():
    allowed, _ = analyst_gate(HOLD_INLINE, "sell")
    assert not allowed


@pytest.mark.unit
def test_empty_report_fails_open():
    # No market analyst configured -> the gate has nothing to judge.
    allowed, _ = analyst_gate("", "buy")
    assert allowed


@pytest.mark.unit
def test_disabled_gate_allows_everything():
    allowed, reason = analyst_gate(FTP_HOLD, "buy", enabled=False)
    assert allowed and reason == "analyst gate off"
