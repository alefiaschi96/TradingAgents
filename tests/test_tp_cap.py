"""TP cap: the PM's stated "Price Target" bounds the mechanical rr bracket.

parse_price_target reads the structured field the PM decision ends with
(observed verbatim in logs/paper_sim JSONL, July 2026: "**Price Target**:
1866.49"). capped_tp then uses it only when it sits strictly between entry and
the rr-based TP — the rr level stays the default and the hard ceiling, so a
missing/garbage target changes nothing.
"""

import pytest

from paper_sim.simulator import capped_tp, parse_price_target

# --- real PM decision tail (2026-07-16 10:21 UTC run) ------------------------

PM_TAIL = (
    "**Investment Thesis**: ETH-USD warrants an Underweight rating because "
    "the live intraday structure favors sellers...\n\n"
    "**Price Target**: 1866.49\n\n"
    "**Time Horizon**: 1-2 hours"
)


# ------------------------------------------------------- parse_price_target

def test_parses_structured_field():
    assert parse_price_target(PM_TAIL) == 1866.49


def test_missing_field_is_none():
    assert parse_price_target("**Rating**: Hold\n\nNo directional view.") is None
    assert parse_price_target("") is None
    assert parse_price_target(None) is None


def test_tolerates_dollar_sign_commas_and_tilde():
    assert parse_price_target("Price Target: $1,900") == 1900.0
    assert parse_price_target("**Price Target**: ~1866") == 1866.0


def test_last_match_wins_over_thesis_mention():
    text = (
        "The bear case argued a price target: 1880 was likely.\n"
        "**Price Target**: 1866.49"
    )
    assert parse_price_target(text) == 1866.49


def test_prose_mention_without_colon_does_not_match():
    assert parse_price_target("a price target of 1880 seems fair") is None


def test_non_numeric_target_is_none():
    assert parse_price_target("**Price Target**: N/A") is None


# ----------------------------------------------------------------- capped_tp

LONG = ("buy", 1918.78, 1937.01)    # entry, rr TP (the July 16 long)
SHORT = ("sell", 1887.72, 1858.86)  # entry, rr TP (the July 16 short)


def test_long_target_between_entry_and_rr_caps():
    side, entry, tp_rr = LONG
    assert capped_tp(side, entry, tp_rr, 1926.60) == (1926.60, "analyst_target")


def test_short_target_between_entry_and_rr_caps():
    side, entry, tp_rr = SHORT
    assert capped_tp(side, entry, tp_rr, 1866.49) == (1866.49, "analyst_target")


@pytest.mark.parametrize("target", [1950.0, 1910.0, 1918.78, 1937.01, None])
def test_long_falls_back_to_rr(target):
    side, entry, tp_rr = LONG
    assert capped_tp(side, entry, tp_rr, target) == (tp_rr, "rr")


@pytest.mark.parametrize("target", [1850.0, 1890.0, 1887.72, 1858.86, None])
def test_short_falls_back_to_rr(target):
    side, entry, tp_rr = SHORT
    assert capped_tp(side, entry, tp_rr, target) == (tp_rr, "rr")
