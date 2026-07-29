"""Tests that the Risk Manager owns SL/TP and the paper sim honours them.

All tests are pure unit tests — no LLM calls, no network, no API keys.
"""

import pytest

from paper_sim.simulator import parse_rm_levels, _TIER_FRACTIONS, capped_tp
from tradingagents.agents.schemas import (
    PortfolioRating,
    PositionSize,
    RiskDecision,
    RiskProfile,
    render_risk_decision,
)


# ---------------------------------------------------------------------------
# parse_rm_levels
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseRmLevels:
    """Parse SL, TP, and position-size tier from Risk Manager markdown."""

    def test_full_decision_text(self):
        text = (
            "**Rating**: Buy\n\n"
            "**Risk Profile**: Standard\n\n"
            "**Stop Loss**: 73.5000\n\n"
            "**Take Profit**: 79.2000\n\n"
            "**Price Target**: 79.2000\n\n"
            "**Position Size**: Full\n\n"
            "**Risk Rationale**: ATR-scaled bracket with strong conviction."
        )
        rm = parse_rm_levels(text)
        assert rm["sl"] == pytest.approx(73.5)
        assert rm["tp"] == pytest.approx(79.2)
        assert rm["size_fraction"] == pytest.approx(0.98)

    def test_half_position(self):
        text = "**Stop Loss**: 100.0\n**Take Profit**: 110.0\n**Position Size**: Half"
        rm = parse_rm_levels(text)
        assert rm["size_fraction"] == pytest.approx(0.49)

    def test_quarter_position(self):
        text = "**Stop Loss**: 100.0\n**Take Profit**: 110.0\n**Position Size**: Quarter"
        rm = parse_rm_levels(text)
        assert rm["size_fraction"] == pytest.approx(0.245)

    def test_none_position(self):
        text = "**Stop Loss**: n/a\n**Take Profit**: n/a\n**Position Size**: None"
        rm = parse_rm_levels(text)
        assert rm["sl"] is None  # "n/a" doesn't match the float regex
        assert rm["tp"] is None
        assert rm["size_fraction"] == pytest.approx(0.0)

    def test_missing_values(self):
        rm = parse_rm_levels("some random text with no markers")
        assert rm["sl"] is None
        assert rm["tp"] is None
        assert rm["size_fraction"] is None

    def test_empty_string(self):
        rm = parse_rm_levels("")
        assert rm["sl"] is None
        assert rm["tp"] is None
        assert rm["size_fraction"] is None

    def test_none_input(self):
        rm = parse_rm_levels(None)
        assert rm["sl"] is None

    def test_case_insensitive_position_size(self):
        text = "**Position Size**: FULL"
        rm = parse_rm_levels(text)
        assert rm["size_fraction"] == pytest.approx(0.98)


# ---------------------------------------------------------------------------
# Tier fractions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTierFractions:
    """The tier map covers all expected tiers and sums to sensible values."""

    def test_all_tiers_present(self):
        for tier in ("full", "half", "quarter", "none"):
            assert tier in _TIER_FRACTIONS

    def test_full_is_below_one(self):
        assert 0 < _TIER_FRACTIONS["full"] < 1.0

    def test_half_is_roughly_half_of_full(self):
        assert _TIER_FRACTIONS["half"] == pytest.approx(
            _TIER_FRACTIONS["full"] / 2, abs=0.01
        )

    def test_none_is_zero(self):
        assert _TIER_FRACTIONS["none"] == 0.0


# ---------------------------------------------------------------------------
# render_risk_decision includes Risk Profile
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderRiskDecision:
    def test_includes_risk_profile(self):
        d = RiskDecision(
            rating=PortfolioRating.BUY,
            risk_profile=RiskProfile.WIDE,
            stop_loss=73.5,
            take_profit=79.2,
            position_size=PositionSize.FULL,
            risk_rationale="Wide bracket for volatile session.",
        )
        md = render_risk_decision(d)
        assert "**Risk Profile**: Wide" in md
        assert "**Stop Loss**: 73.5" in md
        assert "**Take Profit**: 79.2" in md
        assert "**Position Size**: Full" in md

    def test_hold_decision_renders_na(self):
        d = RiskDecision(
            rating=PortfolioRating.HOLD,
            risk_profile=RiskProfile.STANDARD,
            stop_loss=None,
            take_profit=None,
            position_size=PositionSize.NONE,
            risk_rationale="No trade.",
        )
        md = render_risk_decision(d)
        assert "**Stop Loss**: n/a" in md
        assert "**Take Profit**: n/a" in md

    def test_default_profile_is_standard(self):
        d = RiskDecision(
            rating=PortfolioRating.BUY,
            stop_loss=100.0,
            take_profit=110.0,
            position_size=PositionSize.FULL,
            risk_rationale="Default profile.",
        )
        assert d.risk_profile == RiskProfile.STANDARD

    def test_roundtrip_parse(self):
        """render → parse_rm_levels recovers the numbers."""
        d = RiskDecision(
            rating=PortfolioRating.SELL,
            risk_profile=RiskProfile.TIGHT,
            stop_loss=82.1234,
            take_profit=75.4567,
            position_size=PositionSize.HALF,
            risk_rationale="Tight bracket.",
        )
        md = render_risk_decision(d)
        rm = parse_rm_levels(md)
        assert rm["sl"] == pytest.approx(82.1234)
        assert rm["tp"] == pytest.approx(75.4567)
        assert rm["size_fraction"] == pytest.approx(0.49)


# ---------------------------------------------------------------------------
# RiskProfile schema validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRiskProfileSchema:
    def test_all_profiles_valid(self):
        for profile in RiskProfile:
            d = RiskDecision(
                rating=PortfolioRating.BUY,
                risk_profile=profile,
                stop_loss=100.0,
                take_profit=110.0,
                position_size=PositionSize.FULL,
                risk_rationale="Test.",
            )
            assert d.risk_profile == profile


# ---------------------------------------------------------------------------
# SL/TP tool profile table
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlTpProfiles:
    """Verify the PROFILES constant in sl_tp_tools covers all enum values."""

    def test_profiles_match_enum(self):
        from tradingagents.agents.utils.sl_tp_tools import PROFILES

        for profile in RiskProfile:
            assert profile.value in PROFILES, f"{profile.value} missing from PROFILES"

    def test_profile_values_are_positive(self):
        from tradingagents.agents.utils.sl_tp_tools import PROFILES

        for name, (atr_mult, rr) in PROFILES.items():
            assert atr_mult > 0, f"{name} ATR mult must be positive"
            assert rr > 0, f"{name} RR must be positive"


# ---------------------------------------------------------------------------
# capped_tp still works with RM-provided TP
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCappedTpWithRm:
    def test_analyst_target_caps_rm_tp(self):
        # Long: entry=100, RM TP=110, analyst target=105 → cap to 105
        tp, src = capped_tp("buy", 100.0, 110.0, 105.0)
        assert tp == pytest.approx(105.0)
        assert src == "analyst_target"

    def test_analyst_target_beyond_rm_tp_keeps_rm(self):
        # Long: entry=100, RM TP=110, analyst target=115 → keep 110
        tp, src = capped_tp("buy", 100.0, 110.0, 115.0)
        assert tp == pytest.approx(110.0)
        assert src == "rr"

    def test_no_analyst_target_keeps_rm(self):
        tp, src = capped_tp("buy", 100.0, 110.0, None)
        assert tp == pytest.approx(110.0)
        assert src == "rr"

    def test_short_capping(self):
        # Short: entry=100, RM TP=90, analyst target=95 → cap to 95
        tp, src = capped_tp("sell", 100.0, 90.0, 95.0)
        assert tp == pytest.approx(95.0)
        assert src == "analyst_target"
