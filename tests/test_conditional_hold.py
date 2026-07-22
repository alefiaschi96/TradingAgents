"""Conditional-hold pass: a HOLD that names a NEARBY trigger level may skip
the market-gate short-circuit so the PM can park it as a conditional entry.

Born from the July 2026 paper-sim logs: 42% of HOLD reports carried an
explicit "unless price reclaims X" style trigger — an actionable plan under
the OCO executor that the gate was throwing away. The pass is env-gated
(CONDITIONAL_HOLDS), needs an explicit level in the report, and an ATR
distance filter (fail closed on any data problem).
"""

import pytest

import tradingagents.graph.setup as gs
from tradingagents.agents.utils.market_lean import conditional_trigger_levels
from tradingagents.dataflows.config import get_config, set_config

# --- real phrasings observed verbatim in the paper-sim reports --------------

HOLD_NEAR = (
    "FINAL TRANSACTION PROPOSAL: **HOLD**\n\n"
    "ETH-USD is still constructive intraday, but the edge has faded. "
    "I'd avoid initiating new longs here unless ETH reclaims **1898-1900** "
    "with follow-through. Reclaiming and holding above **1900.5** would "
    "revive the bullish continuation case."
)
HOLD_NO_LEVEL = (
    "FINAL TRANSACTION PROPOSAL: **HOLD**\n\n"
    "Mixed signals; no clear directional edge for the next 1-2 hours."
)


# ------------------------------------------------- conditional_trigger_levels

def test_extracts_levels_from_real_phrasings():
    assert conditional_trigger_levels(HOLD_NEAR) == [1898.0, 1900.5]


@pytest.mark.parametrize(("text", "expected"), [
    ("I would not short aggressively unless price reclaims **1907.44**", [1907.44]),
    ("wait for either a pullback toward 1912 or acceptance above 1906.86", [1912.0, 1906.86]),
    ("stay flat unless price reclaims 1933-1936 or clears the band", [1933.0]),
    ("a 'wait for confirmation' setup: only above a clean break of 1,926.50", [1926.5]),
    ("rejects the upper band and rejection of 1938.60 confirms", [1938.6]),
])
def test_more_observed_phrasings(text, expected):
    assert conditional_trigger_levels(text) == expected


def test_no_level_reports_return_empty():
    assert conditional_trigger_levels(HOLD_NO_LEVEL) == []
    assert conditional_trigger_levels("") == []
    assert conditional_trigger_levels(None) == []
    # plain support/resistance mentions are NOT triggers
    assert conditional_trigger_levels("support at 1890, resistance at 1910") == []


def test_low_priced_asset_levels_parse():
    # ADA-style prices must work too: the pattern is not ETH-specific
    assert conditional_trigger_levels("flat unless price reclaims 0.1642") == [0.1642]


# --------------------------------------------------------------- the router

@pytest.fixture
def _gate_config():
    """Intraday config with the gate on; restore whatever was set after."""
    before = get_config()
    cfg = dict(before)
    cfg.update(
        market_gate_short_circuit=True,
        conditional_holds=True,
        conditional_max_atr=2.0,
        intraday=True,
        analysis_symbol="ETH-USD",
    )
    set_config(cfg)
    yield cfg
    set_config(before)


def _route(report):
    return gs._market_gate_router("continue")({"market_report": report})


@pytest.mark.unit
def test_near_trigger_passes_through(_gate_config, monkeypatch):
    monkeypatch.setattr(gs, "_intraday_price_atr", lambda c: (1894.66, 5.0))
    # trigger 1898 is 3.34 away, within 2 x ATR 5.0
    assert _route(HOLD_NEAR) == "continue"


@pytest.mark.unit
def test_far_trigger_still_gates(_gate_config, monkeypatch):
    monkeypatch.setattr(gs, "_intraday_price_atr", lambda c: (1860.0, 5.0))
    # trigger 1898 is 38 away >> 2 x ATR: short-circuit stands
    assert _route(HOLD_NEAR) == gs.MARKET_GATE_NODE


@pytest.mark.unit
def test_flag_off_keeps_old_behaviour(_gate_config, monkeypatch):
    cfg = dict(get_config())
    cfg["conditional_holds"] = False
    set_config(cfg)
    monkeypatch.setattr(gs, "_intraday_price_atr", lambda c: (1894.66, 5.0))
    assert _route(HOLD_NEAR) == gs.MARKET_GATE_NODE


@pytest.mark.unit
def test_no_level_hold_still_gates(_gate_config, monkeypatch):
    monkeypatch.setattr(gs, "_intraday_price_atr", lambda c: (1894.66, 5.0))
    assert _route(HOLD_NO_LEVEL) == gs.MARKET_GATE_NODE


@pytest.mark.unit
def test_data_failure_fails_closed(_gate_config, monkeypatch):
    def _boom(c):
        raise RuntimeError("kraken down")
    monkeypatch.setattr(gs, "_intraday_price_atr", _boom)
    assert _route(HOLD_NEAR) == gs.MARKET_GATE_NODE


@pytest.mark.unit
def test_directional_report_never_gated(_gate_config):
    # a directional report continues regardless of the conditional machinery
    assert _route("FINAL TRANSACTION PROPOSAL: **SELL**\nClear breakdown.") == "continue"
