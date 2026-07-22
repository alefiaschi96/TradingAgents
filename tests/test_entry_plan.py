"""Conditional entries: the PM's **Entry Plan** drives pending OCO orders.

Chain under test, all pure or stub-driven (no network, no LLM):

  render_pm_decision  →  parse_entry_plan  →  validate_legs  →  check_entry
                                            →  pending lifecycle (fill/expire)
                                            →  SL widened to plan invalidation

The rendered markdown is the interchange contract (same pattern as the
**Price Target** field tested in test_tp_cap.py): the paper-sim parses back
exactly what render_entry_leg produced, never LLM prose.
"""

import json

import pytest

from paper_sim.simulator import (
    PaperSimulator,
    check_entry,
    parse_entry_plan,
    validate_legs,
)
from tradingagents.agents.schemas import (
    EntryLeg,
    PortfolioDecision,
    PortfolioRating,
    render_pm_decision,
)

# Levels from the real 2026-07-17 17:40 UTC trade (the SL loss the feature
# is built to prevent): buy plan around ref 1843.94.
PULL = {"kind": "pullback", "zone_low": 1837.0, "zone_high": 1840.0,
        "trigger": None, "target": 1849.5, "invalidation": 1832.0}
BRK = {"kind": "breakout", "zone_low": None, "zone_high": None,
       "trigger": 1850.0, "target": 1858.5, "invalidation": 1846.0}
REF = 1843.94


def _decision(**kwargs) -> PortfolioDecision:
    return PortfolioDecision(
        rating=PortfolioRating.OVERWEIGHT,
        executive_summary="summary",
        investment_thesis="thesis",
        **kwargs,
    )


# -------------------------------------------------------- render → parse

def test_roundtrip_render_then_parse():
    md = render_pm_decision(_decision(
        price_target=1858.5,
        time_horizon="1-2 hours",
        entry_legs=[
            EntryLeg(kind="pullback", zone_low=1837.0, zone_high=1840.0,
                     price_target=1849.5, invalidation=1832.0),
            EntryLeg(kind="breakout", trigger=1850.0, price_target=1858.5,
                     invalidation=1846.0),
        ],
    ))
    assert parse_entry_plan(md) == [PULL, BRK]


def test_no_entry_plan_block_is_empty():
    assert parse_entry_plan("**Rating**: Hold\n\nNo view.") == []
    assert parse_entry_plan("") == []
    assert parse_entry_plan(None) == []


def test_zone_bounds_are_sorted_on_render():
    md = render_pm_decision(_decision(entry_legs=[
        EntryLeg(kind="pullback", zone_low=1840.0, zone_high=1837.0),
    ]))
    (leg,) = parse_entry_plan(md)
    assert (leg["zone_low"], leg["zone_high"]) == (1837.0, 1840.0)


def test_unusable_legs_are_skipped():
    md = (
        "**Entry Plan**:\n"
        "- pullback | target 1849.5\n"        # no zone: unusable
        "- breakout | zone 1-2\n"             # no trigger: unusable
        "- sideways | trigger 10\n"           # unknown kind
        "- breakout | trigger 1,850.5\n"      # usable (comma tolerated)
    )
    legs = parse_entry_plan(md)
    assert len(legs) == 1
    assert legs[0]["kind"] == "breakout" and legs[0]["trigger"] == 1850.5


def test_block_ends_at_next_section():
    md = (
        "**Entry Plan**:\n"
        "- breakout | trigger 1850.0\n"
        "\n"
        "**Note**: prose mentioning trigger 999 must not become a leg\n"
        "- pullback | zone 1-2\n"
    )
    legs = parse_entry_plan(md)
    assert len(legs) == 1 and legs[0]["trigger"] == 1850.0


# ------------------------------------------------------------ validate_legs

def test_valid_buy_plan_passes_unchanged():
    legs, dropped = validate_legs("buy", [PULL, BRK], REF)
    assert legs == [PULL, BRK] and dropped == []


def test_buy_pullback_zone_above_price_is_dropped():
    bad = {**PULL, "zone_low": 1845.0, "zone_high": 1848.0}
    legs, dropped = validate_legs("buy", [bad], REF)
    assert legs == [] and "retracement side" in dropped[0]


def test_buy_breakout_trigger_below_price_is_dropped():
    bad = {**BRK, "trigger": 1840.0}
    legs, dropped = validate_legs("buy", [bad], REF)
    assert legs == [] and "not beyond" in dropped[0]


def test_wrong_side_invalidation_is_nulled_but_leg_kept():
    bad = {**PULL, "invalidation": 1841.0}  # above the zone: nonsense for a buy
    (leg,), dropped = validate_legs("buy", [bad], REF)
    assert leg["invalidation"] is None and "stop side" in dropped[0]


def test_wrong_side_target_is_nulled_but_leg_kept():
    bad = {**BRK, "target": 1845.0}  # below the trigger: nonsense for a buy
    (leg,), dropped = validate_legs("buy", [bad], REF)
    assert leg["target"] is None and "profit side" in dropped[0]


def test_duplicate_kind_keeps_first():
    legs, dropped = validate_legs("buy", [BRK, {**BRK, "trigger": 1855.0}], REF)
    assert len(legs) == 1 and legs[0]["trigger"] == 1850.0
    assert "duplicate" in dropped[0]


def test_sell_plan_mirrors():
    pull = {**PULL, "zone_low": 1848.0, "zone_high": 1851.0,
            "invalidation": 1856.0, "target": 1840.0}
    brk = {**BRK, "trigger": 1837.0, "invalidation": 1841.0, "target": 1830.0}
    legs, dropped = validate_legs("sell", [pull, brk], REF)
    assert len(legs) == 2 and dropped == []


# -------------------------------------------------------------- check_entry

def test_buy_pullback_fills_at_zone_edge():
    leg, fill = check_entry("buy", [PULL, BRK], 1843.0, 1843.5, 1839.8, 1841.0)
    assert leg["kind"] == "pullback" and fill == 1840.0


def test_buy_breakout_fills_at_trigger():
    leg, fill = check_entry("buy", [PULL, BRK], 1848.0, 1851.2, 1847.5, 1850.8)
    assert leg["kind"] == "breakout" and fill == 1850.0


def test_no_touch_waits():
    assert check_entry("buy", [PULL, BRK], 1843.0, 1844.0, 1841.5, 1842.0) == (None, None)


def test_both_hit_rising_candle_prefers_pullback():
    # open→low→high assumed on a rising candle: the retracement traded first
    leg, fill = check_entry("buy", [PULL, BRK], 1841.0, 1851.0, 1839.0, 1850.5)
    assert leg["kind"] == "pullback" and fill == 1840.0


def test_both_hit_falling_candle_prefers_breakout():
    # open→high→low assumed on a falling candle: the breakout traded first
    leg, fill = check_entry("buy", [PULL, BRK], 1849.0, 1851.0, 1839.0, 1840.5)
    assert leg["kind"] == "breakout" and fill == 1850.0


def test_sell_mirror_fills():
    pull = {**PULL, "zone_low": 1848.0, "zone_high": 1851.0, "invalidation": None}
    brk = {**BRK, "trigger": 1837.0}
    leg, fill = check_entry("sell", [pull, brk], 1844.0, 1848.5, 1843.0, 1845.0)
    assert leg["kind"] == "pullback" and fill == 1848.0
    leg, fill = check_entry("sell", [pull, brk], 1840.0, 1841.0, 1836.5, 1837.5)
    assert leg["kind"] == "breakout" and fill == 1837.0


# ------------------------------------------- SL widened to plan invalidation

class _Cfg:
    entry_max_stop_widen = 2.0


def _bare_sim() -> PaperSimulator:
    sim = PaperSimulator.__new__(PaperSimulator)
    sim.cfg = _Cfg()
    return sim


@pytest.mark.unit
def test_sl_stays_atr_without_invalidation():
    sim = _bare_sim()
    assert sim._sl_with_invalidation("buy", 1840.0, 1831.0, None) == (1831.0, 1.0, "atr")
    leg_without_inval = {**PULL, "invalidation": None}
    assert sim._sl_with_invalidation("buy", 1840.0, 1831.0, leg_without_inval) == (
        1831.0, 1.0, "atr",
    )


@pytest.mark.unit
def test_sl_never_tightens_to_nearer_invalidation():
    sim = _bare_sim()
    # ATR stop 1831 is already beyond invalidation 1832: keep the ATR stop.
    sl, scale, src = sim._sl_with_invalidation("buy", 1840.0, 1831.0, PULL)
    assert (sl, scale, src) == (1831.0, 1.0, "atr")


@pytest.mark.unit
def test_sl_widens_to_invalidation_and_scales_size():
    sim = _bare_sim()
    # entry 1840, ATR stop 1835 (dist 5), invalidation 1832 (dist 8) < cap 10
    sl, scale, src = sim._sl_with_invalidation("buy", 1840.0, 1835.0, PULL)
    assert sl == 1832.0 and src == "invalidation"
    assert scale == pytest.approx(5.0 / 8.0)


@pytest.mark.unit
def test_sl_widening_is_capped():
    sim = _bare_sim()
    # entry 1840, ATR stop 1838 (dist 2), invalidation 1832 (dist 8) > cap 4
    sl, scale, src = sim._sl_with_invalidation("buy", 1840.0, 1838.0, PULL)
    assert sl == 1836.0 and src == "invalidation_capped"
    assert scale == pytest.approx(0.5)


@pytest.mark.unit
def test_sl_widens_for_sell_side_too():
    sim = _bare_sim()
    leg = {"invalidation": 1856.0}
    sl, scale, src = sim._sl_with_invalidation("sell", 1848.0, 1852.0, leg)
    assert sl == 1856.0 and src == "invalidation"
    assert scale == pytest.approx(0.5)


# ------------------------------------------------------- pending lifecycle

class _PendCfg:
    symbol = "PF_ETHUSD"
    entry_ttl_min = 120.0


NOW = 1_000_000.0


def _pending_sim(candle, *, expires_in_min=60.0):
    """Bare simulator with a parked buy plan and a scripted next 1m candle."""
    sim = PaperSimulator.__new__(PaperSimulator)
    sim.cfg = _PendCfg()
    sim.events = []
    sim._emit_event = lambda ev, **f: sim.events.append((ev, f))
    sim.minute_candle = lambda: candle
    sim.state = {
        "pending": {
            "side": "buy", "rating": "Overweight", "legs": [PULL, BRK],
            "analyst_target": 1858.5, "ref_price": REF,
            "created_at": "2026-07-17T17:40:00+00:00",
            "created_ts": NOW - 300.0,
            "expires_ts": NOW + expires_in_min * 60.0,
        },
        "open": None,
    }
    sim.opened = []

    def _fake_open(side, rating, analyst_target=None, *, entry_kind="market",
                   leg=None, trigger_price=None):
        sim.opened.append({"side": side, "entry_kind": entry_kind, "leg": leg,
                           "trigger_price": trigger_price})
        # far-away bracket: the same-candle flush check must not trip
        sim.state["open"] = {"side": side, "sl": 1000.0, "tp": 3000.0}

    sim.open_position = _fake_open
    return sim


@pytest.mark.unit
def test_pending_waits_when_nothing_touches():
    sim = _pending_sim((1843.0, 1844.0, 1841.5, 1842.0))
    sim.check_pending(NOW)
    assert sim.has_pending() and sim.opened == []


@pytest.mark.unit
def test_pending_fill_opens_with_leg_and_clears_plan():
    sim = _pending_sim((1843.0, 1843.5, 1839.8, 1841.0))
    sim.check_pending(NOW)
    assert not sim.has_pending()
    assert sim.opened == [{"side": "buy", "entry_kind": "pullback",
                           "leg": PULL, "trigger_price": 1840.0}]
    assert [e for e, _ in sim.events] == ["pending_fill"]


@pytest.mark.unit
def test_pending_expires_without_fill():
    sim = _pending_sim((1843.0, 1844.0, 1841.5, 1842.0), expires_in_min=-1.0)
    sim.check_pending(NOW)
    assert not sim.has_pending() and sim.opened == []
    assert [e for e, _ in sim.events] == ["pending_expired"]


@pytest.mark.unit
def test_same_candle_flush_closes_immediately():
    # The candle fills the pullback AND runs through the (stubbed) stop.
    sim = _pending_sim((1843.0, 1843.5, 1830.0, 1831.0))
    closed = []
    sim.close_position = lambda price, reason: closed.append((price, reason))

    def _tight_open(side, rating, analyst_target=None, *, entry_kind="market",
                    leg=None, trigger_price=None):
        sim.state["open"] = {"side": side, "sl": 1832.0, "tp": 1849.5}

    sim.open_position = _tight_open
    sim.check_pending(NOW)
    assert closed == [(1832.0, "SL")]


# --------------------------------------------------------- state round-trip

def test_state_file_accepts_missing_pending_and_rejects_bad(tmp_path):
    # Old state files (pre-feature) have no "pending" key: default to None.
    old = tmp_path / "state.json"
    old.write_text(json.dumps({"equity": 100.0, "open": None, "closed": []}))
    state = PaperSimulator._read_state_file(str(old), 100.0)
    assert state["pending"] is None

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"equity": 100.0, "closed": [], "pending": [1, 2]}))
    with pytest.raises(ValueError):
        PaperSimulator._read_state_file(str(bad), 100.0)
