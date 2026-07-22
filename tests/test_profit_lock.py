"""Profit lock: once price trades within PROFIT_LOCK_PCT of the TP, that band
edge becomes a protective floor — a later return to it closes the trade there,
in gain, instead of riding the reversal back to the original stop.

The user scenario this encodes (long, TP 1845, pct 0.2 → floor 1841.31):
price climbs to 0.2% from TP, pushes on to 0.1% from TP, then falls back to
the 0.2% level → the position closes AT the floor, profit collected. A clean
push through the TP still exits at the full target: the lock never caps the
upside, it only cuts the give-back.
"""

import pytest

from paper_sim.simulator import (
    PaperSimulator,
    maybe_arm_profit_lock,
    profit_lock_level,
)

# Long from the real 2026-07-17 16:42 UTC trade: entry 1834.97, TP 1845.
ENTRY, SL, TP, PCT = 1834.97, 1826.12, 1845.0, 0.2
FLOOR = 1845.0 * (1 - 0.002)  # 1841.31


# ---------------------------------------------------------------- pure logic

def test_level_math():
    assert profit_lock_level("buy", TP, PCT) == pytest.approx(FLOOR)
    assert profit_lock_level("sell", 1805.0, PCT) == pytest.approx(1805.0 * 1.002)


def test_disabled_when_pct_zero():
    assert maybe_arm_profit_lock("buy", SL, TP, 0.0, 1846.0, 1830.0) is None


def test_never_rearms():
    assert maybe_arm_profit_lock("buy", FLOOR, TP, PCT, 1844.0, 1842.0,
                                 already=True) is None


def test_arms_when_candle_reaches_band():
    got = maybe_arm_profit_lock("buy", SL, TP, PCT, 1841.5, 1839.0)
    assert got == pytest.approx(FLOOR)


def test_no_arm_below_band():
    assert maybe_arm_profit_lock("buy", SL, TP, PCT, 1841.0, 1839.0) is None


def test_sell_mirror():
    sl, tp = 1832.05, 1805.0
    floor = 1805.0 * 1.002  # 1808.61
    assert maybe_arm_profit_lock("sell", sl, tp, PCT, 1812.0, 1808.0) == pytest.approx(floor)
    assert maybe_arm_profit_lock("sell", sl, tp, PCT, 1812.0, 1809.0) is None


# ------------------------------------------------------------- monitor flow

class _Cfg:
    symbol = "PF_ETHUSD"
    profit_lock_pct = PCT


def _sim(candle):
    """Bare simulator with an open long and a scripted next 1m candle."""
    sim = PaperSimulator.__new__(PaperSimulator)
    sim.cfg = _Cfg()
    sim.fee = 0.05
    sim.slippage = 0.02
    sim.events = []
    sim._emit_event = lambda ev, **f: sim.events.append((ev, f))
    sim.minute_candle = lambda: candle
    sim.state = {
        "equity": 111.81, "pnl_total": 0.0, "wins": 0, "losses": 0,
        "closed": [], "last_decision_at": None, "pending": None,
        "open": {"side": "buy", "rating": "Overweight", "entry": ENTRY,
                 "size": 0.597, "sl": SL, "tp": TP},
    }
    return sim


@pytest.mark.unit
def test_below_band_nothing_happens():
    sim = _sim((1838.0, 1840.0, 1837.0, 1839.5))
    sim.monitor()
    pos = sim.state["open"]
    assert pos["sl"] == SL and not pos.get("lock_armed") and sim.events == []


@pytest.mark.unit
def test_arming_moves_sl_to_floor_and_keeps_position():
    # touches 0.1% from TP (1843.2) and closes above the floor: armed, open
    sim = _sim((1840.0, 1843.2, 1839.5, 1842.0))
    sim.monitor()
    pos = sim.state["open"]
    assert pos["lock_armed"] and pos["sl"] == pytest.approx(FLOOR)
    assert pos["sl_source"] == "profit_lock"
    assert [e for e, _ in sim.events] == ["profit_lock"]


@pytest.mark.unit
def test_return_to_floor_closes_in_gain_as_lock():
    sim = _sim((1842.0, 1842.5, 1841.0, 1841.2))  # armed earlier; now returns
    pos = sim.state["open"]
    pos["sl"], pos["lock_armed"], pos["sl_source"] = FLOOR, True, "profit_lock"
    sim.monitor()
    assert sim.state["open"] is None
    trade = sim.state["closed"][-1]
    assert trade["outcome"] == "LOCK"
    assert trade["pnl"] > 0          # profit collected, not a loss
    assert sim.state["wins"] == 1 and sim.state["losses"] == 0
    # fill = floor minus adverse slippage, like any stop-triggered close
    assert trade["exit"] == pytest.approx(FLOOR * (1 - 0.0002))


@pytest.mark.unit
def test_reversal_inside_arming_minute_closes_at_floor():
    # touches the band (high 1843.0) but the minute closes back below the
    # floor: pessimistic read = peak first, drop after -> collect at floor now
    sim = _sim((1840.5, 1843.0, 1840.0, 1840.8))
    sim.monitor()
    assert sim.state["open"] is None
    trade = sim.state["closed"][-1]
    assert trade["outcome"] == "LOCK" and trade["pnl"] > 0
    assert [e for e, _ in sim.events][0] == "profit_lock"


@pytest.mark.unit
def test_clean_push_through_still_exits_at_full_tp():
    sim = _sim((1842.0, 1845.4, 1841.9, 1845.1))  # straight through the TP
    sim.monitor()
    trade = sim.state["closed"][-1]
    assert trade["outcome"] == "TP"
    assert trade["exit_trigger"] == TP  # full target, lock never capped it


@pytest.mark.unit
def test_armed_position_can_still_reach_tp():
    sim = _sim((1843.0, 1845.2, 1842.5, 1844.0))  # stays above floor, hits TP
    pos = sim.state["open"]
    pos["sl"], pos["lock_armed"] = FLOOR, True
    sim.monitor()
    assert sim.state["closed"][-1]["outcome"] == "TP"


@pytest.mark.unit
def test_original_sl_still_labelled_sl_when_lock_never_armed():
    sim = _sim((1830.0, 1831.0, 1825.5, 1826.0))  # crashes to the real stop
    sim.monitor()
    trade = sim.state["closed"][-1]
    assert trade["outcome"] == "SL" and sim.state["losses"] == 1
