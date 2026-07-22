"""Early analysis trigger on regime transitions.

The periodic cooldown (decision_interval) remains the re-check cadence for a
standing trend, but the moment the regime transitions into a directional state
(chop→up/down, or an up↔down flip) the analysis fires early, rate-limited by
analysis_min_gap so regime flip-flop can't spam LLM runs. Pure logic tests: the
regime read is stubbed, no network and no LLM.
"""

import pytest

from paper_sim.simulator import PaperSimulator


def _sim(*, last_decision_at, interval_min=45.0, gap_min=30.0):
    """Bare simulator with only the fields should_decide touches."""
    sim = PaperSimulator.__new__(PaperSimulator)
    sim.decision_interval = interval_min * 60.0
    sim.analysis_min_gap = gap_min * 60.0
    sim.state = {"last_decision_at": last_decision_at}
    sim._last_regime_log = None
    sim._prev_regime = None
    return sim


def _stub_regime(sim, ready, regime, reason="stub"):
    sim._regime_ready = lambda: (ready, regime, reason)


NOW = 100_000.0
MIN = 60.0


@pytest.mark.unit
def test_periodic_path_still_fires_when_cooldown_elapsed():
    sim = _sim(last_decision_at=NOW - 46 * MIN)  # interval 45min elapsed
    _stub_regime(sim, True, "up")
    assert sim.should_decide(NOW) is True


@pytest.mark.unit
def test_standing_trend_waits_for_cooldown():
    sim = _sim(last_decision_at=NOW - 40 * MIN)  # cooldown running
    sim._prev_regime = "up"                       # same trend as before: no news
    _stub_regime(sim, True, "up")
    assert sim.should_decide(NOW) is False


@pytest.mark.unit
def test_transition_bypasses_cooldown():
    sim = _sim(last_decision_at=NOW - 31 * MIN)  # cooldown running, gap elapsed
    sim._prev_regime = "chop"
    _stub_regime(sim, True, "up")
    assert sim.should_decide(NOW) is True        # chop -> up fires early
    assert sim._prev_regime == "up"              # transition consumed


@pytest.mark.unit
def test_direction_flip_counts_as_transition():
    sim = _sim(last_decision_at=NOW - 31 * MIN)
    sim._prev_regime = "up"
    _stub_regime(sim, True, "down")
    assert sim.should_decide(NOW) is True


@pytest.mark.unit
def test_transition_is_rate_limited_but_not_lost():
    sim = _sim(last_decision_at=NOW - 10 * MIN)  # only 10min since last analysis
    sim._prev_regime = "chop"
    _stub_regime(sim, True, "up")
    assert sim.should_decide(NOW) is False       # inside the 30min gap: blocked
    assert sim._prev_regime == "chop"            # baseline kept: still pending
    later = NOW + 21 * MIN                       # gap now elapsed (31min)
    assert sim.should_decide(later) is True      # pending transition fires
    assert sim._prev_regime == "up"


@pytest.mark.unit
def test_not_ready_resets_baseline_so_next_trend_is_fresh():
    sim = _sim(last_decision_at=NOW - 31 * MIN)
    sim._prev_regime = "up"
    _stub_regime(sim, False, "chop", "chop (no clear HTF trend)")
    assert sim.should_decide(NOW) is False
    assert sim._prev_regime == "chop"            # up -> chop recorded
    _stub_regime(sim, True, "up")                # trend re-forms
    assert sim.should_decide(NOW + MIN) is True  # counts as a fresh transition


@pytest.mark.unit
def test_first_tick_with_cooldown_running_treats_trend_as_transition():
    # Restart mid-cooldown into a standing trend: prev is unknown (None), so it
    # fires as a transition once the min gap from the persisted timestamp is ok.
    sim = _sim(last_decision_at=NOW - 31 * MIN)
    _stub_regime(sim, True, "up")
    assert sim.should_decide(NOW) is True
