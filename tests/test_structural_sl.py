"""Offline tests for the structural (soft/hard) stop-loss feature.

No network: KrakenClient is stubbed and every price/candle feed is injected.
Covers the plan's test matrix (docs/structural-sl-plan.md §9): contract
parsing, cascade validation with ATR fallback, the invalidation entry gate,
the 15m confirmation machinery (straddling bar, consecutive-reset, catch-up
after restart), basis translation, hard-distance sizing, and the flag-off
byte-identical guarantee.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from paper_sim import simulator
from paper_sim.simulator import (
    normalize_execution_plan,
    parse_execution_plan,
    resolve_structural_levels,
)


class _StubKraken:
    def __init__(self, *args, **kwargs):
        pass

    def connect(self):
        pass

    def get_last_price(self):
        return 100.0

    def amount_for_notional(self, notional, price):
        return notional / price


def make_sim(tmp_path, monkeypatch, *, structural=True, events=None, **kwargs):
    monkeypatch.setattr(simulator, "KrakenClient", _StubKraken)
    cfg = SimpleNamespace(
        symbol="PF_SOLUSD",
        analysis_symbol="SOL-USD",
        balance_pct=0.5,
        leverage=5.0,
        take_profit_rr=2.0,
        stop_mode="atr",
        stop_pct=0.5,
        stop_atr_mult=1.0,
        atr_period=14,
        atr_timeframe="15m",
    )
    sink = None
    if events is not None:
        sink = lambda kind, **kw: events.append((kind, kw))  # noqa: E731
    sim = simulator.PaperSimulator(
        cfg,
        state_path=str(tmp_path / "state.json"),
        start_equity=1000.0,
        fee_pct_per_side=0.0,
        decision_interval_min=60.0,
        slippage_pct_per_side=0.0,
        event_sink=sink,
        structural_sl=structural,
        **kwargs,
    )
    sim.last_price = lambda: 100.0
    sim._atr = lambda: 2.0
    sim._spot_price = lambda: 100.0  # ratio 1 unless a test overrides
    sim.minute_range = lambda: (100.5, 99.5)
    return sim


def make_plan(**overrides):
    plan = {
        "invalidation_level": 99.0,
        "invalidation_semantics": "hold",
        "confirm_bars": 2,
        "hard_level": 97.0,
        "price_target": 110.0,
        "horizon_minutes": 120,
    }
    plan.update(overrides)
    return plan


def bars_15m(*specs):
    """OHLCV rows [ts_ms, o, h, l, c, v] from (ts_seconds, close) pairs."""
    return [[ts * 1000.0, c, c + 0.2, c - 0.2, c, 10.0] for ts, c in specs]


_DEFAULT = object()


def open_structural(sim, plan=_DEFAULT, side="buy"):
    if plan is _DEFAULT:
        plan = make_plan()
    sim.open_position(side, "Buy" if side == "buy" else "Sell", exec_plan=plan)
    return sim.state["open"]


# ---------------------------------------------------------------- contract

def test_parse_execution_plan_roundtrip():
    payload = make_plan()
    md = f"**Rating**: Buy\n\n**Execution Plan**:\n```json\n{json.dumps(payload)}\n```"
    assert parse_execution_plan(md) == payload
    assert parse_execution_plan("**Rating**: Buy — no block here") is None
    assert parse_execution_plan("**Execution Plan**:\n```json\nnot json\n```") is None


def test_normalize_rejects_bad_fields():
    assert normalize_execution_plan({"invalidation_semantics": "hold"})[1]
    assert normalize_execution_plan(make_plan(invalidation_semantics="weird"))[1]
    assert normalize_execution_plan(make_plan(confirm_bars=7))[1]
    assert normalize_execution_plan(make_plan(hard_level="soon"))[1]
    norm, err = normalize_execution_plan(make_plan(confirm_bars=None))
    assert err is None and norm["confirm_bars"] == 2  # hold default
    norm, _ = normalize_execution_plan(
        make_plan(invalidation_semantics="close", confirm_bars=None)
    )
    assert norm["confirm_bars"] == 1  # close default


def test_resolve_geometry_cascade():
    norm, _ = normalize_execution_plan(make_plan())
    # already crossed for a long -> veto, not reject
    _, err, vetoed = resolve_structural_levels(dict(norm), "buy", 98.5, 2.0)
    assert vetoed and err is None
    # hard on the wrong side of soft -> reject
    bad = dict(norm, hard=99.5)
    _, err, vetoed = resolve_structural_levels(bad, "buy", 100.0, 2.0)
    assert err and not vetoed
    # hard distance beyond 3xATR -> reject
    far = dict(norm, hard=80.0)
    _, err, _ = resolve_structural_levels(far, "buy", 100.0, 2.0)
    assert err and "outside" in err
    # missing hard -> ATR buffer, scaled by confirm_bars (0.4 * 2 * 2 bars)
    auto = dict(norm, hard=None)
    levels, err, _ = resolve_structural_levels(auto, "buy", 100.0, 2.0)
    assert err is None
    assert levels["hard"] == 99.0 - 0.4 * 2.0 * 2


# ---------------------------------------------------------------- entry path

def test_open_arms_structural_levels(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    pos = open_structural(sim)
    st = pos["structural"]
    assert (st["soft"], st["hard"]) == (99.0, 97.0)
    assert pos["sl"] == 97.0  # hard IS the touch stop
    assert pos["stop_pct"] == 3.0  # sizing runs on the hard distance
    assert pos["tp"] == 106.0  # rr 2 x hard distance
    open_ev = [e for e in events if e[0] == "open"][0][1]
    assert open_ev["sl_mode"] == "structural"


def test_already_invalidated_entry_is_skipped(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    pos = open_structural(sim, make_plan(invalidation_level=101.0))
    assert pos is None
    vetoes = [e for e in events if e[0] == "veto"]
    assert vetoes and vetoes[0][1]["gate"] == "invalidation"


def test_invalid_block_falls_back_to_atr(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    pos = open_structural(sim, make_plan(hard_level=99.5))  # not beyond soft
    assert pos is not None and "structural" not in pos
    assert pos["stop_pct"] == 2.0  # ATR stop: 1.0 x ATR(2.0) / 100
    assert [e for e in events if e[0] == "structural_rejected"]


def test_missing_block_falls_back_to_atr(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    pos = open_structural(sim, plan={})
    assert pos is not None and "structural" not in pos
    assert [e for e in events if e[0] == "structural_rejected"]


def test_basis_ratio_translates_all_levels(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch)
    sim._spot_price = lambda: 50.0  # perp 100 / spot 50 -> ratio 2
    plan = make_plan(
        invalidation_level=49.0, hard_level=48.0, price_target=54.0
    )
    pos = open_structural(sim, plan)
    st = pos["structural"]
    assert st["ratio"] == 2.0
    assert (st["soft"], st["hard"]) == (98.0, 96.0)
    # translated target (108) caps the TP inside the rr level (100 + 2x4 = 108)
    assert pos["tp"] == 108.0


def test_spot_failure_degrades_to_ratio_one(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch)
    sim._spot_price = lambda: None
    pos = open_structural(sim)
    assert pos["structural"]["ratio"] == 1.0
    assert pos["structural"]["soft"] == 99.0


def test_target_gate_runs_on_hard_distance(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events, min_analyst_rr=1.5)
    # hard distance 3% of entry; target 103 implies 1.0x -> veto
    pos = open_structural(sim, make_plan(price_target=103.0))
    assert pos is None
    assert [e for e in events if e[0] == "veto" and e[1]["gate"] == "target"]


def test_sizing_runs_on_hard_distance(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch, risk_pct_per_trade=2.0)
    pos = open_structural(sim)
    # equity 1000 * 2% / 3% stop = 666.67 notional, under the 2500 cap
    assert abs(pos["notional"] - 1000.0 * 2.0 / 3.0) < 1e-6


def test_touch_semantics_single_stop(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch)
    # declared hard differs -> ignored with a warning, soft IS the stop
    plan = make_plan(invalidation_semantics="touch", invalidation_level=98.0,
                     hard_level=95.0)
    pos = open_structural(sim, plan)
    st = pos["structural"]
    assert st["soft"] == st["hard"] == 98.0 == pos["sl"]
    sim.minute_range = lambda: (100.0, 97.9)  # touch the level
    sim.monitor()
    assert sim.state["open"] is None
    assert sim.state["closed"][-1]["outcome"] == "SL"


# ---------------------------------------------------------------- monitor

def test_straddling_bar_never_counts(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch)
    pos = open_structural(sim, make_plan(invalidation_semantics="close",
                                         confirm_bars=1))
    now = time.time()
    pos["structural"]["opened_ts"] = now - 1000.0
    # one FINAL bar closing beyond the soft, but it opened before the entry
    sim._fetch_ohlcv = lambda tf, limit: bars_15m((now - 1200.0, 98.5))
    sim.monitor()
    assert sim.state["open"] is not None
    assert pos["structural"]["count"] == 0


def test_close_confirms_on_first_full_bar(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch)
    pos = open_structural(sim, make_plan(invalidation_semantics="close",
                                         confirm_bars=1))
    now = time.time()
    pos["structural"]["opened_ts"] = now - 2000.0
    sim._fetch_ohlcv = lambda tf, limit: bars_15m(
        (now - 2200.0, 98.5),  # straddling: ignored
        (now - 1000.0, 98.5),  # full bar beyond -> confirm
    )
    sim.last_price = lambda: 98.4
    sim.monitor()
    assert sim.state["open"] is None
    closed = sim.state["closed"][-1]
    assert closed["outcome"] == "SOFT"
    assert closed["exit"] == 98.4


def test_hold_requires_consecutive_closes(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    pos = open_structural(sim)  # hold / 2 bars
    now = time.time()
    st = pos["structural"]
    st["opened_ts"] = now - 6000.0
    sim._fetch_ohlcv = lambda tf, limit: bars_15m(
        (now - 5000.0, 98.5),  # beyond -> 1
        (now - 4000.0, 99.5),  # back inside -> reset
        (now - 3000.0, 98.7),  # beyond -> 1
        (now - 2000.0, 98.6),  # beyond -> 2 -> confirmed
    )
    sim.last_price = lambda: 98.5
    sim.monitor()
    assert sim.state["open"] is None
    closed = sim.state["closed"][-1]
    assert closed["outcome"] == "SOFT"
    assert closed["structural"]["count"] == 2
    close_ev = [e for e in events if e[0] == "close"][-1][1]
    assert close_ev["soft_delay"] == 98.5 - 99.0


def test_wick_beyond_soft_saves_and_emits(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    pos = open_structural(sim)
    sim.minute_range = lambda: (100.0, 98.8)  # wick through soft, above hard
    sim._fetch_ohlcv = lambda tf, limit: []   # no closed bar yet
    sim.monitor()
    assert sim.state["open"] is not None      # the old stop would have exited
    assert pos["structural"]["soft_touched"] is True
    assert [e for e in events if e[0] == "soft_touched_not_confirmed"]
    sim.monitor()  # flagged once, not per tick
    assert len([e for e in events if e[0] == "soft_touched_not_confirmed"]) == 1


def test_hard_touch_beats_confirmation_window(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    pos = open_structural(sim)
    pos["structural"]["count"] = 1  # mid-confirmation
    sim.minute_range = lambda: (100.0, 96.9)  # through the hard
    sim.monitor()
    closed = sim.state["closed"][-1]
    assert closed["outcome"] == "SL"
    assert closed["exit"] == 97.0
    close_ev = [e for e in events if e[0] == "close"][-1][1]
    assert close_ev["hard_hit_direct"] is False  # one confirm had accrued


def test_restart_resumes_and_catches_up(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch)
    pos = open_structural(sim)
    now = time.time()
    st = pos["structural"]
    st.update(opened_ts=now - 9000.0, count=1,
              last_eval_bar_ts=now - 5000.0)
    sim.save()

    sim2 = make_sim(tmp_path, monkeypatch)
    pos2 = sim2.state["open"]
    st2 = pos2["structural"]
    assert st2["count"] == 1 and st2["last_eval_bar_ts"] == now - 5000.0
    # two bars closed while the daemon was down: already-evaluated bars are
    # skipped, the missed beyond-close completes the hold-2 confirmation
    sim2._fetch_ohlcv = lambda tf, limit: bars_15m(
        (now - 6000.0, 98.5),  # before last_eval: already counted
        (now - 3000.0, 98.5),  # missed while down -> count 2 -> confirmed
    )
    sim2.last_price = lambda: 98.3
    sim2.monitor()
    assert sim2.state["open"] is None
    assert sim2.state["closed"][-1]["outcome"] == "SOFT"


def test_horizon_overrides_time_stop(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch, time_stop_hours=6.0)
    pos = open_structural(sim, make_plan(horizon_minutes=30))
    opened = datetime.now(timezone.utc) - timedelta(hours=1.5)
    pos["opened_at"] = opened.isoformat()
    sim._fetch_ohlcv = lambda tf, limit: []
    sim.monitor()  # 1.5h > 2 x 30m -> TIME, despite the 6h profile default
    assert sim.state["closed"][-1]["outcome"] == "TIME"


# ---------------------------------------------------------------- flag off

def test_flag_off_is_todays_behaviour(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch, structural=False)
    sim.open_position("buy", "Buy", analyst_target=110.0)
    pos = sim.state["open"]
    assert "structural" not in pos
    assert pos["stop_pct"] == 2.0  # plain ATR stop


def test_render_and_prompt_only_change_with_flag():
    from tradingagents.agents.schemas import (
        ExecutionPlan,
        PortfolioDecision,
        PortfolioDecisionStructural,
        render_pm_decision,
    )

    base = dict(rating="Buy", executive_summary="s", investment_thesis="t",
                price_target=110.0)
    assert "Execution Plan" not in render_pm_decision(PortfolioDecision(**base))
    plan = ExecutionPlan(invalidation_level=99.0, invalidation_semantics="hold",
                         confirm_bars=2, hard_level=97.0, horizon_minutes=120)
    md = render_pm_decision(
        PortfolioDecisionStructural(**base, execution_plan=plan)
    )
    parsed = parse_execution_plan(md)
    assert parsed == {
        "invalidation_level": 99.0,
        "invalidation_semantics": "hold",
        "confirm_bars": 2,
        "hard_level": 97.0,
        "price_target": 110.0,
        "horizon_minutes": 120,
    }
