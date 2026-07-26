"""Offline tests for the conditional OCO entry feature (ENTRY_MODE=plan).

No network: KrakenClient is stubbed and every price/candle feed is injected.
Covers: leg normalization/validation (geometry + the ATR distance band),
check_entry fill semantics (limit vs stop-entry, same-candle both-legs
heuristic), the park path (per-leg gates, veto vs market fallback, TTL
bounded by the horizon), the pending lifecycle (expiry, hard/soft kills with
dual-track semantics, regime cancel, fill + same-candle flush) and the
flag-off guarantee that ENTRY_MODE=market keeps today's behaviour.
"""

import json
import time
from types import SimpleNamespace

from paper_sim import simulator
from paper_sim.simulator import (
    check_entry,
    normalize_entry_legs,
    validate_legs,
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


def make_sim(tmp_path, monkeypatch, *, events=None, **kwargs):
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
        allow_short=True,
        long_signals={"Buy", "Overweight"},
        short_signals={"Sell", "Underweight"},
    )
    sink = None
    if events is not None:
        sink = lambda kind, **kw: events.append((kind, kw))  # noqa: E731
    kwargs.setdefault("entry_mode", "plan")
    kwargs.setdefault("slippage_pct_per_side", 0.0)
    sim = simulator.PaperSimulator(
        cfg,
        state_path=str(tmp_path / "state.json"),
        start_equity=1000.0,
        fee_pct_per_side=0.0,
        decision_interval_min=60.0,
        event_sink=sink,
        structural_sl=True,
        **kwargs,
    )
    sim.last_price = lambda: 100.0
    sim._atr = lambda: 2.0
    sim._spot_price = lambda: 100.0  # ratio 1 unless a test overrides
    sim._regime_ready = lambda: (True, "off", "filter off")
    sim._fetch_ohlcv = lambda tf, limit: []
    return sim


def make_plan(**overrides):
    plan = {
        "invalidation_level": 94.0,
        "invalidation_semantics": "hold",
        "confirm_bars": 2,
        "hard_level": None,
        "price_target": 110.0,
        "horizon_minutes": 240,
        "entry_legs": [
            {"kind": "pullback", "zone_low": 97.0, "zone_high": 98.0},
            {"kind": "breakout", "trigger": 102.0},
        ],
    }
    plan.update(overrides)
    return plan


def set_candle(sim, o, h, low, c):
    sim.minute_candle = lambda: (o, h, low, c)


def park(sim, plan=None, side="buy", now=None):
    plan = make_plan() if plan is None else plan
    return sim._try_park(
        side, "Buy" if side == "buy" else "Sell", plan, None,
        time.time() if now is None else now,
    )


# ------------------------------------------------------------ pure helpers

def test_normalize_entry_legs_drops_malformed():
    legs = normalize_entry_legs([
        {"kind": "pullback", "zone_low": 98.0, "zone_high": 97.0},  # swapped ok
        {"kind": "pullback", "zone_low": 90.0, "zone_high": 91.0},  # dup kind
        {"kind": "breakout"},                                       # no trigger
        {"kind": "weird", "trigger": 1.0},
        "not a dict",
        {"kind": "breakout", "trigger": "soon"},
    ])
    assert len(legs) == 1
    assert legs[0]["kind"] == "pullback"
    assert (legs[0]["zone_low"], legs[0]["zone_high"]) == (97.0, 98.0)
    assert normalize_entry_legs("nope") == []


def test_validate_legs_geometry():
    legs = normalize_entry_legs([
        {"kind": "pullback", "zone_low": 101.0, "zone_high": 102.0},  # above price
        {"kind": "breakout", "trigger": 99.0},                         # below price
    ])
    valid, reasons = validate_legs("buy", legs, 100.0, 2.0)
    assert valid == [] and len(reasons) == 2


def test_validate_legs_distance_band():
    atr = 2.0
    legs = normalize_entry_legs([
        {"kind": "pullback", "zone_low": 98.0, "zone_high": 99.0},   # 0.5 ATR: too near
        {"kind": "breakout", "trigger": 110.0},                       # 5 ATR: too far
    ])
    valid, reasons = validate_legs("buy", legs, 100.0, atr)
    assert valid == [] and len(reasons) == 2
    legs = normalize_entry_legs([
        {"kind": "pullback", "zone_low": 97.0, "zone_high": 98.0},   # 1 ATR edge
        {"kind": "breakout", "trigger": 102.0},                       # 1 ATR
    ])
    valid, reasons = validate_legs("buy", legs, 100.0, atr)
    assert len(valid) == 2 and reasons == []


def test_check_entry_fills():
    legs = normalize_entry_legs([
        {"kind": "pullback", "zone_low": 97.0, "zone_high": 98.0},
        {"kind": "breakout", "trigger": 102.0},
    ])
    # Neither side touched.
    assert check_entry("buy", legs, 100.0, 100.5, 99.5, 100.2) == (None, None)
    # Wick into the zone: limit fills at the near edge.
    leg, fill = check_entry("buy", legs, 100.0, 100.5, 97.9, 100.2)
    assert leg["kind"] == "pullback" and fill == 98.0
    # Wick through the trigger: stop-entry fires.
    leg, fill = check_entry("buy", legs, 100.0, 102.3, 99.5, 101.0)
    assert leg["kind"] == "breakout" and fill == 102.0
    # Both in one candle: a rising candle went open->low->high, so the
    # retracement leg filled first; a falling candle the reverse.
    leg, _ = check_entry("buy", legs, 100.0, 102.5, 97.5, 101.5)
    assert leg["kind"] == "pullback"
    leg, _ = check_entry("buy", legs, 100.0, 102.5, 97.5, 99.0)
    assert leg["kind"] == "breakout"


def test_check_entry_sell_mirror():
    legs = normalize_entry_legs([
        {"kind": "pullback", "zone_low": 102.0, "zone_high": 103.0},
        {"kind": "breakout", "trigger": 98.0},
    ])
    leg, fill = check_entry("sell", legs, 100.0, 102.4, 99.8, 100.1)
    assert leg["kind"] == "pullback" and fill == 102.0
    leg, fill = check_entry("sell", legs, 100.0, 100.2, 97.8, 99.0)
    assert leg["kind"] == "breakout" and fill == 98.0


# ------------------------------------------------------------------- park

def test_park_creates_pending(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    now = time.time()
    assert park(sim, now=now) == "parked"
    p = sim.state["pending"]
    assert p is not None and len(p["legs"]) == 2
    assert p["side"] == "buy" and p["soft"] == 94.0
    # hold semantics with confirm 2 and no declared hard: auto buffer
    # soft - 0.4 * ATR * confirm = 94 - 1.6.
    assert abs(p["hard"] - 92.4) < 1e-9
    assert [e[0] for e in events] == ["pending"]
    # every leg carries its fill-time contract, spot geometry
    for leg in p["legs"]:
        assert leg["raw_plan"]["invalidation_level"] == 94.0
        assert "entry_legs" not in leg["raw_plan"]


def test_park_ttl_bounded_by_horizon(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch, entry_ttl_min=240.0)
    now = time.time()
    assert park(sim, make_plan(horizon_minutes=120), now=now) == "parked"
    assert abs(sim.state["pending"]["expires_ts"] - (now + 120 * 60.0)) < 1.0


def test_park_without_legs_degrades_to_market(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch)
    assert park(sim, make_plan(entry_legs=None)) == "market"
    assert park(sim, make_plan(entry_legs=[])) == "market"
    assert sim.state["pending"] is None


def test_park_geometry_junk_degrades_to_market(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    plan = make_plan(entry_legs=[
        {"kind": "pullback", "zone_low": 101.0, "zone_high": 102.0},
    ])
    # Zone above price for a buy -> dropped; no survivor on the GEOMETRY
    # step means market fallback, not a veto.
    assert park(sim, plan) == "market"
    assert sim.state["pending"] is None
    assert "leg_dropped" in [e[0] for e in events]


def test_park_gate_veto_stays_flat(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events, min_analyst_rr=1.5)
    # Pullback entry at 98, hard at 92.4 -> stop 5.71%; plan target 103 is
    # only ~0.89x the stop -> the leg (and the lone breakout, worse) dies.
    plan = make_plan(price_target=103.0)
    assert park(sim, plan) == "vetoed"
    assert sim.state["pending"] is None
    vetoes = [e for e in events if e[0] == "leg_vetoed"]
    assert len(vetoes) == 2 and all(e[1]["gate"] == "target" for e in vetoes)


def test_park_per_leg_invalidation_override(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch)
    plan = make_plan(
        hard_level=93.0,
        entry_legs=[
            {"kind": "pullback", "zone_low": 97.0, "zone_high": 98.0},
            {"kind": "breakout", "trigger": 102.0, "invalidation_level": 99.5,
             "price_target": 108.0},
        ],
    )
    assert park(sim, plan) == "parked"
    legs = {leg["kind"]: leg for leg in sim.state["pending"]["legs"]}
    # The pullback leg keeps the plan-level contract.
    assert legs["pullback"]["raw_plan"]["invalidation_level"] == 94.0
    assert legs["pullback"]["raw_plan"]["hard_level"] == 93.0
    # The breakout leg overrides the invalidation; the declared hard (93)
    # is NOT beyond 99.5, so it falls back to the auto buffer.
    assert legs["breakout"]["raw_plan"]["invalidation_level"] == 99.5
    assert legs["breakout"]["raw_plan"]["hard_level"] is None
    assert legs["breakout"]["raw_plan"]["price_target"] == 108.0


def test_park_invalidation_veto(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    # Thesis already dead: invalidation above price for a buy.
    assert park(sim, make_plan(invalidation_level=101.0)) == "vetoed"
    assert [e[0] for e in events if e[0] == "veto"] == ["veto"]


# ---------------------------------------------------------------- pending

def parked_sim(tmp_path, monkeypatch, *, events=None, plan=None, side="buy",
               **kwargs):
    sim = make_sim(tmp_path, monkeypatch, events=events, **kwargs)
    assert park(sim, plan, side=side) == "parked"
    return sim


def test_pending_expiry(tmp_path, monkeypatch):
    events = []
    sim = parked_sim(tmp_path, monkeypatch, events=events)
    set_candle(sim, 100.0, 100.2, 99.8, 100.1)
    sim.check_pending(sim.state["pending"]["expires_ts"] + 1.0)
    assert sim.state["pending"] is None and sim.state["open"] is None
    assert [e[0] for e in events if e[0].startswith("pending_")] == ["pending_expired"]


def test_pending_fill_pullback_no_slippage(tmp_path, monkeypatch):
    events = []
    sim = parked_sim(
        tmp_path, monkeypatch, events=events, slippage_pct_per_side=0.1
    )
    # Basis anchor at fill: keep the ratio at 1 for readable numbers.
    sim._spot_price = lambda: 98.0
    set_candle(sim, 100.0, 100.1, 97.9, 100.0)
    sim.check_pending(time.time())
    pos = sim.state["open"]
    assert sim.state["pending"] is None and pos is not None
    assert pos["entry"] == 98.0  # resting limit: no adverse slippage
    assert pos["entry_kind"] == "pullback"
    assert pos["structural"] is not None
    assert abs(pos["structural"]["soft"] - 94.0) < 1e-9
    fills = [e for e in events if e[0] == "pending_fill"]
    assert len(fills) == 1 and fills[0][1]["leg_kind"] == "pullback"


def test_pending_fill_breakout_slips(tmp_path, monkeypatch):
    sim = parked_sim(
        tmp_path, monkeypatch, slippage_pct_per_side=0.1
    )
    sim._spot_price = lambda: 102.0
    set_candle(sim, 100.0, 102.2, 99.9, 102.1)
    sim.check_pending(time.time())
    pos = sim.state["open"]
    assert pos is not None and pos["entry_kind"] == "breakout"
    assert abs(pos["entry"] - 102.0 * 1.001) < 1e-9  # stop-market slips


def test_same_candle_flush(tmp_path, monkeypatch):
    events = []
    sim = parked_sim(tmp_path, monkeypatch, events=events)
    sim._spot_price = lambda: 98.0
    # The minute that fills the pullback also runs through the hard level.
    set_candle(sim, 100.0, 100.1, 92.0, 100.0)
    sim.check_pending(time.time())
    assert sim.state["open"] is None
    closes = [e for e in events if e[0] == "close"]
    assert len(closes) == 1 and closes[0][1]["outcome"] == "SL"


def test_pending_killed_hard_touch(tmp_path, monkeypatch):
    events = []
    plan = make_plan(entry_legs=[{"kind": "breakout", "trigger": 102.0}])
    sim = parked_sim(tmp_path, monkeypatch, events=events, plan=plan)
    # Price collapses to the hard level without ever triggering the entry.
    set_candle(sim, 100.0, 100.2, 92.0, 93.0)
    sim.check_pending(time.time())
    assert sim.state["pending"] is None and sim.state["open"] is None
    kills = [e for e in events if e[0] == "pending_killed"]
    assert len(kills) == 1 and kills[0][1]["cause"] == "hard_touch"


def test_pending_soft_wick_does_not_kill(tmp_path, monkeypatch):
    plan = make_plan(entry_legs=[{"kind": "breakout", "trigger": 102.0}])
    sim = parked_sim(tmp_path, monkeypatch, plan=plan)
    # Wick through the soft (94) but not the hard (92.4), hold semantics,
    # no final 15m closes -> the sweep must NOT cancel the plan.
    set_candle(sim, 100.0, 100.2, 93.5, 99.0)
    sim.check_pending(time.time())
    assert sim.state["pending"] is not None


def test_pending_killed_soft_touch_semantics(tmp_path, monkeypatch):
    events = []
    plan = make_plan(
        invalidation_semantics="touch", confirm_bars=None, hard_level=None,
        entry_legs=[{"kind": "breakout", "trigger": 102.0}],
    )
    sim = parked_sim(tmp_path, monkeypatch, events=events, plan=plan)
    set_candle(sim, 100.0, 100.2, 93.9, 99.0)
    sim.check_pending(time.time())
    assert sim.state["pending"] is None
    kills = [e for e in events if e[0] == "pending_killed"]
    # touch semantics: hard == soft, so the touch registers as the hard kill
    assert len(kills) == 1 and kills[0][1]["cause"] == "hard_touch"


def test_pending_killed_soft_confirmed(tmp_path, monkeypatch):
    events = []
    plan = make_plan(entry_legs=[{"kind": "breakout", "trigger": 102.0}])
    sim = parked_sim(tmp_path, monkeypatch, events=events, plan=plan)
    now = time.time()
    p = sim.state["pending"]
    p["opened_ts"] = now - 4000.0  # parked over an hour ago
    bar1, bar2 = now - 2700.0, now - 1800.0  # both final 15m bars
    sim._fetch_ohlcv = lambda tf, limit: [
        [bar1 * 1000.0, 95.0, 95.2, 93.0, 93.5, 10.0],
        [bar2 * 1000.0, 93.5, 94.0, 93.0, 93.2, 10.0],
    ]
    set_candle(sim, 93.4, 93.6, 93.2, 93.3)
    sim.check_pending(now)
    assert sim.state["pending"] is None
    kills = [e for e in events if e[0] == "pending_killed"]
    assert len(kills) == 1 and kills[0][1]["cause"] == "soft_confirmed"


def test_pending_regime_cancel_needs_persistence(tmp_path, monkeypatch):
    events = []
    sim = parked_sim(
        tmp_path, monkeypatch, events=events, regime_persist_ticks=2
    )
    sim._regime_ready = lambda: (True, "down", "test regime")
    set_candle(sim, 100.0, 100.2, 99.8, 100.1)
    sim.check_pending(time.time())
    assert sim.state["pending"] is not None  # 1 tick: not yet
    sim.check_pending(time.time())
    assert sim.state["pending"] is None
    cancels = [e for e in events if e[0] == "pending_regime_cancel"]
    assert len(cancels) == 1 and cancels[0][1]["regime"] == "down"


def test_pending_regime_reset_on_flapping(tmp_path, monkeypatch):
    sim = parked_sim(tmp_path, monkeypatch, regime_persist_ticks=2)
    set_candle(sim, 100.0, 100.2, 99.8, 100.1)
    sim._regime_ready = lambda: (True, "down", "against")
    sim.check_pending(time.time())
    sim._regime_ready = lambda: (True, "up", "with us again")
    sim.check_pending(time.time())
    assert sim.state["pending"]["regime_against_ticks"] == 0
    sim._regime_ready = lambda: (True, "down", "against")
    sim.check_pending(time.time())
    assert sim.state["pending"] is not None  # count restarted from zero


# ------------------------------------------------------- decide integration

def decide(sim, monkeypatch, md):
    monkeypatch.setattr(simulator, "run_analysis", lambda cfg: ("Buy", {
        "final_trade_decision": md,
    }))
    monkeypatch.setattr(simulator, "reasoning_from_state", lambda s: {})
    sim.maybe_decide(time.time())


def plan_md(plan):
    return (
        "**Rating**: Buy\n\n**Price Target**: 110.0\n\n"
        f"**Execution Plan**:\n```json\n{json.dumps(plan)}\n```"
    )


def test_maybe_decide_parks_with_plan_mode(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events)
    decide(sim, monkeypatch, plan_md(make_plan()))
    assert sim.state["pending"] is not None and sim.state["open"] is None
    decisions = [e for e in events if e[0] == "decision"]
    assert decisions[-1][1]["action"] == "pending"


def test_maybe_decide_market_mode_unchanged(tmp_path, monkeypatch):
    events = []
    sim = make_sim(tmp_path, monkeypatch, events=events, entry_mode="market")
    decide(sim, monkeypatch, plan_md(make_plan()))
    # Legs in the contract are ignored: immediate market entry, structural armed.
    assert sim.state["pending"] is None and sim.state["open"] is not None
    assert sim.state["open"]["entry_kind"] == "market"
    decisions = [e for e in events if e[0] == "decision"]
    assert decisions[-1][1]["action"] == "open"


# ------------------------------------------------------------------- state

def test_state_validation_pending(tmp_path, monkeypatch):
    sim = make_sim(tmp_path, monkeypatch)
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"equity": 1.0, "open": None, "closed": [],
                                "pending": "nope", "wins": 0, "losses": 0,
                                "pnl_total": 0.0, "last_decision_at": None}))
    try:
        sim._read_state_file(str(path), 1.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # A pre-feature state file simply gains the default.
    path.write_text(json.dumps({"equity": 1.0, "open": None, "closed": [],
                                "wins": 0, "losses": 0, "pnl_total": 0.0,
                                "last_decision_at": None}))
    state = sim._read_state_file(str(path), 1.0)
    assert state["pending"] is None
