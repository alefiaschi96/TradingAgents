"""Dashboard rendering of the conditional entry plan (ENTRY_MODE=plan).

Covers the pure view helpers only — no HTTP server: the `_pending` payload
built from the simulator's state["pending"] schema (legs with price_target /
invalidation_level, plan-level soft/hard kill levels, expires_ts), the price
path parsed from 'pending watch' log lines, and the plain-Italian feed lines
for the pending lifecycle events.
"""

import time

from paper_sim import dashboard as dash


def _pending_state(**over):
    now = time.time()
    p = {
        "side": "buy",
        "rating": "BUY",
        "legs": [
            {
                "kind": "pullback",
                "zone_low": 97.0,
                "zone_high": 98.0,
                "price_target": None,
                "invalidation_level": None,
            },
            {
                "kind": "breakout",
                "trigger": 102.0,
                "price_target": 110.0,
                "invalidation_level": 99.5,
            },
        ],
        "ref_price": 100.0,
        "soft": 94.0,
        "hard": 92.4,
        "semantics": "close",
        "created_at": "2026-07-26T10:00:00+00:00",
        "created_ts": now,
        "expires_ts": now + 3600.0,
    }
    p.update(over)
    return p


WATCH_LINES = [
    "2026-07-26 12:00:00 | PENDING buy PF_SOLUSD: pullback zone 97.0000-98.0000 "
    "OR breakout >=102.0000 | valid 120min | ref 100.0000 (basis ratio 1.000000)",
    "2026-07-26 12:01:00 | pending watch buy: 1m range [99.0000, 99.5000] vs pullback zone",
    "2026-07-26 12:02:00 | pending watch buy: 1m range [99.5000, 100.5000] vs pullback zone",
]


def test_pending_view_builds_rows_and_levels():
    v = dash._pending(_pending_state(), WATCH_LINES)
    assert v["up"] is True
    assert [r["k"] for r in v["leg_rows"]] == ["pull", "brk"]
    assert "limit" in v["leg_rows"][0]["head"]
    assert "stop-entry" in v["leg_rows"][1]["head"]
    assert v["leg_rows"][1]["target"] == "obiettivo 110.00"
    assert "99.500" in v["leg_rows"][1]["inval"]
    kinds = {(x["k"], x["v"]) for x in v["levels"]}
    assert ("entry", 98.0) in kinds          # pullback near edge for a buy
    assert ("entry", 102.0) in kinds         # breakout trigger
    assert ("target", 110.0) in kinds
    assert ("inval", 92.4) in kinds          # plan hard kill level
    assert ("soft", 94.0) in kinds           # plan soft kill level (15m closes)
    assert "SOLO se" in v["plain"]
    assert "Annulla tutto prima dell'ingresso" in v["plain"]
    assert "15m" in v["plain"]               # close semantics spelled out


def test_pending_view_countdown_and_path():
    v = dash._pending(_pending_state(), WATCH_LINES)
    assert v["expires_epoch"] is not None
    assert 0 < v["left_min"] <= 60.0
    # path: ref price + the two watch-candle mids
    assert v["path"] == [100.0, 99.25, 100.0]
    assert v["now"] == 100.0                 # last pending-watch mid
    assert "validi fino alle" in v["meta"]


def test_pending_view_dedupes_soft_equal_to_leg_invalidation():
    p = _pending_state(soft=99.5, hard=None)
    v = dash._pending(p, [])
    soft_levels = [x for x in v["levels"] if x["k"] == "soft"]
    assert soft_levels == []                 # already drawn as the leg's inval


def test_pending_view_touch_semantics_wording():
    v = dash._pending(_pending_state(semantics="touch", hard=None), [])
    assert "tocca 94.000" in v["plain"]
    assert "15m" not in v["plain"]


def test_pending_view_sell_mirrors_geometry():
    p = _pending_state(side="sell")
    p["legs"] = [
        {"kind": "pullback", "zone_low": 102.0, "zone_high": 103.0,
         "price_target": None, "invalidation_level": None},
    ]
    p.update(soft=106.0, hard=107.6)
    v = dash._pending(p, [])
    assert v["up"] is False
    assert ("entry", 102.0) in {(x["k"], x["v"]) for x in v["levels"]}  # near edge for a sell
    assert "sale" in v["leg_rows"][0]["head"]


def test_pending_view_none_when_flat():
    assert dash._pending(None, []) is None


def _feed(msg):
    rows = dash._humanize([f"2026-07-26 12:00:00 | {msg}"])
    assert rows, msg
    return rows[-1]


def test_humanize_pending_lifecycle():
    assert _feed("PENDING buy PF_SOLUSD: pullback zone | valid 120min | ref 1 (basis ratio 1)")["k"] == "pend"
    assert _feed("PENDING FILLED (pullback) buy @ 97.9 — other leg cancelled")["k"] == "open"
    assert "ROTTURA" in _feed("PENDING FILLED (breakout) buy @ 102.1 — other leg cancelled")["h"]
    assert _feed("PENDING EXPIRED after 120min without a fill (was: ...)")["k"] == "muted"
    assert "protezione" in _feed("PENDING KILLED: 1m range crossed the hard level 92.4 before any fill")["h"]
    assert "15m" in _feed("PENDING KILLED: soft invalidation 94.0 confirmed by 2 15m close(s) before any fill")["h"]
    assert "GIRATO" in _feed("regime flipped against pending buy (down) — cancelling the plan: x")["h"]
    assert _feed("same-candle flush: the minute that filled the entry also ran through SL — closing immediately")["k"] == "warn"


def test_humanize_leg_gates_before_generic_veto():
    row = _feed("entry leg VETOED (breakout >=102.0000): cost gate: tp 0.4% < 2.0x cost")
    assert row["k"] == "veto" and "Gamba" in row["h"]
    row = _feed("entry leg VETOED (pullback zone): target gate: implied rr 0.8 < 1.5")
    assert "margine" in row["h"]
    assert "fuori misura" in _feed("entry leg dropped: pullback zone 0.2 ATR from price")["h"]
    assert "a mercato" in _feed("no entry legs in the execution plan -> market entry")["h"]
    assert "rischio" in _feed("every entry leg failed a risk gate -> no trade")["h"]


def test_humanize_conditional_hold_lines():
    assert _feed(
        "market gate: HOLD names trigger 100.0 within 2.0 ATR of price 99.0 — "
        "conditional-hold pass, pipeline continues"
    )["k"] == "think"
    assert "LONTANO" in _feed(
        "market gate: HOLD trigger(s) [120.0] all beyond 2.0 ATR of price 99.0 — "
        "short-circuit stands"
    )["h"]


def test_event_key_filter_keeps_pending_lines():
    keep = [
        "2026-07-26 12:00:00 | PENDING buy ...",
        "2026-07-26 12:01:00 | entry leg VETOED (x): cost gate",
        "2026-07-26 12:02:00 | conditional-hold pass, pipeline continues",
    ]
    for ln in keep:
        assert any(k in ln for k in dash._KEY), ln
    # per-minute pending watch lines are chart data, not feed events
    assert not any(k in WATCH_LINES[1] for k in dash._KEY)
