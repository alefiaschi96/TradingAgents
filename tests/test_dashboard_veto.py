"""Dashboard veto messages: every veto reason maps to a clear, specific line.

The two gates log ``... VETOED by <analyst|regime> gate: <reason>``. The
dashboard must spell out each case in plain Italian rather than collapsing them
into one catch-all, so anyone watching sees *why* a trade was blocked.
"""

import sys

import pytest

# Import is safe even though pytest fills sys.argv (the module guards the
# optional port arg), but set a clean argv first to be explicit.
sys.argv = ["dashboard"]
from paper_sim import dashboard  # noqa: E402


def _line(reason_line: str) -> str:
    """A full log line as the simulator writes it, for the end-to-end path."""
    return f"2026-07-14 18:01:41 INFO paper_sim | {reason_line}"


# (log reason, expected distinctive fragment of the Italian message)
CASES = [
    # analyst gate
    ("decision Overweight (buy) -> VETOED by analyst gate: market analyst had no reliable market data",
     "non aveva dati affidabili"),
    ("decision Overweight (buy) -> VETOED by analyst gate: market analyst recommends HOLD/flat (no directional signal)",
     "consigliava di ASPETTARE"),
    ("decision Overweight (buy) -> VETOED by analyst gate: buy against market analyst bear read",
     "CONTRO il parere dell'analista"),
    ("decision Underweight (sell) -> VETOED by analyst gate: sell against market analyst bull read",
     "CONTRO il parere dell'analista"),
    # regime gate
    ("decision Overweight (buy) -> VETOED by regime gate: chop (no clear HTF trend)",
     "LATERALE"),
    ("decision Overweight (buy) -> VETOED by regime gate: long against HTF trend",
     "CONTRO l'andamento di fondo"),
    ("decision Underweight (sell) -> VETOED by regime gate: short against HTF trend",
     "CONTRO l'andamento di fondo"),
    ("decision Underweight (sell) -> VETOED by regime gate: overextended (3.1 ATR from EMA — chasing)",
     "troppo lontano dalla media"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("reason_line", "fragment"), CASES)
def test_veto_message_is_specific(reason_line, fragment):
    assert fragment in dashboard._veto_message(reason_line.lower())


@pytest.mark.unit
def test_every_veto_case_is_distinct():
    # No two of the mapped cases collapse to the same text: all are explicit.
    msgs = {dashboard._veto_message(r.lower()) for r, _ in CASES}
    assert len(msgs) == 6  # analyst: no-data / HOLD / opposite ; regime: chop / against-trend / overextended


@pytest.mark.unit
@pytest.mark.parametrize(("reason_line", "fragment"), CASES)
def test_humanize_end_to_end(reason_line, fragment):
    # A veto line flows through the real event pipeline to a "veto" event.
    events = dashboard._humanize([_line(reason_line)])
    assert len(events) == 1
    assert events[0]["k"] == "veto"
    assert fragment in events[0]["h"]
    assert events[0]["t"] == "18:01:41"


@pytest.mark.unit
def test_unknown_veto_falls_back_generically():
    msg = dashboard._veto_message("decision x -> vetoed by some gate: mystery reason")
    assert "annullata" in msg.lower()


SHORT_CIRCUIT_CASES = [
    ("market gate short-circuit: HOLD/flat (no directional signal) — "
     "skipping remaining analysts/debate/trader/PM",
     "consiglia di ASPETTARE"),
    ("market gate short-circuit: no reliable market data — "
     "skipping remaining analysts/debate/trader/PM",
     "non aveva dati affidabili"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("reason_line", "fragment"), SHORT_CIRCUIT_CASES)
def test_market_gate_short_circuit_is_explicit(reason_line, fragment):
    # The early stop (no debate/PM ever ran) reads differently from a veto.
    line = f"2026-07-14 20:01:29 INFO tradingagents.graph.setup | {reason_line}"
    events = dashboard._humanize([line])
    assert len(events) == 1
    assert events[0]["k"] == "veto"
    assert fragment in events[0]["h"]
    assert "fermata subito" in events[0]["h"]


@pytest.mark.unit
def test_regime_transition_shows_early_analysis():
    line = ("2026-07-15 09:14:07 INFO paper_sim | regime transition (chop -> up) — "
            "early analysis, cooldown bypassed: up-trend setup")
    events = dashboard._humanize([line])
    assert len(events) == 1
    assert events[0]["k"] == "think"
    assert "analisi anticipata" in events[0]["h"]


@pytest.mark.unit
def test_port_from_argv_ignores_non_numeric():
    saved = sys.argv
    try:
        sys.argv = ["prog", "tests/test_x.py"]  # pytest-style arg
        assert dashboard._port_from_argv() == 8765
        sys.argv = ["prog", "9000"]
        assert dashboard._port_from_argv() == 9000
    finally:
        sys.argv = saved
