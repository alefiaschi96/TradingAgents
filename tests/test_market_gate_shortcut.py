"""Early market gate: short-circuit the graph when the market analyst's report
backs no trade (HOLD/flat or no usable data).

The end-of-run analyst gate vetoes those cases regardless of what the
downstream agents decide, so running them is pure cost. These tests cover the
shared classifier, the router, the terminal Hold node, and the graph wiring —
no LLM calls anywhere.
"""

import copy
from unittest.mock import MagicMock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.agents.utils.market_lean import market_gate_shortcut_reason
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.config import set_config
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import (
    MARKET_GATE_NODE,
    GraphSetup,
    _market_gate_hold_node,
    _market_gate_router,
)

HOLD_REPORT = (
    "FINAL TRANSACTION PROPOSAL: **HOLD**\n\n"
    "ETH-USD is stretched; wait for a pullback or a clean breakout."
)
NO_DATA_REPORT = (
    "Data unavailable for `ETH-USD` on the configured market-data source, so "
    "I can’t make a reliable 1–2 hour intraday call without fabricating values."
)
BULL_REPORT = (
    "FINAL TRANSACTION PROPOSAL: **BUY**\n\nUptrend intact, buyers in control."
)


@pytest.fixture(autouse=True)
def _clean_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


# --- classifier ----------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (HOLD_REPORT, "HOLD/flat (no directional signal)"),
        (NO_DATA_REPORT, "no reliable market data"),
        (BULL_REPORT, None),
        ("", None),  # empty report: gate cannot judge — fail open
    ],
)
def test_shortcut_reason(report, expected):
    assert market_gate_shortcut_reason(report) == expected


# --- router ----------------------------------------------------------------------

@pytest.mark.unit
def test_router_disabled_by_default():
    # DEFAULT_CONFIG carries no flag: the router must be a no-op.
    route = _market_gate_router("Bull Researcher")
    assert route({"market_report": HOLD_REPORT}) == "Bull Researcher"


@pytest.mark.unit
def test_router_enabled_routes_only_untradeable_reports():
    set_config({"market_gate_short_circuit": True})
    route = _market_gate_router("Bull Researcher")
    assert route({"market_report": HOLD_REPORT}) == MARKET_GATE_NODE
    assert route({"market_report": NO_DATA_REPORT}) == MARKET_GATE_NODE
    assert route({"market_report": BULL_REPORT}) == "Bull Researcher"
    assert route({"market_report": ""}) == "Bull Researcher"


# --- terminal Hold node -----------------------------------------------------------

@pytest.mark.unit
def test_hold_node_yields_parseable_hold():
    out = _market_gate_hold_node({"market_report": HOLD_REPORT})
    decision = out["final_trade_decision"]
    assert parse_rating(decision) == "Hold"
    assert "short-circuited" in decision


@pytest.mark.unit
def test_hold_node_names_the_no_data_reason():
    out = _market_gate_hold_node({"market_report": NO_DATA_REPORT})
    assert "no reliable market data" in out["final_trade_decision"]


# --- graph wiring -----------------------------------------------------------------

def _setup(selected):
    tool_nodes = {key: MagicMock(name=f"tools_{key}") for key in selected}
    gs = GraphSetup(
        quick_thinking_llm=MagicMock(),
        deep_thinking_llm=MagicMock(),
        tool_nodes=tool_nodes,
        conditional_logic=ConditionalLogic(),
    )
    return gs.setup_graph(selected).compile()


@pytest.mark.unit
def test_gate_node_wired_when_market_selected():
    graph = _setup(["market", "news"])
    assert MARKET_GATE_NODE in graph.get_graph().nodes


@pytest.mark.unit
def test_gate_node_absent_without_market_analyst():
    graph = _setup(["news"])
    assert MARKET_GATE_NODE not in graph.get_graph().nodes


# --- end-to-end invoke (stubbed agents, no LLM) -------------------------------------

@pytest.mark.unit
def test_log_state_tolerates_short_circuited_state(tmp_path):
    """A short-circuited final state lacks every downstream key (trader plan,
    debates, PM). ``_log_state`` must log it with defaults, not crash — this is
    the exact KeyError('trader_investment_plan') seen on the first live run."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    g = TradingAgentsGraph.__new__(TradingAgentsGraph)  # skip heavy __init__
    g.log_states_dict = {}
    g.ticker = "ETH-USD"
    g.config = {"results_dir": str(tmp_path)}

    state = {
        "company_of_interest": "ETH-USD",
        "trade_date": "2026-07-14",
        "market_report": HOLD_REPORT,
        "final_trade_decision": _market_gate_hold_node(
            {"market_report": HOLD_REPORT}
        )["final_trade_decision"],
    }
    g._log_state("2026-07-14", state)  # must not raise

    logged = g.log_states_dict["2026-07-14"]
    assert logged["trader_investment_decision"] == ""
    assert logged["investment_debate_state"]["judge_decision"] == ""
    assert parse_rating(logged["final_trade_decision"]) == "Hold"


@pytest.mark.unit
def test_invoke_short_circuits_before_downstream_agents(monkeypatch):
    from langchain_core.messages import AIMessage

    import tradingagents.graph.setup as setup_mod

    set_config({"market_gate_short_circuit": True})

    def market_stub(state):
        # Finished market analyst: no tool calls, HOLD report in state.
        return {"messages": [AIMessage(content="done")], "market_report": HOLD_REPORT}

    def news_stub(state):
        raise AssertionError("news analyst must be skipped by the market gate")

    monkeypatch.setattr(setup_mod, "create_market_analyst", lambda llm: market_stub)
    monkeypatch.setattr(setup_mod, "create_news_analyst", lambda llm: news_stub)

    out = _setup(["market", "news"]).invoke(
        {
            "messages": [],
            "company_of_interest": "ETH-USD",
            "trade_date": "2026-07-14",
            "asset_type": "crypto",
        }
    )
    assert parse_rating(out["final_trade_decision"]) == "Hold"
    assert "short-circuited" in out["final_trade_decision"]
