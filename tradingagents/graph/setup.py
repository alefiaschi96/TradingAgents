# TradingAgents/graph/setup.py

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.market_lean import market_gate_shortcut_reason
from tradingagents.dataflows.config import get_config

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic

logger = logging.getLogger(__name__)

# Node label for the early exit taken when the market analyst's report backs no
# trade (HOLD/flat or no usable data). Ending there skips every downstream LLM
# stage (remaining analysts, debate, trader, risk, PM) — the end-of-run analyst
# gate would veto those cases anyway, so running them is pure cost.
MARKET_GATE_NODE = "Market Gate Hold"


def _market_gate_hold_node(state):
    """Terminal node for a short-circuited run: emit an explicit Hold.

    ``**Rating**: Hold`` keeps ``parse_rating`` (and thus ``propagate``'s
    returned rating) working exactly as for a full run.
    """
    reason = market_gate_shortcut_reason(state.get("market_report", "")) or "no trade backed"
    logger.info(
        "market gate short-circuit: %s — skipping remaining analysts/debate/trader/PM",
        reason,
    )
    text = (
        "**Rating**: Hold\n\n"
        f"**Executive Summary**: Analysis short-circuited by the market gate: {reason}. "
        "The remaining pipeline (other analysts, debate, trader, risk, portfolio "
        "manager) was skipped to save cost; the market analyst's report did not "
        "back a directional trade."
    )
    return {"final_trade_decision": text}


def _market_gate_router(next_node: str):
    """Route to the early Hold exit when enabled and the report backs no trade.

    The flag is read from the live dataflows config at run time (set by
    ``TradingAgentsGraph.__init__`` via ``set_config``), so the same compiled
    graph honours whatever the current run configured. Off (or an empty
    report) falls through to the normal pipeline — fail open, like the
    end-of-run analyst gate.
    """

    def route(state) -> str:
        if not get_config().get("market_gate_short_circuit"):
            return next_node
        if market_gate_shortcut_reason(state.get("market_report", "")) is None:
            return next_node
        return MARKET_GATE_NODE

    return route


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        analyst_concurrency_limit: int = 1,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.analyst_concurrency_limit = analyst_concurrency_limit

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(
            selected_analysts,
            concurrency_limit=self.analyst_concurrency_limit,
        )

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
            workflow.add_node(spec.clear_node, create_msg_delete())
            workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # Start with the first analyst
        workflow.add_edge(START, plan.specs[0].agent_node)

        # Early market gate: when the market analyst's finished report backs no
        # trade, exit through an explicit Hold instead of paying for the rest
        # of the pipeline. Wired only when the market analyst is selected; the
        # router itself is a no-op unless config enables it at run time.
        market_gate_wired = any(spec.key == "market" for spec in plan.specs)
        if market_gate_wired:
            workflow.add_node(MARKET_GATE_NODE, _market_gate_hold_node)
            workflow.add_edge(MARKET_GATE_NODE, END)

        # Connect analysts in sequence
        for i, spec in enumerate(plan.specs):
            current_analyst = spec.agent_node
            current_tools = spec.tool_node
            current_clear = spec.clear_node

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            next_node = (
                plan.specs[i + 1].agent_node
                if i < len(plan.specs) - 1
                else "Bull Researcher"
            )
            if spec.key == "market":
                # The market report is final once its tool loop ends (the clear
                # node), so this is the earliest point the gate can judge it.
                workflow.add_conditional_edges(
                    current_clear,
                    _market_gate_router(next_node),
                    [next_node, MARKET_GATE_NODE],
                )
            else:
                workflow.add_edge(current_clear, next_node)

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
