"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools
import logging

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import (
    TraderProposal,
    extract_trader_params,
    render_trader_proposal,
)
from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_instrument_context_from_state,
    get_language_instruction,
    get_scenario_instruction,
)
from tradingagents.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    "In addition to the direction, you MUST specify concrete risk-management "
                    "parameters for the position:\n"
                    "  • stop_loss_pct — stop-loss distance as a % of entry (typical 0.2–2.0)\n"
                    "  • take_profit_rr — reward:risk ratio for the take-profit (typical 1.0–4.0)\n"
                    "  • balance_pct — fraction of equity to commit as margin, 0.0–1.0 "
                    "(e.g. 0.5 = use 50%)\n"
                    "  • stop_mode — 'fixed' (use stop_loss_pct) or 'atr' (volatility-scaled)\n"
                    "  • stop_atr_mult — ATR multiplier when stop_mode is 'atr' (typical 1.0–3.0)\n"
                    "Choose values that reflect the current volatility regime, conviction level, "
                    "and risk/reward of the setup. Tighter stops and smaller sizing for low-"
                    "conviction or choppy setups; wider stops and larger sizing when the edge "
                    "is clear and the trend is strong."
                    + get_language_instruction()
                    + get_horizon_instruction()
                    + get_scenario_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"the analysts' intraday technical read and any fresh catalysts. "
                    f"Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        # Structured path: extract both the rendered markdown AND the
        # machine-readable risk params from the Pydantic model.
        trader_plan: str | None = None
        trader_params: dict = {}

        if structured_llm is not None:
            try:
                proposal: TraderProposal = structured_llm.invoke(messages)
                trader_plan = render_trader_proposal(proposal)
                trader_params = extract_trader_params(proposal)
            except Exception as exc:
                logger.warning(
                    "Trader: structured-output invocation failed (%s); "
                    "retrying once as free text",
                    exc,
                )

        # Free-text fallback: no structured params available.
        if trader_plan is None:
            response = llm.invoke(messages)
            trader_plan = response.content

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "trader_params": trader_params,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
