"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_instrument_context_from_state,
    get_language_instruction,
    get_scenario_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        critic_review = state["critic_review"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent on an intraday crypto-perpetual desk. "
                    "Based on the critic-reviewed signal below, decide whether to "
                    "Buy, Sell, or Hold. Anchor your reasoning in the analysts' "
                    "reports and the critic's review — do not second-guess the "
                    "direction the critic already scrutinized. "
                    "The Risk Manager computes the exact stop-loss, take-profit, "
                    "and position size deterministically from ATR after you decide, "
                    "so focus only on direction and conviction, not entry/stop "
                    "levels or sizing."
                    + get_language_instruction()
                    + get_horizon_instruction()
                    + get_scenario_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the critic-reviewed signal for {company_name}. "
                    f"{instrument_context} It incorporates the analysts' intraday "
                    f"technical read, any fresh catalysts, and a critique of "
                    f"whether the move is genuinely worth trading.\n\n"
                    f"Critic-Reviewed Signal: {critic_review}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]


        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
