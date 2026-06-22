"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
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


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one) — each label is a DIRECTION plus a CONVICTION:
- **Buy**: strong LONG - high conviction price goes UP
- **Overweight**: mild LONG - lean long
- **Hold**: FLAT - no position; for when the evidence on both sides is genuinely balanced (no edge either way)
- **Underweight**: mild SHORT - lean short
- **Sell**: strong SHORT - high conviction price goes DOWN

Pick the rating that matches the direction you would ACTUALLY take, and commit to a clear LONG or SHORT whenever the debate's strongest arguments warrant one. Reserve Hold/FLAT for genuinely balanced evidence — it is a real conclusion, not a default to avoid deciding. Two things that do NOT justify flipping direction: taking a position merely to be active, and mild caution about an extended move — trimming conviction is fine, but it is never a reason to bet the OTHER way.

---

**Debate History:**
{history}""" + get_language_instruction() + get_horizon_instruction() + get_scenario_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
