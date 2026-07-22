"""Critic Manager: doubts whether the Signal Synthesizer's call is worth trading.

Replaces the old aggressive/neutral/conservative risk debate. Instead of
arguing for a position size, the critic's only job is to scrutinize the
edge itself: is the predicted move big enough, clean enough, and
well-supported enough to pay the cost of trading it, once noise and
conflicting signals are accounted for? It may only hold or soften
conviction — it never flips the direction the Synthesizer picked.
"""

from __future__ import annotations

from tradingagents.agents.schemas import CriticVerdict, render_critic_verdict
from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_critic_manager(llm):
    structured_llm = bind_structured(llm, CriticVerdict, "Critic Manager")

    def critic_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        signal_decision = state["signal_decision"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"\n**Lessons from prior decisions and outcomes:**\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""You are the Critic Manager on an intraday crypto-perpetual desk. The Signal Synthesizer below has already made its directional call — your job is NOT to re-litigate the direction, it is to doubt whether the predicted move is actually worth trading.

{instrument_context}

---

Scrutinize the Synthesizer's case for:
- **Edge size**: is the expected move meaningfully larger than normal intraday noise (spread, slippage, ATR chop), or is this a coin-flip dressed up as conviction?
- **Signal agreement**: do the underlying analyst reports actually agree, or is the "conviction" resting on one strong source while others are silent or mixed?
- **Data quality**: is the evidence concrete (specific levels, readings, positioning) or vague/hand-wavy?

**Signal Synthesizer's call:**
{signal_decision}
{lessons_line}
---

Render exactly one verdict:
- **Approve**: the edge is real and worth trading at the stated conviction — keep the rating unchanged.
- **Downgrade**: there is some edge but it's weaker than claimed — soften conviction one notch in the SAME direction (e.g. Buy -> Overweight, Sell -> Underweight). Overweight/Underweight downgrade to Hold.
- **Veto**: the case is not worth trading at all this round — force Hold, regardless of the original direction.

Never flip Buy into Sell or vice versa; you may only hold steady or move toward Hold.""" + get_language_instruction() + get_horizon_instruction()

        critic_review = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_critic_verdict,
            "Critic Manager",
        )

        return {"critic_review": critic_review}

    return critic_manager_node
