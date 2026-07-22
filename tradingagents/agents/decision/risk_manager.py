"""Risk Manager: the only agent that speaks to stop-loss, take-profit, and size.

Replaces the old aggressive/neutral/conservative risk debate + Portfolio
Manager. Direction and conviction are already settled by the time this node
runs (Signal Synthesizer -> Critic Manager -> Trader); this agent's sole job
is to size the trade and set levels *deterministically*:

- stop-loss / take-profit are ALWAYS read from the ATR-based ``get_sl_tp_levels``
  tool, called directly in code so the numbers can never be an LLM invention
  ("non sull'LLM a sentimento").
- position size is a conviction-scaled tier (Full/Half/Quarter/None), capped
  by the Critic Manager's verdict.

The LLM's only job is to copy those numbers into the final structured
decision and write a short rationale — it never derives them itself.
"""

from __future__ import annotations

import re

from tradingagents.agents.schemas import (
    PortfolioRating,
    PositionSize,
    RiskDecision,
    render_risk_decision,
)
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.agents.utils.sl_tp_tools import get_sl_tp_levels
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

_ACTION_RE = re.compile(r"\*\*Action\*\*:\s*(\w+)", re.IGNORECASE)
_VERDICT_RE = re.compile(r"\*\*Verdict\*\*:\s*(\w+)", re.IGNORECASE)


def _parse_trader_action(text: str) -> str:
    """Extract the Trader's Buy/Hold/Sell action from its rendered proposal."""
    match = _ACTION_RE.search(text)
    if match:
        return match.group(1).capitalize()
    return parse_rating(text)  # heuristic fallback; defaults to "Hold"


def _parse_critic_verdict(text: str) -> str:
    """Extract the Critic Manager's Approve/Downgrade/Veto verdict."""
    match = _VERDICT_RE.search(text)
    if match:
        return match.group(1).capitalize()
    return "Approve"


def _no_trade_decision(reason: str) -> str:
    decision = RiskDecision(
        rating=PortfolioRating.HOLD,
        stop_loss=None,
        take_profit=None,
        position_size=PositionSize.NONE,
        risk_rationale=reason,
    )
    return render_risk_decision(decision)


def create_risk_manager(llm):
    structured_llm = bind_structured(llm, RiskDecision, "Risk Manager")

    def risk_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        symbol = state["company_of_interest"]

        critic_review = state.get("critic_review", "")
        trader_plan = state.get("trader_investment_plan", "")

        critic_rating = parse_rating(critic_review)
        critic_verdict = _parse_critic_verdict(critic_review)
        trader_action = _parse_trader_action(trader_plan)

        if critic_rating == "Hold" or trader_action == "Hold":
            return {
                "final_trade_decision": _no_trade_decision(
                    "No trade this round: the critic-reviewed signal is flat "
                    "(Hold), so there is nothing to size — staying out "
                    "preserves capital."
                )
            }

        direction = "long" if trader_action == "Buy" else "short"

        # Deterministic, always sourced from the tool — never from the LLM.
        sl_tp_table = get_sl_tp_levels.invoke({"symbol": symbol, "direction": direction})
        if "data unavailable" in sl_tp_table.lower():
            return {
                "final_trade_decision": _no_trade_decision(
                    "No trade this round: ATR/SL-TP data was unavailable for "
                    f"{symbol}, so a position cannot be sized safely."
                )
            }

        max_size = "Half" if critic_verdict == "Downgrade" else "Full"
        if critic_rating in ("Overweight", "Underweight"):
            max_size = "Half"

        prompt = f"""You are the Risk Manager on an intraday crypto-perpetual desk. Direction and conviction are ALREADY decided upstream — your only job is to finalize sizing using the ATR-based levels below. Do NOT invent, adjust, or re-derive stop-loss / take-profit numbers; copy them exactly as given.

{instrument_context}

**Critic-reviewed signal:**
{critic_review}

**Trader's proposal:**
{trader_plan}

**ATR-based SL/TP levels (source of truth — copy exactly, do not alter):**
{sl_tp_table}

---

Set `rating` to the critic's rating ({critic_rating}) unless the data above makes a trade impossible, in which case fall back to Hold. Set `position_size` to at most **{max_size}** — "Full" only for high conviction (Buy/Sell) with no critic downgrade; otherwise "Half" or lower. Copy `stop_loss` and `take_profit` directly from the table above (the SL row and the 2R TP row — use the numeric values exactly). Write a short risk_rationale explaining the levels and size in plain terms.""" + get_language_instruction()

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_risk_decision,
            "Risk Manager",
        )

        return {"final_trade_decision": final_trade_decision}

    return risk_manager_node
