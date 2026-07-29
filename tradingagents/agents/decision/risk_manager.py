"""Risk Manager: the only agent that speaks to stop-loss, take-profit, and size.

Replaces the old aggressive/neutral/conservative risk debate + Portfolio
Manager. Direction and conviction are already settled by the time this node
runs (Signal Synthesizer -> Critic Manager -> Trader); this agent's sole job
is to size the trade and set levels *deterministically*:

- stop-loss / take-profit are ALWAYS precomputed by the ATR-based
  ``get_sl_tp_levels`` tool, which returns three profiles (Tight / Standard
  / Wide).  The LLM picks one profile; the code then forces that profile's
  exact SL/TP values — the LLM never invents the numbers itself.
- position size is a conviction-scaled tier (Full/Half/Quarter/None), capped
  by the Critic Manager's verdict.
"""

from __future__ import annotations

import re

from tradingagents.agents.schemas import (
    PortfolioRating,
    PositionSize,
    RiskDecision,
    RiskProfile,
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

# Parse each profile row from the ATR tool's table:
#   | Tight (1.0×ATR, 2R) | 74.7955 | 78.4990 | 1.2345 | 1:2.0 |
_PROFILE_ROW_RE = re.compile(
    r"\|\s*(Tight|Standard|Wide)\b[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
    re.IGNORECASE,
)

# Parse the LLM's chosen profile from rendered markdown:
#   **Risk Profile**: Standard
_PROFILE_CHOICE_RE = re.compile(
    r"\*\*Risk Profile\*\*:\s*(\w+)", re.IGNORECASE,
)


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

        # ── Parse all profiles from tool output ──────────────────────
        profiles: dict[str, tuple[float, float]] = {}
        for m in _PROFILE_ROW_RE.finditer(sl_tp_table):
            profiles[m.group(1).capitalize()] = (float(m.group(2)), float(m.group(3)))

        if not profiles:
            return {
                "final_trade_decision": _no_trade_decision(
                    "No trade this round: could not parse SL/TP profiles "
                    f"from ATR tool output for {symbol}."
                )
            }

        prompt = f"""You are the Risk Manager on an intraday crypto-perpetual desk. Direction and conviction are ALREADY decided upstream — your only job is to pick the right risk profile, set position size, and write a rationale.

{instrument_context}

**Critic-reviewed signal:**
{critic_review}

**Trader's proposal:**
{trader_plan}

**ATR-based SL/TP profiles (precomputed — pick one):**
{sl_tp_table}

---

**Instructions:**
1. Set `rating` to the critic's rating ({critic_rating}).
2. Set `risk_profile` to **Tight**, **Standard**, or **Wide** based on current volatility and signal quality:
   - **Tight** (1.0×ATR, 2R): high-conviction, clean setup, low current volatility
   - **Standard** (1.5×ATR, 2R): balanced default for most setups
   - **Wide** (2.0×ATR, 3R): volatile conditions or weaker conviction
3. Set `position_size` to at most **{max_size}** — "Full" only for high conviction (Buy/Sell) with no critic downgrade; otherwise "Half" or lower.
4. You do NOT need to set stop_loss or take_profit — they are injected from the chosen profile automatically.
5. Write a short risk_rationale explaining why you chose this profile and size.""" + get_language_instruction()

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_risk_decision,
            "Risk Manager",
        )

        # ── Resolve chosen profile and override SL/TP ────────────────
        choice_match = _PROFILE_CHOICE_RE.search(final_trade_decision)
        chosen = choice_match.group(1).capitalize() if choice_match else "Standard"
        if chosen not in profiles:
            chosen = "Standard" if "Standard" in profiles else next(iter(profiles))

        parsed_sl, parsed_tp = profiles[chosen]

        final_trade_decision = re.sub(
            r"(\*\*Stop Loss\*\*:\s*)\S+",
            rf"\g<1>{parsed_sl}",
            final_trade_decision,
        )
        final_trade_decision = re.sub(
            r"(\*\*Take Profit\*\*:\s*)\S+",
            rf"\g<1>{parsed_tp}",
            final_trade_decision,
        )
        final_trade_decision = re.sub(
            r"(\*\*Price Target\*\*:\s*)\S+",
            rf"\g<1>{parsed_tp}",
            final_trade_decision,
        )
        # Ensure the chosen profile is reflected in the text
        final_trade_decision = re.sub(
            r"(\*\*Risk Profile\*\*:\s*)\S+",
            rf"\g<1>{chosen}",
            final_trade_decision,
        )

        return {"final_trade_decision": final_trade_decision}

    return risk_manager_node
