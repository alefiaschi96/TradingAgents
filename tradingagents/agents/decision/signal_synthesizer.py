"""Signal Synthesizer: single-pass directional call from the analysts' reports.

Replaces the old bull/bear debate + Research Manager. There is no back-and-
forth: the synthesizer reads the analysts' reports directly and commits to a
rating in one shot. Intraday-only (the last remaining trade horizon).
"""

from __future__ import annotations

from tradingagents.agents.schemas import SignalDecision, render_signal_decision
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
from tradingagents.dataflows.config import get_config


def create_signal_synthesizer(llm):
    structured_llm = bind_structured(llm, SignalDecision, "Signal Synthesizer")

    def signal_synthesizer_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        timeframe = get_config().get("intraday_timeframe", "15m")

        market_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]

        prompt = f"""You are the Signal Synthesizer on an intraday crypto-perpetual desk. Your job is to read the analysts' reports and commit, in one pass, to a directional call for the **next 1-2 hours** on {timeframe} bars. There is no debate round: weigh the evidence yourself and decide.

{instrument_context}

---

**Rating Scale** (use exactly one) — each label is a DIRECTION plus a CONVICTION:
- **Buy**: strong LONG - high conviction price goes UP
- **Overweight**: mild LONG - lean long
- **Hold**: FLAT - no position; for when the evidence is genuinely balanced or too thin to trade
- **Underweight**: mild SHORT - lean short
- **Sell**: strong SHORT - high conviction price goes DOWN

Pick the rating that matches the direction you would ACTUALLY take. Reserve Hold for genuinely balanced or insufficient evidence — it is a real conclusion, not a default to avoid deciding. Weigh agreement across sources (technicals, sentiment, news, positioning) more heavily than any single one; when sources conflict, say so plainly and let that pull you toward Hold or a milder conviction rather than picking a side arbitrarily.

---

**Market Analyst report (intraday technicals + futures positioning):**
{market_report}

**Sentiment Analyst report:**
{sentiment_report}

**News Analyst report:**
{news_report}""" + get_language_instruction() + get_horizon_instruction() + get_scenario_instruction()

        signal_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_signal_decision,
            "Signal Synthesizer",
        )

        return {"signal_decision": signal_decision}

    return signal_synthesizer_node
