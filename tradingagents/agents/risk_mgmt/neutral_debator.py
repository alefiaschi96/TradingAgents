from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_instrument_context_from_state,
    get_language_instruction,
    get_scenario_instruction,
)
from tradingagents.dataflows.config import get_config


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)

        trader_decision = state["trader_investment_plan"]

        if get_config().get("intraday"):
            timeframe = get_config().get("intraday_timeframe", "15m")
            prompt = f"""You are the Neutral Risk Analyst weighing an **intraday, leveraged perpetual-future scalp** over the **next 1-2 hours** on {timeframe} bars. Your job is to judge whether the intraday edge is strong enough to justify putting on a LEVERAGED 1-2h position, or whether the disciplined move is to stay FLAT and wait. You challenge BOTH the Aggressive Analyst's eagerness to chase AND the Conservative Analyst's over-caution, and you steer toward a measured, defensible stance. Here is the trader's decision:

{trader_decision}

Hold both sides to account. Against the aggressive case: is this a real continuation backed by fresh open interest and order-book pressure, or is it late-cycle chasing into crowded positioning where the funding rate flags squeeze risk? Against the conservative case: is the caution warranted, or is it leaving a clean, well-defined setup on the table out of pure fear? Balance momentum/continuation against mean-reversion risk, and separate a genuine catalyst from intraday noise. Recommend taking the trade with a disciplined invalidation level when the edge is genuinely there, or waiting FLAT when it is not. Do NOT reason from diversification, broad-economic shifts, or long-term framing — stay entirely on the 1-2h horizon.

Use the following resources to support a measured stance:

{instrument_context}
Market Research Report (intraday technicals AND futures positioning — funding rate, open interest, order-book imbalance): {market_research_report}
Latest news / catalysts: {news_report}
Here is the current conversation history: {history} Here is the last response from the aggressive analyst: {current_aggressive_response} Here is the last response from the conservative analyst: {current_conservative_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage actively by critiquing both sides, pointing out where the aggressive analyst overstates the edge and where the conservative analyst overstates the danger. Reference futures positioning where it matters — crowded funding and fading open interest cut against chasing, while a contrarian squeeze setup can justify a measured entry the conservative view would skip. Focus on debating rather than simply presenting data, and land on a balanced, executable read: trade with tight invalidation, or stay FLAT. Output conversationally as if you are speaking without any special formatting.""" + get_language_instruction() + get_horizon_instruction() + get_scenario_instruction()
        else:
            prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies.Here is the trader's decision:

{trader_decision}

Your task is to challenge both the Aggressive and Conservative Analysts, pointing out where each perspective may be overly optimistic or overly cautious. Use insights from the following data sources to support a moderate, sustainable strategy to adjust the trader's decision:

{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} Here is the last response from the aggressive analyst: {current_aggressive_response} Here is the last response from the conservative analyst: {current_conservative_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage actively by analyzing both sides critically, addressing weaknesses in the aggressive and conservative arguments to advocate for a more balanced approach. Challenge each of their points to illustrate why a moderate risk strategy might offer the best of both worlds, providing growth potential while safeguarding against extreme volatility. Focus on debating rather than simply presenting data, aiming to show that a balanced view can lead to the most reliable outcomes. Output conversationally as if you are speaking without any special formatting.""" + get_language_instruction() + get_horizon_instruction()

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
