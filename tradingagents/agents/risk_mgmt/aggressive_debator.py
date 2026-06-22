from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_instrument_context_from_state,
    get_language_instruction,
    get_scenario_instruction,
)
from tradingagents.dataflows.config import get_config


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)

        trader_decision = state["trader_investment_plan"]

        if get_config().get("intraday"):
            prompt = f"""You are the Aggressive Risk Analyst on an **intraday, leveraged crypto-perpetual** desk. Your job is to champion **taking the directional trade** (long OR short) over the **next 1-2 hours** whenever the intraday edge is genuinely there — and to push for conviction and size when the setup is clean. This is a short-term leveraged scalp, NOT a multi-day or multi-week investment.

Your core argument: excessive caution leaves high-probability 1-2h moves on the table, and FLAT is not free — sitting out a clean setup has a real opportunity cost. When the read lines up, you press the trade. A genuine edge looks like: clean momentum/trend alignment (price working with EMA/VWAP rather than fighting it), a fresh catalyst in the news flow, or favorable derivatives positioning (e.g. the crowd heavily offside on funding/open interest, so a squeeze becomes fuel in your direction). Lean on order-book imbalance for near-term pressure. Where funding is extreme or OI is rising into the move, frame that as a reason to push harder, not to flinch.

Here is the trader's decision:

{trader_decision}

Your task is to build a compelling case for taking this trade with conviction by questioning and critiquing the conservative and neutral stances — show where their caution misses a clean intraday move or where their assumptions are overly defensive. Keep everything anchored to intraday price action and derivatives positioning on a leveraged perpetual. Do NOT use growth, innovation, long-term, valuation, or market-share framing — none of that applies to a 1-2h scalp.

Incorporate insights from the following sources into your arguments:

{instrument_context}
Market Research Report (intraday technicals AND futures positioning — funding rate, open interest, order-book imbalance): {market_research_report}
Latest news / catalysts: {news_report}
Here is the current conversation history: {history} Here are the last arguments from the conservative analyst: {current_conservative_response} Here are the last arguments from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage actively by addressing the concerns raised, refuting the weaknesses in their logic, and asserting why taking the trade — at conviction size on a clean setup — beats sitting flat. Maintain a focus on debating and persuading, not just presenting data. Challenge each counterpoint to underscore why pressing a high-probability intraday edge is optimal. Output conversationally as if you are speaking without any special formatting.""" + get_language_instruction() + get_horizon_instruction() + get_scenario_instruction()
        else:
            prompt = f"""As the Aggressive Risk Analyst, your role is to actively champion high-reward, high-risk opportunities, emphasizing bold strategies and competitive advantages. When evaluating the trader's decision or plan, focus intently on the potential upside, growth potential, and innovative benefits—even when these come with elevated risk. Use the provided market data and sentiment analysis to strengthen your arguments and challenge the opposing views. Specifically, respond directly to each point made by the conservative and neutral analysts, countering with data-driven rebuttals and persuasive reasoning. Highlight where their caution might miss critical opportunities or where their assumptions may be overly conservative. Here is the trader's decision:

{trader_decision}

Your task is to create a compelling case for the trader's decision by questioning and critiquing the conservative and neutral stances to demonstrate why your high-reward perspective offers the best path forward. Incorporate insights from the following sources into your arguments:

{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} Here are the last arguments from the conservative analyst: {current_conservative_response} Here are the last arguments from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage actively by addressing any specific concerns raised, refuting the weaknesses in their logic, and asserting the benefits of risk-taking to outpace market norms. Maintain a focus on debating and persuading, not just presenting data. Challenge each counterpoint to underscore why a high-risk approach is optimal. Output conversationally as if you are speaking without any special formatting.""" + get_language_instruction() + get_horizon_instruction()

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
