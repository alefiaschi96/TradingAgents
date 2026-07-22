from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_instrument_context_from_state,
    get_language_instruction,
    get_scenario_instruction,
)
from tradingagents.dataflows.config import get_config


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)

        trader_decision = state["trader_investment_plan"]

        if get_config().get("intraday"):
            prompt = f"""As the Conservative Risk Analyst, your job is **capital preservation** on a single LEVERAGED perpetual-future position held for the **next 1-2 hours**. On leverage a stop-out is real money lost fast, so your bar for taking risk is high. When evaluating the trader's decision or plan, hunt for the ways this scalp can hurt the account and argue for the safest viable posture — frequently FLAT or reduced size when the edge is marginal. Here is the trader's decision:

{trader_decision}

Your task is to actively counter the Aggressive and Neutral Analysts, exposing the downside they wave away. Build the low-risk case around the specific hazards of an intraday leveraged scalp:
- Invalidation quality: insist on a tight, structurally-valid ATR stop that sits just beyond real structure (swing, VWAP, band edge) — not at a noise distance that gets clipped, and not so wide that a stop-out is a large loss.
- Squeeze / crowding risk: flag extreme funding (crowded one-sided positioning that can be squeezed against the trade) and stretched open interest, using the futures positioning data, not just price.
- Order-book fragility: thin or imbalanced depth means slippage and air-pockets — a level that looks like support can vanish.
- Chop and false breakouts: in a rangebound or low-conviction tape, breakouts fail and round-trip the stop.
- Chasing: warn against entries stretched far from VWAP where mean-reversion can snap back into the stop.
- Catalyst timing: opening right before a known news/data catalyst is a coin-flip, not an edge.
When the trade lacks a clean invalidation, fights crowded positioning, or chases an extended move, argue for FLAT or smaller size for the next 1-2 hours.

Respond directly to the other analysts using these resources:
{instrument_context}
Market Research Report (intraday technicals plus futures positioning — funding rate, open interest, order-book imbalance): {market_research_report}
Latest news / catalysts: {news_report}
Here is the current conversation history: {history} Here is the last response from the aggressive analyst: {current_aggressive_response} Here is the last response from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage by questioning their optimism and emphasizing the downside they overlooked. Address each of their counterpoints to show why a low-risk stance — a tighter stop, smaller size, or staying FLAT — best protects this leveraged position over the next 1-2 hours. Focus on debating and critiquing their arguments. Output conversationally as if you are speaking without any special formatting.""" + get_language_instruction() + get_horizon_instruction() + get_scenario_instruction()
        else:
            prompt = f"""As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:

{trader_decision}

Your task is to actively counter the arguments of the Aggressive and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} Here is the last response from the aggressive analyst: {current_aggressive_response} Here is the last response from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting.""" + get_language_instruction() + get_horizon_instruction()

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
