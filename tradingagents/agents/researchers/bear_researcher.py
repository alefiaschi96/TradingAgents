from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        if get_config().get("intraday"):
            timeframe = get_config().get("intraday_timeframe", "15m")
            prompt = f"""You are an intraday Bear Analyst building the SHORT case for the {target_label} over the NEXT 1-2 HOURS on {timeframe} bars. This is a short-term trade, NOT a multi-day or multi-week investment. Your goal is to argue, with intraday price action, why the next 1-2 hours favor being short or flat, and to refute the intraday long case.

Key points to focus on:

- Momentum breakdown: Highlight loss of upside momentum — EMA9 rolling over or crossing below EMA21, bearish MACD crossover or fading histogram, RSI losing the 50 level or printing bearish divergence into recent highs.
- Rejection at VWAP / resistance: Emphasize price being rejected at or below VWAP (sellers in control), failure to reclaim VWAP, or stalling into intraday resistance, prior swing highs, round numbers, or the upper Bollinger band.
- Breakdown of intraday levels: Point to loss of intraday support, lower-high / lower-low structure, breakdowns through recent swing lows, and failed bounces.
- Exhaustion / over-extension: Note over-extension above VWAP or band tags that tend to mean-revert lower, climactic volume into highs, and thinning buy-side participation.
- Futures positioning: Weigh the funding rate, open interest, and order-book imbalance from the market report — crowded-long funding, falling OI on the bounce, or an ask-skewed book confirm short conviction or supply squeeze fuel; flag crowded-short funding as contrarian squeeze-up risk against the short.
- Long Counterpoints: Critically attack the bull's intraday long thesis with the price action — expose chases into resistance, weak-volume pushes, and bounces likely to fail within the next 1-2 hours.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Frame everything around the next 1-2 hours: directional bias, the level/setup that would trigger a short, and the intraday level (e.g. an ATR-based stop above resistance/VWAP) that would invalidate the short read.

Resources available:

{instrument_context}
Market research report (intraday technicals): {market_research_report}
Latest news / catalysts: {news_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Use this information to deliver a compelling intraday bear argument, refute the bull's intraday long claims, and engage in a dynamic debate that demonstrates why sellers are likely in control of the {target_label} for the next 1-2 hours.
""" + get_language_instruction() + get_horizon_instruction()
        else:
            prompt = f"""You are a Bear Analyst making the case against investing in the {target_label}. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the {target_label}.
""" + get_language_instruction() + get_horizon_instruction()

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
