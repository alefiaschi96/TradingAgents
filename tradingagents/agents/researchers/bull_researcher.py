from tradingagents.agents.utils.agent_utils import (
    get_horizon_instruction,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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
            prompt = f"""You are a Bull Analyst building the **intraday LONG case** for the {target_label} over the **next 1-2 hours** on {timeframe} bars. This is a short-term trade, NOT a multi-day or multi-week investment. Your task is to argue that buyers are in control and that price is likely to push higher in the next couple of hours. Use the provided intraday data to make your case and to refute the intraday short case.

Key points to focus on:
- Momentum Continuation: Argue that the current up-move has room to extend over the next 1-2 hours — rising momentum (MACD, RSI), expanding range, and recent higher highs/higher lows.
- Trend Strength: Emphasize bullish trend alignment — price above EMA9 and EMA21, EMA9 above EMA21, and price holding above VWAP (intraday buyers in control).
- Favorable Intraday Levels: Point to the price sitting above key intraday support (recent swing lows, VWAP, round numbers) with clear room to the next resistance, and breakout potential above recent swing highs or band edges.
- Futures Positioning: Weigh the positioning signals — crowded-short (negative) funding plus rising open interest behind the up-move and a bid-skewed order book all confirm or even add squeeze fuel to the LONG, while strongly positive (crowded-long) funding is a contrarian risk to flag.
- Bear Counterpoints: Critically rebut the intraday short case (overbought RSI, VWAP fade, mean-reversion) with specific price-action evidence, showing why upside continuation is the higher-probability read for the next 1-2 hours.
- Risk Frame: Note the ATR-based level just below structure (e.g. below VWAP or the last swing low) that would invalidate the long thesis — without it, the trade is naked.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Do NOT use growth, revenue, valuation, scalability, or fundamentals framing — keep everything anchored to intraday price action.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Latest world affairs news: {news_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling intraday bull argument, refute the bear's intraday short case, and engage in a dynamic debate that demonstrates the strength of the long position for the next 1-2 hours.
""" + get_language_instruction() + get_horizon_instruction()
        else:
            prompt = f"""You are a Bull Analyst advocating for investing in the {target_label}. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
""" + get_language_instruction() + get_horizon_instruction()

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
