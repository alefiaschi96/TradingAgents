from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from tradingagents.dataflows.config import get_config


def _daily_system_message(asset_label: str, prediction_markets: bool = True) -> str:
    pm = (
        " and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events)"
        if prediction_markets
        else ""
    )
    return (
        f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for {asset_label}-specific or targeted news searches, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'){pm}. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
    )


def _intraday_system_message(asset_label: str, prediction_markets: bool = True) -> str:
    pm = (
        " Use get_prediction_markets(topic, limit) when a live event (e.g. an imminent decision, listing vote, or rate decision) has a near-term resolution that could jolt price in the next couple of hours."
        if prediction_markets
        else ""
    )
    return (
        f"You are an **intraday catalyst scout** scanning for breaking news and events from the **last few hours** that can move price in the **next 1–2 hours**. Your job is NOT to summarize week-old macro — it is to surface fresh, price-moving catalysts and judge their immediate directional impact. Prioritize, in roughly this order: exchange hacks/exploits and protocol drains, listings/delistings and major exchange announcements, regulatory or enforcement headlines, large whale moves and notable on-chain flows (big transfers to/from exchanges, stablecoin mints/burns), exchange outages or withdrawal halts, and any sudden sentiment shift around the {asset_label}. "
        f"Use a SHORT lookback: call get_news(query, start_date, end_date) with start_date and end_date covering only the last 1 day, and get_global_news(curr_date, look_back_days, limit) with look_back_days=1 for the freshest broad headlines. Use get_macro_indicators only if a macro print is hitting RIGHT NOW and could spike intraday volatility — otherwise de-emphasize it.{pm} "
        f"For each catalyst, state: what happened, how recent it is, and whether it pushes price UP, DOWN, or is noise for the next 1–2 hours. Flag anything stale (more than a day old) as background context, not an active catalyst. If there is no fresh catalyst, say so plainly — silence is a valid intraday signal. Provide specific, actionable, time-sensitive insights to help a trader decide whether to be long, short, or flat for the next couple of hours."
    )


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        # Prediction markets can be turned off by setting the vendor to "none"
        # (e.g. where Polymarket is network-blocked). When off, drop the tool so
        # the model never calls it, and omit it from the instructions.
        pm_vendor = str(
            get_config().get("data_vendors", {}).get("prediction_markets", "polymarket")
        ).strip().lower()
        prediction_markets = pm_vendor not in ("none", "off", "disabled", "")

        tools = [
            get_news,
            get_global_news,
            get_macro_indicators,
        ]
        if prediction_markets:
            tools.append(get_prediction_markets)

        intraday = bool(get_config().get("intraday"))
        base_message = (
            _intraday_system_message(asset_label, prediction_markets)
            if intraday
            else _daily_system_message(asset_label, prediction_markets)
        )
        system_message = (
            base_message
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
