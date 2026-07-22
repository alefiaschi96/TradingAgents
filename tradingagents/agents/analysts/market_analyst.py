from datetime import datetime, timezone

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_indicators,
    get_instrument_context_from_state,
    get_language_instruction,
    get_stock_data,
    get_verified_market_snapshot,
)
from tradingagents.agents.utils.futures_data_tools import (
    get_funding_rate,
    get_open_interest,
    get_orderbook_imbalance,
)
from tradingagents.dataflows.config import get_config

# --- Daily (default) indicator guide -------------------------------------
_DAILY_SYSTEM_MESSAGE = """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names.

Before writing the final report, call get_verified_market_snapshot for this ticker and the current date, and treat it as the source of truth for any exact OHLCV, price-level, or indicator-value claim. If another tool's output conflicts with the verified snapshot, flag the discrepancy rather than inventing a reconciled number. Do not claim historical validation, support/resistance bounces, or exact percentage moves unless they are directly supported by tool output with concrete dates and prices.

Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."""


# --- Intraday indicator guide (Path 1) -----------------------------------
def _intraday_system_message(timeframe: str) -> str:
    return f"""You are an **intraday technical analyst** trading **{timeframe} bars**. Your decision horizon is the **next 1–2 hours** — NOT a multi-day or multi-week investment. The latest bar is the current moment; everything you read should inform whether to be long, short, or flat for the next couple of hours.

All indicator periods below are in **bars**, not days (e.g. a 9-period EMA spans 9×{timeframe}). Select up to **8** complementary indicators (avoid redundancy). Use the EXACT names below.

Trend (fast):
- close_9_ema: 9-bar EMA — fast intraday trend. Price above it = short-term upward pressure.
- close_21_ema: 21-bar EMA — intraday trend filter. EMA9 crossing ABOVE EMA21 = bullish momentum; crossing below = bearish.

Fair value / volume:
- vwap: rolling 24h volume-weighted average price — the intraday **fair-value anchor**. Price ABOVE VWAP = intraday buyers in control (long bias); BELOW = sellers in control (short bias). Stretched distance from VWAP often mean-reverts.
- vwma: volume-weighted MA — confirms whether a move has real volume behind it.

Momentum:
- rsi: 14-bar RSI — intraday overbought (>70) / oversold (<30). Watch divergence vs price for reversals; in strong intraday trends it can stay extreme.
- macd / macds / macdh: 12/26/9-bar MACD — crossovers and histogram flips flag intraday momentum shifts early.

Volatility / levels:
- boll / boll_ub / boll_lb: 20-bar Bollinger Bands — band tags flag intraday over-extension; squeezes precede breakouts.
- atr: 14-bar ATR — current intraday volatility. Use it to judge whether a move is significant vs noise and how far a stop must realistically sit.

Futures market-structure (Kraken Futures, public): also call get_funding_rate, get_open_interest, and get_orderbook_imbalance for this symbol. Read them as: very positive/negative **funding** = the crowd is heavily on one side (squeeze risk against that side); **open interest** rising during a move = fresh money behind a "real" move, while falling OI = a fading move; an **order-book** skewed to bids/asks = short-term buying/selling pressure. These are positioning signals, not price levels — use only what the tools return and never fabricate values.

Workflow: call get_stock_data first (recent {timeframe} OHLCV incl. VWAP), then get_indicators once per chosen indicator (exact names), then get_verified_market_snapshot for ground-truth values. Treat the verified snapshot as the source of truth for any exact price level or indicator value — never invent numbers, support/resistance bounces, or percentage moves.

Then write a focused report for the NEXT 1–2 HOURS covering:
- intraday trend & momentum (EMA9/21 alignment, MACD, RSI),
- position vs VWAP (bias + how stretched),
- key intraday levels: recent swing highs/lows, band edges, round numbers,
- whether this is a breakout/continuation or a mean-reversion setup,
- volatility (ATR) and the level that would invalidate the read.
Be explicit about the short-term directional bias."""


def create_market_analyst(llm):

    def market_analyst_node(state):
        intraday = bool(get_config().get("intraday"))
        timeframe = get_config().get("intraday_timeframe", "15m")
        instrument_context = get_instrument_context_from_state(state)

        # Intraday injects the current time so the analyst anchors to "now".
        if intraday:
            now = datetime.now(timezone.utc).strftime("%H:%M")
            current_date = f"{state['trade_date']} {now} UTC"
        else:
            current_date = state["trade_date"]

        tools = [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
        ]
        # Intraday adds Kraken Futures market-structure tools (public, no keys):
        # funding, open interest, and top-of-book imbalance. The daily path is
        # untouched.
        if intraday:
            tools += [
                get_funding_rate,
                get_open_interest,
                get_orderbook_imbalance,
            ]

        base_message = _intraday_system_message(timeframe) if intraday else _DAILY_SYSTEM_MESSAGE
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
            "market_report": report,
        }

    return market_analyst_node
