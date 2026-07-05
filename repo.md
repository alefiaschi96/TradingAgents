# TradingAgents — Repository Map

## Architecture Overview

A multi-agent LLM framework for trading decisions built on **LangGraph**. Specialized agents (analysts, researchers, debaters, managers) collaborate through a graph workflow to produce a final trade signal. Supports multiple LLM providers, data vendors, live trading (Kraken Futures), and paper simulation.

```
Entry Points (CLI / Live / Paper / main.py)
        ↓
TradingAgentsGraph  →  LangGraph Workflow
        ↓
┌─────────────────────────────────────────────┐
│  Analysts (Market, Sentiment, News, Fundamentals)  │
│        ↓                                    │
│  Bull/Bear Debate → Research Manager        │
│        ↓                                    │
│  Trader (concrete proposal)                 │
│        ↓                                    │
│  Risk Debate (Aggressive/Conservative/Neutral) │
│        ↓                                    │
│  Portfolio Manager → Final Decision         │
└─────────────────────────────────────────────┘
        ↓
Data Tools → route_to_vendor() → yfinance / Alpha Vantage / FRED / etc.
```

---

## Top-Level Files

| File | Purpose |
|---|---|
| `main.py` | Minimal entry point. Imports `TradingAgentsGraph` + `DEFAULT_CONFIG`, calls `propagate("NVDA", date)` to run a full analysis cycle. |
| `test.py` | Quick functional test for `get_stock_stats_indicators_window()` from `tradingagents.dataflows.y_finance`. |
| `pyproject.toml` | Package metadata + dependencies (langchain, langgraph, yfinance, stockstats, typer, rich, etc.). |
| `docker-compose.yml` / `Dockerfile` / `Dockerfile.live` | Container configs for paper sim and live trading. |
| `requirements.txt` / `requirements-live.txt` | Pip dependency lists. |

---

## `tradingagents/` — Core Framework

### `default_config.py`
Central configuration with env-var overrides (`TRADINGAGENTS_*`). Defines LLM provider/models, data vendors per category, debate rounds, output language, temperature, checkpoint toggle, concurrency limits, intraday settings.

---

### `agents/` — Agent Implementations

#### `agents/analysts/`
Each analyst is a LangGraph node that calls data tools via LLM tool-calling, then synthesizes a report.

| File | Role | Data tools used |
|---|---|---|
| `market_analyst.py` | Technical analysis (MA, MACD, RSI, Bollinger, ATR, volume). Daily + intraday variants. | `get_stock_data`, `get_indicators`, `get_verified_market_snapshot` |
| `sentiment_analyst.py` | Retail/institutional sentiment. Pre-fetches news + StockTwits + Reddit, produces structured `SentimentReport`. | `get_news`, StockTwits, Reddit fetchers |
| `news_analyst.py` | News, macro, insider transactions, prediction markets synthesis. | `get_global_news`, `get_insider_transactions`, `get_macro_indicators`, `get_prediction_markets` |
| `fundamentals_analyst.py` | Balance sheet, income statement, cash flow, valuation. | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` |

#### `agents/researchers/`
Bull/bear debate team. Each reads analyst reports and argues a directional case.

| File | Role |
|---|---|
| `bull_researcher.py` | Bullish case — growth, competitive advantages, positive indicators. |
| `bear_researcher.py` | Bearish case — risks, overvaluation, negative catalysts. |

#### `agents/managers/`

| File | Role |
|---|---|
| `research_manager.py` | Synthesizes bull/bear debate into structured `InvestmentPlan` (Buy/Overweight/Hold/Underweight/Sell). Uses `deep_thinking_llm`. |
| `portfolio_manager.py` | Final synthesis: research plan + trader proposal + risk debate → `PortfolioDecision`. Produces the final trade signal. |

#### `agents/trader/`

| File | Role |
|---|---|
| `trader.py` | Converts `InvestmentPlan` into concrete `TraderProposal` (direction, size, entry, stops). |

#### `agents/risk_mgmt/`
Three-way risk debate before final decision.

| File | Role |
|---|---|
| `aggressive_debator.py` | Advocates taking trades, emphasizes opportunity cost of inaction. |
| `conservative_debator.py` | Advocates caution, capital preservation, position sizing. |
| `neutral_debator.py` | Balanced middle ground. |

#### `agents/utils/` — Shared Agent Infrastructure

| File | Purpose |
|---|---|
| `agent_states.py` | TypedDict state classes (`AgentState`, `InvestDebateState`, `RiskDebateState`) carrying messages, reports, debates through the workflow. |
| `agent_utils.py` | Central tool aggregator. Imports/exports all data tools. Provides `get_horizon_instruction()`, `get_scenario_instruction()` for intraday/language config. `create_msg_delete` for clearing message history. |
| `core_stock_tools.py` | `get_stock_data` tool (OHLCV via `route_to_vendor`). |
| `technical_indicators_tools.py` | `get_indicators` tool (technical analysis via vendors). |
| `fundamental_data_tools.py` | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`. |
| `news_data_tools.py` | `get_news`, `get_global_news`, `get_insider_transactions`. |
| `macro_data_tools.py` | `get_macro_indicators` (FRED integration). |
| `prediction_markets_tools.py` | `get_prediction_markets` (Polymarket). |
| `market_data_validation_tools.py` | `get_verified_market_snapshot` (ground-truth OHLCV for current date). |
| `futures_data_tools.py` | `get_funding_rate`, `get_open_interest`, `get_orderbook_imbalance` (Kraken Futures). |
| `schemas.py` | Pydantic models: `TraderProposal`, `ResearchPlan`, `PortfolioDecision`, `SentimentReport`. |
| `structured.py` | `bind_structured`, `invoke_structured_or_freetext` — structured output helpers with free-text fallback. |
| `memory.py` | `TradingMemoryLog` — append-only markdown log of decisions + reflections. `get_past_context()` injects prior lessons into future runs. |
| `rating.py` | `parse_rating()` — extracts Buy/Overweight/Hold/Underweight/Sell from decision text. |

---

### `dataflows/` — Data Vendor Integration

Abstraction layer. Tools call `route_to_vendor()` which delegates to the configured vendor.

| File | Purpose |
|---|---|
| `interface.py` | `route_to_vendor(tool_name, args)` — vendor registry and routing. Checks config to decide which vendor handles each tool. |
| `config.py` | `get_config()` / `set_config()` — runtime config management for data vendor selection. |
| `y_finance.py` | Yahoo Finance: OHLCV, indicators (via stockstats), fundamentals, insider transactions, news. |
| `alpha_vantage.py` | Alpha Vantage API: alternative vendor for OHLCV, indicators, fundamentals, news. |
| `fred.py` | Federal Reserve Economic Data: interest rates, inflation, employment, GDP. |
| `polymarket.py` | Prediction markets: market-implied event probabilities. |
| `crypto_intraday.py` | Kraken Futures: 1m/5m/15m candles for crypto perpetuals. |
| `reddit.py` | `fetch_reddit_posts` from r/wallstreetbets, r/stocks, r/investing. |
| `stocktwits.py` | `fetch_stocktwits_messages` — cashtag sentiment. |
| `yfinance_news.py` | `get_news_yfinance`, `get_global_news_yfinance`. |
| `symbol_utils.py` | Symbol normalization (e.g. XAUUSD → GC=F for yfinance). |
| `market_data_validator.py` | Stale data guards — ensures fetched data is current. |
| `stockstats_utils.py` | Technical indicator computation using `stockstats` library. |
| `errors.py` | `NoMarketDataError`, `VendorRateLimitError`, `VendorNotConfiguredError`. |

---

### `graph/` — LangGraph Orchestration

| File | Purpose |
|---|---|
| `trading_graph.py` | **Main class**: `TradingAgentsGraph`. Initializes LLMs (deep + quick), creates tool nodes, builds workflow via `GraphSetup`, exposes `propagate(ticker, date)`. Handles checkpoint integration, memory log, return/alpha calculation. |
| `setup.py` | `GraphSetup` — constructs the `StateGraph` workflow. Adds analyst nodes, researcher nodes, risk debate nodes, manager nodes. Connects them with conditional edges. |
| `propagation.py` | `Propagator` — creates initial `AgentState` from ticker/date/context. Manages state initialization and thread config for graph execution. |
| `analyst_execution.py` | `build_analyst_execution_plan()` — defines which analysts run and how (parallel batches vs sequential). `AnalystNodeSpec` per analyst type. |
| `conditional_logic.py` | `ConditionalLogic` — routing functions: `should_continue_debate` (tool loops for analysts), `should_continue_debate` / `should_continue_risk_analysis` (debate round control). |
| `checkpointer.py` | SQLite checkpoint management (per-ticker DBs). Enables resumable runs via `thread_id` from ticker+date. |
| `reflection.py` | `Reflector` — generates 2-4 sentence reflections on trading outcomes. Lessons stored in memory log for future injection. |
| `signal_processing.py` | `SignalProcessor` — extracts 5-tier rating (Buy/Overweight/Hold/Underweight/Sell) from Portfolio Manager output. |

---

### `llm_clients/` — Multi-Provider LLM Abstraction

| File | Purpose |
|---|---|
| `factory.py` | `create_llm_client(provider, model, ...)` — lazy-loads and routes to provider-specific client. |
| `base_client.py` | `BaseLLMClient` abstract class with `get_llm()` interface. |
| `openai_client.py` | OpenAI API client. Supports reasoning models (o1, gpt-4) with `reasoning_effort`. |
| `anthropic_client.py` | Claude client. Supports extended thinking with effort config. |
| `google_client.py` | Gemini client. Supports native thinking mode. |
| `azure_client.py` | Azure OpenAI client. |
| `bedrock_client.py` | AWS Bedrock client. |
| `api_key_env.py` | Environment variable key loading per provider. |
| `capabilities.py` | Model capability detection (feature support per provider). |
| `model_catalog.py` | Known models and their configurations. |
| `validators.py` | Input validation for LLM requests. |

---

## `cli/` — Command-Line Interface

| File | Purpose |
|---|---|
| `main.py` | Typer app entry point. Prompts for ticker, provider, model, analysts, depth, language. Calls `TradingAgentsGraph.propagate()`, displays results with Rich panels. |
| `utils.py` | Interactive prompts: `select_llm_provider`, `select_analysts`, `ask_openai_reasoning_effort`, `confirm_ollama_endpoint`, etc. |
| `config.py` | CLI-specific configuration handling. |
| `models.py` | Data models for CLI state. |
| `stats_handler.py` | `StatsCallbackHandler` — tracks LLM calls, token usage, tool execution time. |
| `announcements.py` | Display release announcements. |

---

## `live/` — Live Trading (Kraken Futures)

| File | Purpose |
|---|---|
| `runner.py` | Cron entry point. One decision cycle: check positions → run analysis → map decision to side → pre-trade guards → place bracket orders. |
| `analysis.py` | `run_analysis(cfg)` — executes `TradingAgentsGraph` for a live decision. `reasoning_from_state()` extracts decision data. |
| `bridge.py` | `ensure_protective_orders()` — manages SL/TP on existing positions. |
| `guards.py` | `map_decision_to_side()`, `check_pre_trade()` — validates against leverage caps, equity floors. |
| `kraken_client.py` | `KrakenClient` — Kraken Futures API wrapper (ccxt-based). |
| `config.py` | Live config from env vars (symbol, leverage, balance, allow_short). |
| `regime.py` | Market regime detection (uptrend/downtrend/ranging). |
| `run_logging.py` | JSONL event stream logging for audit trail. |

---

## `paper_sim/` — Paper Trading Simulator

| File | Purpose |
|---|---|
| `run.py` | Daemon loop. Monitors 1m candles, checks SL/TP, runs `TradingAgentsGraph` analysis on decision interval (~40min). |
| `simulator.py` | `PaperSimulator` — simulated positions, P&L, fills with slippage/fees, state persistence. |
| `dashboard.py` | Real-time monitoring dashboard. |
| `report.py` | Post-analysis backtest reports. |
| `watch.py` | Live feed monitoring. |

---

## `scripts/` — Utility Scripts

| File | Purpose |
|---|---|
| `show_reasoning.py` | Display agent reasoning traces. |
| `smoke_models.py` | Smoke test LLM provider connectivity. |
| `smoke_structured_output.py` | Smoke test structured output support. |
| `start-paper.sh` | Shell script to launch paper simulation. |

---

## `tests/` — Test Suite

Unit and integration tests covering vendor APIs, symbol normalization, checkpoint/resume, memory log, analyst execution plans, structured outputs, provider routing, CLI, temperature config, data validation, and more. Key files:

| File | Tests |
|---|---|
| `test_provider_registry.py` | LLM provider routing |
| `test_market_data_validator.py` | Stale data detection |
| `test_symbol_utils.py` | Symbol normalization (XAUUSD → GC=F) |
| `test_checkpoint_resume.py` | Resumable graph runs |
| `test_memory_log.py` | Decision log + reflection |
| `test_analyst_execution.py` | Analyst execution plan building |
| `test_structured_agents.py` | Structured output validation |
| `test_signal_processing.py` | Rating extraction |
| `conftest.py` | Shared fixtures and test configuration |

---

## Key Design Patterns

- **Multi-Agent Debate**: Bull/Bear → Research Manager synthesis; Aggressive/Conservative/Neutral → Portfolio Manager synthesis. Provides reasoning transparency.
- **Vendor Abstraction**: `route_to_vendor()` in `dataflows/interface.py` decouples tools from specific data sources. Config-driven vendor selection per data category.
- **Structured Output**: Agents produce Pydantic models (`TraderProposal`, `PortfolioDecision`, etc.) with free-text fallback for providers that lack structured output.
- **Memory & Reflection**: Append-only markdown log stores decisions. After outcomes resolve, `Reflector` generates lessons injected into future analyses via `past_context`.
- **LangGraph Workflow**: `StateGraph` with typed state, conditional edges, tool-calling loops, and SQLite checkpoints for resumability.
- **Intraday Mode**: `intraday: true` config activates alternate prompts directing agents to 1-2h scalp reasoning on leveraged perpetuals.
