"""End-to-end smoke for structured-output agents against a real LLM provider.

Runs the four decision-making agents (Signal Synthesizer, Critic Manager,
Trader, Risk Manager) directly with their structured-output bindings and
prints the typed Pydantic instance + the rendered markdown for each. Use
this to verify a provider's native structured-output mode (json_schema for
OpenAI / xAI / DeepSeek / Qwen / GLM, response_schema for Gemini, tool-use
for Anthropic) returns clean instances on the schemas we ship.

Usage:
    OPENAI_API_KEY=... python scripts/smoke_structured_output.py openai
    GOOGLE_API_KEY=... python scripts/smoke_structured_output.py google
    ANTHROPIC_API_KEY=... python scripts/smoke_structured_output.py anthropic
    DEEPSEEK_API_KEY=... python scripts/smoke_structured_output.py deepseek

The script does NOT call propagate(), to keep the surface tight and the
cost low — it exercises only the four structured-output calls, plus the
heuristic SignalProcessor. The Risk Manager still calls the real
``get_sl_tp_levels`` tool (network I/O against the configured symbol), since
that determinism is the whole point of the agent.
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.agents.decision.critic_manager import create_critic_manager
from tradingagents.agents.decision.risk_manager import create_risk_manager
from tradingagents.agents.decision.signal_synthesizer import create_signal_synthesizer
from tradingagents.agents.trader.trader import create_trader
from tradingagents.graph.signal_processing import SignalProcessor
from tradingagents.llm_clients import create_llm_client

PROVIDER_DEFAULTS = {
    "openai": ("gpt-5.4-mini", None),
    "google": ("gemini-2.5-flash", None),
    "anthropic": ("claude-sonnet-4-6", None),
    "deepseek": ("deepseek-chat", None),
    "qwen": ("qwen-plus", None),
    "glm": ("glm-5", None),
    "xai": ("grok-4", None),
}

SYMBOL = "SOL-USD"

MARKET_REPORT = """Intraday technicals on SOL-USD (10m bars): price is above VWAP and EMA9 >
EMA21, RSI at 62 and rising, MACD histogram expanding. Funding rate is
mildly negative (crowded shorts), open interest rising into the move —
squeeze fuel for longs. Order book is bid-skewed."""

SENTIMENT_REPORT = """Social sentiment is Mildly Bullish (score 6.2/10, medium confidence):
StockTwits message volume up 30% with a 65/35 bullish/bearish ratio; Reddit
mentions modest but positive."""

NEWS_REPORT = """No major SOL-specific catalysts in the last few hours. General crypto
market news is neutral-to-positive on ETF flow commentary."""


def _make_synth_state():
    return {
        "company_of_interest": SYMBOL,
        "asset_type": "crypto",
        "market_report": MARKET_REPORT,
        "sentiment_report": SENTIMENT_REPORT,
        "news_report": NEWS_REPORT,
    }


def _make_critic_state(signal_decision: str):
    return {
        "company_of_interest": SYMBOL,
        "asset_type": "crypto",
        "signal_decision": signal_decision,
        "past_context": "",
    }


def _make_trader_state(critic_review: str):
    return {
        "company_of_interest": SYMBOL,
        "asset_type": "crypto",
        "critic_review": critic_review,
    }


def _make_risk_state(critic_review: str, trader_plan: str):
    return {
        "company_of_interest": SYMBOL,
        "asset_type": "crypto",
        "critic_review": critic_review,
        "trader_investment_plan": trader_plan,
    }


def _print_section(title: str, content: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}\n{content}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=list(PROVIDER_DEFAULTS.keys()))
    parser.add_argument("--deep-model", default=None, help="Override deep_think_llm")
    parser.add_argument("--quick-model", default=None, help="Override quick_think_llm")
    args = parser.parse_args()

    default_model, _ = PROVIDER_DEFAULTS[args.provider]
    deep_model = args.deep_model or default_model
    quick_model = args.quick_model or default_model

    print(f"Provider: {args.provider}")
    print(f"Deep model:  {deep_model}")
    print(f"Quick model: {quick_model}")

    # Build the LLM clients via the framework's factory.
    deep_client = create_llm_client(provider=args.provider, model=deep_model)
    quick_client = create_llm_client(provider=args.provider, model=quick_model)
    deep_llm = deep_client.get_llm()
    quick_llm = quick_client.get_llm()

    # 1) Signal Synthesizer
    synth = create_signal_synthesizer(deep_llm)
    synth_result = synth(_make_synth_state())
    signal_decision = synth_result["signal_decision"]
    _print_section("[1] Signal Synthesizer — signal_decision", signal_decision)

    # 2) Critic Manager (consumes the synthesizer's call)
    critic = create_critic_manager(deep_llm)
    critic_result = critic(_make_critic_state(signal_decision))
    critic_review = critic_result["critic_review"]
    _print_section("[2] Critic Manager — critic_review", critic_review)

    # 3) Trader (consumes the critic-reviewed signal)
    trader = create_trader(quick_llm)
    trader_result = trader(_make_trader_state(critic_review))
    trader_plan = trader_result["trader_investment_plan"]
    _print_section("[3] Trader — trader_investment_plan", trader_plan)

    # 4) Risk Manager (consumes both; calls get_sl_tp_levels for real)
    risk = create_risk_manager(deep_llm)
    risk_result = risk(_make_risk_state(critic_review, trader_plan))
    final_decision = risk_result["final_trade_decision"]
    _print_section("[4] Risk Manager — final_trade_decision", final_decision)

    # 5) SignalProcessor extracts the rating with zero LLM calls.
    sp = SignalProcessor()
    rating = sp.process_signal(final_decision)
    _print_section("[5] SignalProcessor → rating", rating)

    # 6) Lightweight checks: each rendered output should carry the expected
    #    section headers so downstream consumers (memory log, CLI display,
    #    saved reports, paper_sim/live's target gate) keep working.
    checks = [
        ("Signal Synthesizer", signal_decision, ["**Rating**:"]),
        ("Critic Manager",     critic_review,   ["**Verdict**:", "**Rating**:"]),
        ("Trader",             trader_plan,     ["**Action**:", "FINAL TRANSACTION PROPOSAL:"]),
        ("Risk Manager",       final_decision,  ["**Rating**:", "**Stop Loss**:", "**Take Profit**:", "**Position Size**:"]),
    ]
    print("\n" + "=" * 70 + "\nStructure checks\n" + "=" * 70)
    failures = 0
    for name, text, required in checks:
        for marker in required:
            ok = marker in text
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: contains {marker!r}")
            failures += int(not ok)

    print()
    if failures:
        print(f"Smoke FAILED: {failures} structure check(s) missing.")
        return 1
    print("Smoke PASSED: structured output → rendered markdown chain works for", args.provider)
    return 0


if __name__ == "__main__":
    sys.exit(main())
