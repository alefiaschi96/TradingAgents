"""Tests for structured-output agents (Signal Synthesizer, Critic Manager, Trader, Sentiment Analyst).

The Risk Manager has its own coverage in tests/test_memory_log.py (Critic
Manager's past_context injection) and exercises get_sl_tp_levels directly
elsewhere. This file covers the schemas, render functions, and
graceful-fallback behavior shared by the Signal Synthesizer, Critic Manager,
Trader, and Sentiment Analyst so they share the same deterministic output
shape.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.decision.critic_manager import create_critic_manager
from tradingagents.agents.decision.signal_synthesizer import create_signal_synthesizer
from tradingagents.agents.schemas import (
    CriticVerdict,
    PortfolioRating,
    SentimentBand,
    SentimentReport,
    SignalDecision,
    TraderAction,
    TraderProposal,
    render_critic_verdict,
    render_sentiment_report,
    render_signal_decision,
    render_trader_proposal,
)
from tradingagents.agents.trader.trader import create_trader

# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderTraderProposal:
    def test_minimal_required_fields(self):
        p = TraderProposal(action=TraderAction.HOLD, reasoning="Balanced setup; no edge.")
        md = render_trader_proposal(p)
        assert "**Action**: Hold" in md
        assert "**Reasoning**: Balanced setup; no edge." in md
        # The trailing FINAL TRANSACTION PROPOSAL line is preserved for the
        # analyst stop-signal text and any external code that greps for it.
        assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in md

    def test_buy_and_sell_render(self):
        for action in (TraderAction.BUY, TraderAction.SELL):
            p = TraderProposal(action=action, reasoning="Some reasoning.")
            md = render_trader_proposal(p)
            assert f"**Action**: {action.value}" in md
            assert f"FINAL TRANSACTION PROPOSAL: **{action.value.upper()}**" in md


# ---------------------------------------------------------------------------
# Trader agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_trader_state():
    return {
        "company_of_interest": "NVDA",
        "critic_review": "**Verdict**: Approve\n\n**Rating**: Buy\n\n**Critique**: ...",
    }


def _structured_trader_llm(captured: dict, proposal: TraderProposal | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real TraderProposal so render_trader_proposal works.
    """
    if proposal is None:
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong setup.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or proposal
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestTraderAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="AI capex cycle intact; institutional flows constructive.",
        )
        llm = _structured_trader_llm(captured, proposal)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Action**: Buy" in plan
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in plan
        # The same rendered markdown is also added to messages for downstream agents.
        assert plan in result["messages"][0].content

    def test_prompt_includes_critic_review(self):
        captured = {}
        llm = _structured_trader_llm(captured)
        trader = create_trader(llm)
        trader(_make_trader_state())
        # The critic-reviewed signal is in the user message of the captured prompt.
        prompt = captured["prompt"]
        assert any("Critic-Reviewed Signal" in m["content"] for m in prompt)

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = (
            "**Action**: Sell\n\nGuidance cut hits margins.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert result["trader_investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Signal Synthesizer agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_synth_state():
    return {
        "company_of_interest": "NVDA",
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
    }


def _structured_synth_llm(captured: dict, decision: SignalDecision | None = None):
    if decision is None:
        decision = SignalDecision(
            rating=PortfolioRating.HOLD,
            rationale="Balanced view across sources.",
            key_evidence="No strong agreement across analysts.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or decision
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestRenderSignalDecision:
    def test_required_fields(self):
        d = SignalDecision(
            rating=PortfolioRating.OVERWEIGHT,
            rationale="Bull evidence carried; tailwinds intact.",
            key_evidence="Price above VWAP; funding crowded short.",
        )
        md = render_signal_decision(d)
        assert "**Rating**: Overweight" in md
        assert "**Rationale**: Bull evidence carried" in md
        assert "**Key Evidence**: Price above VWAP" in md

    def test_all_5_tier_ratings_render(self):
        for rating in PortfolioRating:
            d = SignalDecision(rating=rating, rationale="r", key_evidence="e")
            md = render_signal_decision(d)
            assert f"**Rating**: {rating.value}" in md


@pytest.mark.unit
class TestSignalSynthesizerAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        decision = SignalDecision(
            rating=PortfolioRating.OVERWEIGHT,
            rationale="Bull case is stronger; momentum intact.",
            key_evidence="EMA9 > EMA21; funding crowded short.",
        )
        llm = _structured_synth_llm(captured, decision)
        synth = create_signal_synthesizer(llm)
        result = synth(_make_synth_state())
        sd = result["signal_decision"]
        assert "**Rating**: Overweight" in sd
        assert "**Rationale**: Bull case" in sd
        assert "**Key Evidence**: EMA9" in sd

    def test_prompt_uses_5_tier_rating_scale(self):
        captured = {}
        llm = _structured_synth_llm(captured)
        synth = create_signal_synthesizer(llm)
        synth(_make_synth_state())
        prompt = captured["prompt"]
        for tier in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            assert f"**{tier}**" in prompt, f"missing {tier} in prompt"

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = "**Rating**: Sell\n\n**Rationale**: ...\n\n**Key Evidence**: ..."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        synth = create_signal_synthesizer(llm)
        result = synth(_make_synth_state())
        assert result["signal_decision"] == plain_response


# ---------------------------------------------------------------------------
# Critic Manager agent: render function + structured happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderCriticVerdict:
    def test_required_fields(self):
        v = CriticVerdict(
            verdict="Downgrade",
            rating=PortfolioRating.OVERWEIGHT,
            critique="Edge is real but weaker than the Buy conviction claimed.",
        )
        md = render_critic_verdict(v)
        assert "**Verdict**: Downgrade" in md
        assert "**Rating**: Overweight" in md
        assert "**Critique**: Edge is real" in md


def _make_critic_state():
    return {
        "company_of_interest": "NVDA",
        "signal_decision": "**Rating**: Buy\n\n**Rationale**: ...\n\n**Key Evidence**: ...",
        "past_context": "",
    }


@pytest.mark.unit
class TestCriticManagerAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        verdict = CriticVerdict(
            verdict="Approve",
            rating=PortfolioRating.BUY,
            critique="Multiple sources agree; edge exceeds noise.",
        )
        structured = MagicMock()
        structured.invoke.side_effect = lambda prompt: (
            captured.__setitem__("prompt", prompt) or verdict
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        critic = create_critic_manager(llm)
        result = critic(_make_critic_state())
        cr = result["critic_review"]
        assert "**Verdict**: Approve" in cr
        assert "**Rating**: Buy" in cr


# ---------------------------------------------------------------------------
# Sentiment Analyst: schema, render, structured happy path + fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderSentimentReport:
    def test_header_contains_band_and_score(self):
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.2,
            confidence="high",
            narrative="Source breakdown here.",
        )
        md = render_sentiment_report(report)
        assert "**Overall Sentiment:** **Bullish**" in md
        assert "(Score: 7.2/10)" in md

    def test_header_contains_confidence(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL,
            overall_score=5.0,
            confidence="low",
            narrative="Limited data.",
        )
        assert "**Confidence:** Low" in render_sentiment_report(report)

    def test_narrative_preserved_in_output(self):
        narrative = "## Breakdown\n\nStockTwits: 70% bullish.\n\n| Signal | Direction |\n|---|---|\n| News | Neutral |"
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BULLISH,
            overall_score=6.0,
            confidence="medium",
            narrative=narrative,
        )
        assert narrative in render_sentiment_report(report)

    def test_all_six_bands_render(self):
        for band in SentimentBand:
            report = SentimentReport(
                overall_band=band, overall_score=5.0,
                confidence="medium", narrative="n",
            )
            assert band.value in render_sentiment_report(report)

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                overall_band=SentimentBand.BULLISH, overall_score=11.0,
                confidence="high", narrative="n",
            )


def _make_sentiment_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "asset_type": "stock",
        "messages": [],
    }


def _structured_sentiment_llm(captured: dict, report: SentimentReport | None = None):
    """MagicMock LLM whose structured binding captures the prompt and returns
    a real SentimentReport so render_sentiment_report works."""
    if report is None:
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH, overall_score=7.5,
            confidence="high",
            narrative="StockTwits 75% bullish. News constructive. Reddit upbeat.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or report
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestSentimentAnalystAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BEARISH, overall_score=4.0,
            confidence="medium", narrative="Mixed signals across sources.",
        )
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured, report))
        sr = analyst(_make_sentiment_state())["sentiment_report"]
        assert "**Overall Sentiment:** **Mildly Bearish**" in sr
        assert "(Score: 4.0/10)" in sr
        assert "Mixed signals across sources." in sr

    def test_sentiment_report_also_in_messages(self):
        captured = {}
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured))
        result = analyst(_make_sentiment_state())
        assert len(result["messages"]) == 1
        assert result["sentiment_report"] == result["messages"][0].content

    def test_prompt_contains_ticker(self):
        captured = {}
        create_sentiment_analyst(_structured_sentiment_llm(captured))(_make_sentiment_state())
        assert any("NVDA" in str(m) for m in captured["prompt"])

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain = "**Overall Sentiment:** **Bearish** (Score: 3.0/10)\n**Confidence:** Low\n\nLimited data."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain

    def test_falls_back_to_freetext_when_structured_call_fails(self):
        plain = "Fallback free-text sentiment."
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON from model")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain
