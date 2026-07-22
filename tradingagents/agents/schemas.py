"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Signal Synthesizer
# ---------------------------------------------------------------------------


class SignalDecision(BaseModel):
    """Structured directional call produced by the Signal Synthesizer.

    Single-pass replacement for the old bull/bear debate + Research Manager:
    reads the analysts' reports directly and commits to a rating in one shot,
    with no back-and-forth rounds.
    """

    rating: PortfolioRating = Field(
        description=(
            "The directional call. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell. Reserve Hold for situations where the "
            "evidence is genuinely balanced; otherwise commit to the side "
            "with the stronger intraday evidence."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational synthesis of the analysts' reports, ending with "
            "which evidence led to this rating. Speak naturally, as if to a "
            "teammate."
        ),
    )
    key_evidence: str = Field(
        description=(
            "The 2-4 strongest, most concrete pieces of evidence (specific "
            "price levels, indicator readings, positioning data, or news) "
            "backing this call."
        ),
    )


def render_signal_decision(decision: SignalDecision) -> str:
    """Render a SignalDecision to markdown for storage and downstream prompts."""
    return "\n".join([
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Rationale**: {decision.rationale}",
        "",
        f"**Key Evidence**: {decision.key_evidence}",
    ])


# ---------------------------------------------------------------------------
# Critic Manager
# ---------------------------------------------------------------------------


class CriticVerdict(BaseModel):
    """Structured critique produced by the Critic Manager.

    The critic's job is to doubt the Signal Synthesizer's call: is the
    predicted move actually big enough, clean enough, and well-supported
    enough to be worth trading, once noise and conflicting signals are
    accounted for? It may only hold or soften conviction — it never flips
    the direction the Synthesizer picked.
    """

    verdict: Literal["Approve", "Downgrade", "Veto"] = Field(
        description=(
            "Approve: the edge is real and worth trading at the Synthesizer's "
            "conviction. Downgrade: there is some edge but it's weaker than "
            "claimed (conflicting signals, noise-sized move, thin evidence) — "
            "soften conviction one notch (e.g. Buy -> Overweight) but keep "
            "the direction. Veto: the case is not worth trading at all this "
            "round (no real edge, or evidence too conflicting) — force Hold."
        ),
    )
    rating: PortfolioRating = Field(
        description=(
            "The rating after applying the verdict: unchanged from the "
            "Synthesizer on Approve, one notch softer on Downgrade (same "
            "direction), or Hold on Veto."
        ),
    )
    critique: str = Field(
        description=(
            "2-4 sentences explaining what was scrutinized (edge size, "
            "signal agreement, data quality) and why the verdict follows."
        ),
    )


def render_critic_verdict(verdict: CriticVerdict) -> str:
    """Render a CriticVerdict to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Verdict**: {verdict.verdict}",
        "",
        f"**Rating**: {verdict.rating.value}",
        "",
        f"**Critique**: {verdict.critique}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the critic-reviewed signal. Two to four sentences. Do not "
            "propose entry, stop-loss, or position size — the Risk Manager "
            "computes those deterministically from ATR."
        ),
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------


class PositionSize(str, Enum):
    """Conviction-scaled position-size tier.

    Informational only: no live equity figure is available at analysis time,
    so this is a qualitative sizing recommendation for the report/logs, not a
    dollar amount. Actual order sizing in live/paper execution keeps using
    its own config-driven logic.
    """

    FULL = "Full"
    HALF = "Half"
    QUARTER = "Quarter"
    NONE = "None"


class RiskDecision(BaseModel):
    """Structured output produced by the Risk Manager.

    The Risk Manager is the only agent that speaks to stop-loss, take-profit,
    and position size — and it must source stop_loss/take_profit from the
    deterministic ATR tool (``get_sl_tp_levels``), never invent them. The LLM
    call that fills this schema is instructed to copy the tool's numbers
    verbatim and only decide the final rating and position-size tier.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell — normally the Critic Manager's rating, or Hold "
            "if the critic vetoed the trade."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Stop-loss price from the ATR tool output, copied verbatim. None "
            "when the rating is Hold."
        ),
    )
    take_profit: float | None = Field(
        default=None,
        description=(
            "Take-profit price from the ATR tool output (2R target), copied "
            "verbatim. None when the rating is Hold."
        ),
    )
    position_size: PositionSize = Field(
        description=(
            "Conviction-scaled size tier: Full for Buy/Sell, Half for "
            "Overweight/Underweight or when the critic downgraded conviction, "
            "None for Hold."
        ),
    )
    risk_rationale: str = Field(
        description=(
            "2-4 sentences explaining the SL/TP levels (ATR-based or "
            "structural) and the position-size tier chosen."
        ),
    )


def render_risk_decision(decision: RiskDecision) -> str:
    """Render a RiskDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved reports all read this markdown;
    ``**Rating**`` must stay the first header so ``rating.py``'s
    ``parse_rating`` keeps working unchanged. ``**Price Target**`` mirrors
    ``take_profit`` — ``paper_sim``/``live``'s target-gate veto
    (``parse_price_target`` / ``min_analyst_rr``) reads that exact label, so
    keeping it here means the deterministic ATR take-profit now grounds that
    gate instead of an LLM-guessed price target, with no execution-side code
    changes required.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Stop Loss**: {decision.stop_loss if decision.stop_loss is not None else 'n/a'}",
        "",
        f"**Take Profit**: {decision.take_profit if decision.take_profit is not None else 'n/a'}",
    ]
    if decision.take_profit is not None:
        parts.extend(["", f"**Price Target**: {decision.take_profit}"])
    parts.extend([
        "",
        f"**Position Size**: {decision.position_size.value}",
        "",
        f"**Risk Rationale**: {decision.risk_rationale}",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
