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
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
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
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: float | None = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
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
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class EntryLeg(BaseModel):
    """One conditional entry scenario inside the Portfolio Manager's plan.

    Two kinds exist: a *pullback* leg (limit entry inside a retracement zone
    that is expected to hold) and a *breakout* leg (stop entry once price
    accepts beyond a level).  Legs are executed OCO — the first one whose
    condition triggers becomes the position and the other is cancelled — so
    each leg carries its own target and invalidation for its own scenario.
    """

    kind: Literal["pullback", "breakout"] = Field(
        description=(
            "'pullback' = enter on a retracement into a support/resistance "
            "zone expected to hold; 'breakout' = enter once price breaks and "
            "accepts beyond a trigger level."
        ),
    )
    zone_low: float | None = Field(
        default=None,
        description=(
            "Pullback legs only: lower bound of the entry zone in the "
            "instrument's quote currency (e.g. the 1837 of a 1837-1840 zone). "
            "Omit for breakout legs."
        ),
    )
    zone_high: float | None = Field(
        default=None,
        description=(
            "Pullback legs only: upper bound of the entry zone. "
            "Omit for breakout legs."
        ),
    )
    trigger: float | None = Field(
        default=None,
        description=(
            "Breakout legs only: the level whose break triggers the entry "
            "(above current price for a long, below for a short). "
            "EXECUTION: this is a stop-entry that fires the instant price "
            "TOUCHES it — a single wick is enough, there is no wait for a "
            "confirmed close. So if your thesis needs acceptance beyond a "
            "level, place the trigger past that level by a noise margin "
            "(a fraction of ATR) instead of exactly on it. "
            "Omit for pullback legs."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description=(
            "Target price for THIS entry scenario. It MUST be realistically "
            "reachable within the decision's stated time_horizon: on an "
            "intraday 1-2 hour horizon that means the FIRST support/"
            "resistance level ahead, typically within ~1-2 ATR of the entry "
            "— never a multi-day swing target. For a breakout entry, the "
            "measured extension just beyond the broken level. Omit the field "
            "when no sensible near target exists; execution then falls back "
            "to the decision's overall price_target and the mechanical "
            "bracket."
        ),
    )
    invalidation: float | None = Field(
        default=None,
        description=(
            "Price level that voids this scenario BEFORE entry (the pending "
            "order is cancelled if it trades there first) and, AFTER entry, "
            "becomes the position's actual STOP-LOSS. Below the entry for a "
            "long, above for a short. EXECUTION: the stop is a market order "
            "that fires the instant price TOUCHES this level — one wick ends "
            "the trade, there is no wait for a confirmed close. So when your "
            "thesis is 'invalid if price reclaims and HOLDS above X', do NOT "
            "put X here: place the level beyond X with room for noise "
            "(typically a fraction of ATR past it), otherwise a wick that "
            "never confirms will close the position. Give each leg the "
            "invalidation that fits ITS entry — a breakout leg is not "
            "automatically invalidated closer than a pullback leg. Widening "
            "the stop costs nothing in risk: position size is scaled down "
            "proportionally so the loss at the stop stays the same."
        ),
    )


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description=(
            "Optional target price in the instrument's quote currency. Must "
            "be realistically reachable within the stated time_horizon."
        ),
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    entry_legs: list[EntryLeg] | None = Field(
        default=None,
        description=(
            "Conditional entry plan for actionable ratings (skip for Hold). "
            "Up to two OCO legs — at most one pullback and one breakout — "
            "expressing WHERE the position should be entered rather than "
            "entering at market. Give each leg concrete price levels taken "
            "from the analysis (zone bounds / trigger, its own target, its "
            "invalidation). Omit entirely only when an immediate market "
            "entry at the current price is genuinely the intended execution."
        ),
    )


def render_entry_leg(leg: EntryLeg) -> str:
    """One deterministic ``- kind | ... `` line for an entry leg.

    The exact format is an interchange contract: the paper-sim's
    ``parse_entry_plan`` reads these lines back out of the rendered markdown,
    so keep field labels and separators stable.
    """
    chunks = [leg.kind]
    if leg.kind == "pullback" and leg.zone_low is not None and leg.zone_high is not None:
        lo, hi = sorted((leg.zone_low, leg.zone_high))
        chunks.append(f"zone {lo}-{hi}")
    if leg.kind == "breakout" and leg.trigger is not None:
        chunks.append(f"trigger {leg.trigger}")
    if leg.price_target is not None:
        chunks.append(f"target {leg.price_target}")
    if leg.invalidation is not None:
        chunks.append(f"invalidation {leg.invalidation}")
    return "- " + " | ".join(chunks)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle. The ``**Entry Plan**``
    block renders each conditional entry leg via :func:`render_entry_leg` in
    a deterministic shape the paper-sim parses back.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    if decision.entry_legs:
        parts.extend(["", "**Entry Plan**:"])
        parts.extend(render_entry_leg(leg) for leg in decision.entry_legs)
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
