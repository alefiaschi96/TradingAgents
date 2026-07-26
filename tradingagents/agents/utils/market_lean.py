"""Deterministic read of the market analyst's directional stance.

Shared by two consumers:
- the in-graph market gate (``tradingagents.graph.setup``), which short-circuits
  the pipeline right after the market analyst when the report backs no trade
  (HOLD/flat or no usable data) — saving the debate/trader/PM LLM calls;
- the pre-trade analyst gate (``live.guards``), which vetoes an entry whose
  direction the market analyst's report doesn't back.

Both must read the report the same way, so the classifier lives here, once.
LLM reports use typographic apostrophes (can't), so patterns match both forms.
"""

from __future__ import annotations

import re

_APOS = "['’‘ʼ`´]?"

_FTP_RE = re.compile(
    r"FINAL TRANSACTION PROPOSAL:\s*\**\s*(BUY|SELL|HOLD|LONG|SHORT|FLAT)", re.I
)
_NO_DATA_RE = re.compile(
    rf"(?:can{_APOS}t|cannot|couldn{_APOS}t|unable\s+to)\s+"
    r"(?:produce|make|build|give|provide|generate|form|construct)\s+a\s+"
    r"(?:reliable|grounded|clean|solid|confident|meaningful|trustworthy)\b",
    re.I,
)
_HOLD_HEAD_RE = re.compile(
    r"(?:^|\n|:)\s*\**\s*(?:[A-Z0-9]{2,6}-[A-Z]{3,4}\s*:?\s*)?"
    r"\**(?:HOLD|FLAT|NO[- ]TRADE|STAY FLAT)\b",
    re.I,
)
_STANCE_ROW_RE = re.compile(
    r"(?:Trade stance|Trading stance|Trade decision|Best action|Stance)"
    r"\s*\|?\s*:?\s*([^|\n]+)",
    re.I,
)
_BIAS_ROW_RE = re.compile(
    r"(?:1[\s\-–—]?2\s?h(?:our)?s?\s+bias|Recommendation|Overall bias"
    r"|Net bias|Directional bias|bias)\s*[:|]\s*([^|\n]+)",
    re.I,
)

# Whole words only ("holds" must not read as "hold"); bull/bear are prefixes
# so bullish/bearish match too.
_NEUTRAL_PHRASE_RE = re.compile(
    r"\bneutral|\bflat\b|\bhold\b|\bwait\b|\bno[- ]trade\b|\bconsolidat", re.I
)
_BULL_PHRASE_RE = re.compile(r"\bbull|\blong\b|\bbuy\b|\bhigher\b", re.I)
_BEAR_PHRASE_RE = re.compile(r"\bbear|\bshort\b|\bsell\b|\blower\b|\bfad(?:e|ing)\b", re.I)


def _lean_from_phrase(text: str) -> str | None:
    """Directional lean of one stance/bias phrase, or None if ambiguous."""
    if _NEUTRAL_PHRASE_RE.search(text):
        return "neutral"
    bull = bool(_BULL_PHRASE_RE.search(text))
    bear = bool(_BEAR_PHRASE_RE.search(text))
    if bull and not bear:
        return "bull"
    if bear and not bull:
        return "bear"
    return None


def market_report_has_no_data(report: str) -> bool:
    """True when the analyst declared its market-data feed unusable."""
    text = (report or "").strip()
    if not text:
        return False
    return bool(_NO_DATA_RE.search(text)) or "data unavailable" in text[:300].lower()


def market_analyst_lean(report: str) -> str:
    """Classify the market analyst's stance: bull / bear / neutral / nodata.

    Deterministic text read of the analyst's own report, most explicit signal
    first: the FINAL TRANSACTION PROPOSAL tag, then a leading "data
    unavailable" or HOLD, then the summary-table stance/bias rows, and last a
    bullish-vs-bearish keyword count (which needs a clear margin to call a
    direction). Anything undecidable is "neutral" — no clear signal.
    """
    text = (report or "").strip()
    if not text:
        return "neutral"
    head = text[:300]

    m = _FTP_RE.search(text)
    if m:
        word = m.group(1).lower()
        if word in ("buy", "long"):
            return "bull"
        if word in ("sell", "short"):
            return "bear"
        return "neutral"
    if "data unavailable" in head.lower():
        return "nodata"
    if _HOLD_HEAD_RE.search(head):
        return "neutral"
    for pattern in (_STANCE_ROW_RE, _BIAS_ROW_RE):
        rows = pattern.findall(text)
        if rows:
            lean = _lean_from_phrase(rows[-1])
            if lean is not None:
                return lean
    low = text.lower()
    bull = low.count("bullish")
    bear = low.count("bearish") + low.count("fade")
    if bull > bear + 1:
        return "bull"
    if bear > bull + 1:
        return "bear"
    return "neutral"


_COND_TRIGGER_RE = re.compile(
    r"(?:re-?claim(?:s|ing|ed)?(?:\s+of)?|clears?|closes?\s+(?:above|below)|"
    r"break(?:s|ing)?\s+(?:above|below|over|of)?|accept(?:s|ance)?\s+above|"
    r"hold(?:s|ing)?\s+above|reject(?:s|ion)?\s+(?:of|at|near)|"
    r"pull\s?back\s+(?:toward|into|to))"
    r"[\s:\-]*\**\s*[~≈]?\$?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)",
    re.I,
)


def conditional_trigger_levels(report: str) -> list[float]:
    """Price levels the analyst names as would-change-my-mind triggers.

    A neutral report often carries an actionable condition in prose —
    "avoid new longs unless ETH reclaims 1898–1900", "wait for a pullback
    toward 1912" (observed verbatim in the July 2026 paper-sim logs). Under
    the conditional-entry executor those ARE tradeable plans, so the market
    gate can choose to let such a report continue to the debate/PM instead
    of short-circuiting. Pure text parse: deduped levels in citation order,
    [] when the report names none (the gate then behaves exactly as before).
    """
    out: list[float] = []
    for m in _COND_TRIGGER_RE.finditer(report or ""):
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return out


def market_gate_shortcut_reason(report: str) -> str | None:
    """Why the pipeline should stop right after the market analyst, or None.

    Returns ``"no reliable market data"`` / ``"HOLD/flat (no directional
    signal)"`` when the report backs no trade — the two cases the end-of-run
    analyst gate would veto regardless of what the downstream agents decide, so
    running them is pure cost. Directional reports (and empty ones: gate can't
    judge) return None and the pipeline continues.
    """
    text = (report or "").strip()
    if not text:
        return None
    if market_report_has_no_data(text):
        return "no reliable market data"
    if market_analyst_lean(text) in ("neutral", "nodata"):
        return "HOLD/flat (no directional signal)"
    return None
