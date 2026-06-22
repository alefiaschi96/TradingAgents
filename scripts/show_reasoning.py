"""Print the captured analyst reasoning from a per-run JSONL, readably.

Every decision logs an "analysis" event with the full chain (market + news
reports, bull/bear case, research manager, trader, the three risk debaters, and
the PM's final decision). This pretty-prints it so you can actually read why the
bot decided what it did.

Usage:
    python scripts/show_reasoning.py                 # latest analysis of the current run
    python scripts/show_reasoning.py --all           # every analysis in the current run
    python scripts/show_reasoning.py path/to.jsonl   # a specific run
    python scripts/show_reasoning.py --short          # truncate each field to 500 chars

Tip: pipe to a pager -> python scripts/show_reasoning.py | less -R
"""

from __future__ import annotations

import json
import os
import sys

_FIELDS = [
    ("MARKET ANALYST", "market_report"),
    ("NEWS / CATALYSTS", "news_report"),
    ("BULL", "bull_case"),
    ("BEAR", "bear_case"),
    ("RESEARCH MANAGER", "research_manager_plan"),
    ("TRADER", "trader_proposal"),
    ("RISK — aggressive", "risk_aggressive"),
    ("RISK — conservative", "risk_conservative"),
    ("RISK — neutral", "risk_neutral"),
    ("PORTFOLIO MANAGER (final)", "pm_decision"),
]


def _default_jsonl() -> str:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.environ.get("PAPER_LOG_DIR") or os.path.join(repo, "logs", "paper_sim")
    return os.path.join(log_dir, "latest.jsonl")


def _print_analysis(rec: dict, limit: int | None) -> None:
    print("=" * 78)
    print(f"ANALYSIS  rating={rec.get('rating')}  ts={rec.get('ts')}")
    print("=" * 78)
    for label, key in _FIELDS:
        v = (rec.get(key) or "").strip()
        print(f"\n----- {label} ({len(v)} chars) -----")
        if not v:
            print("(empty)")
        elif limit and len(v) > limit:
            print(v[:limit] + " …[truncated]")
        else:
            print(v)


def main() -> int:
    args = sys.argv[1:]
    show_all = "--all" in args
    limit = 500 if "--short" in args else None
    paths = [a for a in args if not a.startswith("--")]
    path = paths[0] if paths else _default_jsonl()

    if not os.path.exists(path):
        print(f"No JSONL found at {path}. Has a run produced a decision yet?")
        return 1

    analyses = [
        json.loads(line)
        for line in open(path)
        if line.strip() and json.loads(line).get("event") == "analysis"
    ]
    if not analyses:
        print(f"{os.path.basename(path)}: no analysis recorded yet "
              "(the run may still be on its first analysis).")
        return 0

    print(f"# {os.path.basename(os.path.realpath(path))} — {len(analyses)} analysis record(s)")
    for rec in (analyses if show_all else analyses[-1:]):
        _print_analysis(rec, limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
