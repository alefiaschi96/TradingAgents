"""Live, compact dashboard for the paper-sim — run it in a SECOND terminal:

    python -m paper_sim.watch                 # follows the current run (latest.log)
    python -m paper_sim.watch /path/to.log    # tail a specific / older run

Reads the state file (the numbers) and tails only the key lines of the log
(decisions, vetoes, opens/closes, regime changes). No network, no extra deps.
It also flags a likely freeze when the log has been silent too long. Ctrl-C
to quit. Refresh interval: WATCH_REFRESH_SEC (default 5s).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

from live.run_logging import default_text_log

_STATE = os.path.expanduser(
    os.environ.get("PAPER_STATE_PATH", "~/.tradingagents/paper_sim/state.json")
)
# Default to the current run's log (latest.log symlink in the run-log dir);
# pass an explicit path to tail an older run.
_LOG = sys.argv[1] if len(sys.argv) > 1 else default_text_log("paper_sim")
_REFRESH = float(os.environ.get("WATCH_REFRESH_SEC", "5"))
_STALE_MIN = float(os.environ.get("WATCH_STALE_MIN", "12"))  # warn if no log for this long

# Lines worth surfacing; plain per-60s STATUS is excluded (the header shows it).
_KEY = (
    "paper-sim started", "regime ready", "regime not ready", "setup detected",
    "Analysis decision for", "VETOED", "regime gate OK", "OPEN PF", "CLOSE ",
    "monitor open", "exceeded", "loop error", "falling back",
)


def _tail_lines(path: str, nbytes: int = 131072) -> list[str]:
    """Last chunk of the log as lines, robust to large files."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except FileNotFoundError:
        return []


def _last_timestamp(lines: list[str]) -> datetime | None:
    for line in reversed(lines):
        if len(line) >= 19 and line[:4].isdigit() and line[4] == "-":
            try:
                return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
    return None


def _state_block() -> list[str]:
    if not os.path.exists(_STATE):
        return ["state: (assente — primo ciclo non ancora salvato)"]
    try:
        s = json.load(open(_STATE))
    except (json.JSONDecodeError, OSError):
        return ["state: (in scrittura, riprovo...)"]
    eq = float(s.get("equity", 0.0))
    pnl = float(s.get("pnl_total", 0.0))
    w, l = int(s.get("wins", 0)), int(s.get("losses", 0))
    n = w + l
    start = eq - pnl
    ret = (pnl / start * 100.0) if start else 0.0
    wr = (w / n * 100.0) if n else 0.0
    out = [
        f"equity {eq:8.2f}    pnl {pnl:+7.2f} ({ret:+.2f}%)    "
        f"trades {n}  (W{w}/L{l}, win {wr:.0f}%)"
    ]
    op = s.get("open")
    if op:
        sp, rr = op.get("stop_pct"), op.get("rr")
        extra = f"  [stop {sp:.2f}% x rr {rr}]" if isinstance(sp, (int, float)) else ""
        out.append(
            f"POSITION  {str(op.get('side', '?')).upper():4}  entry {op.get('entry', 0):.4f}  "
            f"SL {op.get('sl', 0):.4f}   TP {op.get('tp', 0):.4f}{extra}"
        )
    else:
        out.append("POSITION  flat (in attesa di un setup)")
    return out


def render() -> str:
    lines = _tail_lines(_LOG)
    rows = ["=" * 72, " PAPER-SIM LIVE     (Ctrl-C per uscire)", "=" * 72]
    rows += [" " + r for r in _state_block()]

    ts = _last_timestamp(lines)
    if ts is not None:
        age = (datetime.now() - ts).total_seconds()
        mins = age / 60.0
        flag = f"   ⚠ nessun output da {mins:.0f}m — POSSIBILE FREEZE" if mins >= _STALE_MIN else ""
        rows.append(f" ultimo log: {ts.strftime('%H:%M:%S')} ({mins:.0f}m fa){flag}")

    rows.append("-" * 72)
    rows.append(" Ultimi eventi:")
    key = [ln for ln in lines if any(k in ln for k in _KEY)]
    if not key:
        rows.append("  (nessun evento chiave ancora)")
    else:
        for ln in key[-10:]:
            rows.append("  " + ln[-118:])
    return "\n".join(rows)


def main() -> int:
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(render(), flush=True)
            time.sleep(_REFRESH)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
