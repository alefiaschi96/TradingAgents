"""Readable summary of the paper-sim results.

Run anytime (even while the daemon is running):
    python -m paper_sim.report
    python -m paper_sim.report /path/to/state.json
"""

from __future__ import annotations

import json
import os
import sys

_DEFAULT_STATE = os.path.expanduser("~/.tradingagents/paper_sim/state.json")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAPER_STATE_PATH", _DEFAULT_STATE)
    if not os.path.exists(path):
        print(f"No state file at {path}\nRun `python -m paper_sim.run` first.")
        return 1

    s = json.load(open(path))
    closed = s.get("closed", [])
    equity = float(s.get("equity", 0.0))
    pnl = float(s.get("pnl_total", 0.0))
    wins, losses = int(s.get("wins", 0)), int(s.get("losses", 0))
    n = len(closed)
    start_eq = equity - pnl
    ret = (pnl / start_eq * 100.0) if start_eq else 0.0
    win_rate = (wins / n * 100.0) if n else 0.0
    win_pnls = [t["pnl"] for t in closed if t.get("pnl", 0) >= 0]
    loss_pnls = [t["pnl"] for t in closed if t.get("pnl", 0) < 0]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    expectancy = (pnl / n) if n else 0.0
    tp_count = sum(1 for t in closed if t.get("outcome") == "TP")
    sl_count = sum(1 for t in closed if t.get("outcome") == "SL")
    soft_count = sum(1 for t in closed if t.get("outcome") == "SOFT")

    line = "=" * 60
    print(line)
    print(f" PAPER-SIM REPORT   {path}")
    print(line)
    print(f" Equity:      {start_eq:.2f}  ->  {equity:.2f}   ({ret:+.2f}%)")
    print(f" Total P&L:   {pnl:+.2f}")
    print(f" Trades:      {n}   (W {wins} / L {losses}, win rate {win_rate:.0f}%)")
    soft_txt = f" / SOFT {soft_count}" if soft_count else ""
    print(f" Outcomes:    TP {tp_count} / SL {sl_count}{soft_txt}")
    print(f" Avg win:     {avg_win:+.2f}    Avg loss: {avg_loss:+.2f}")
    print(f" Expectancy:  {expectancy:+.2f} per trade")
    op = s.get("open")
    if op:
        print(f" Open now:    {op['side']} @ {op['entry']:.4f}  (SL {op['sl']:.4f}  TP {op['tp']:.4f})")
    else:
        print(" Open now:    flat")
    print("-" * 60)
    print(" Recent trades (last 12):")
    if not closed:
        print("  (no closed trades yet)")
    for t in closed[-12:]:
        ts = str(t.get("closed_at", ""))[:16]
        print(
            f"  {ts}  {t.get('side',''):4} {str(t.get('rating','')):11} "
            f"{t.get('entry',0):.3f} -> {t.get('outcome','')} {t.get('exit',0):.3f}  "
            f"pnl {t.get('pnl',0):+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
