"""Paper-sim daemon.

Loop: every MONITOR_INTERVAL_SEC check the open position's SL/TP against the
live 1m candle; when flat and DECISION_INTERVAL_MIN has elapsed, run the
analysis and (maybe) open a simulated position. One position at a time.

Run:  python -m paper_sim.run     (needs GOOGLE_API_KEY; prices are public)
Stop: Ctrl-C. State persists, so it resumes where it left off.
"""

from __future__ import annotations

import logging
import os
import time

from live.config import Config
from live.run_logging import setup_run_logging
from paper_sim.simulator import PaperSimulator

_DEFAULT_STATE = os.path.expanduser("~/.tradingagents/paper_sim/state.json")


def main() -> int:
    cfg = Config.from_env()

    # One timestamped text log + JSONL event stream per run (this daemon = one
    # "session"); old runs are kept under ~/.tradingagents/paper_sim/logs/.
    run_log = setup_run_logging("paper_sim", symbol=cfg.symbol, meta=cfg.summary())
    log = logging.getLogger("paper_sim.run")

    sim = PaperSimulator(
        cfg,
        state_path=os.environ.get("PAPER_STATE_PATH", _DEFAULT_STATE),
        start_equity=float(os.environ.get("PAPER_START_EQUITY", cfg.paper_balance)),
        fee_pct_per_side=float(os.environ.get("FEE_PCT_PER_SIDE", "0.05")),
        slippage_pct_per_side=float(os.environ.get("SLIPPAGE_PCT_PER_SIDE", "0.02")),
        decision_interval_min=float(os.environ.get("DECISION_INTERVAL_MIN", "40")),
        analysis_min_gap_min=float(os.environ.get("ANALYSIS_MIN_GAP_MIN", "30")),
        event_sink=run_log.event,
        # Optional risk fixes — all default-off (see PaperSimulator.__init__).
        risk_pct_per_trade=float(os.environ.get("RISK_PCT_PER_TRADE", "0")),
        min_tp_cost_mult=float(os.environ.get("MIN_TP_COST_MULT", "0")),
        time_stop_hours=float(os.environ.get("TIME_STOP_HOURS", "0")),
        regime_persist_ticks=int(os.environ.get("REGIME_PERSIST_TICKS", "0")),
        regime_exit_check=os.environ.get("REGIME_EXIT_CHECK", "0") == "1",
    )
    monitor_interval = float(os.environ.get("MONITOR_INTERVAL_SEC", "60"))

    sim.connect()
    log.info(
        "paper-sim started | %s | LLM re-check >=%.0fmin, early trigger on regime flip "
        "(>=%.0fmin gap), monitor every %.0fs | %s",
        cfg.symbol, sim.decision_interval / 60, sim.analysis_min_gap / 60,
        monitor_interval, sim.summary(),
    )

    while True:
        try:
            now = time.time()
            if sim.has_open():
                sim.monitor()
            elif sim.should_decide(now):
                log.info("setup detected (flat) — running analysis, this takes a few minutes...")
                sim.maybe_decide(now)
            sim.save()
            log.info("STATUS %s", sim.summary())
        except KeyboardInterrupt:
            log.info("stopping — state saved")
            sim.save()
            run_log.close(summary=sim.summary())
            return 0
        except Exception as e:  # noqa: BLE001 - daemon must survive transient errors
            log.exception("loop error (continuing): %s", e)
        time.sleep(monitor_interval)


if __name__ == "__main__":
    raise SystemExit(main())
