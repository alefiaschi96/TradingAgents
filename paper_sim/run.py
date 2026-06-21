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
import sys
import time

from live.config import Config
from paper_sim.simulator import PaperSimulator

_DEFAULT_STATE = os.path.expanduser("~/.tradingagents/paper_sim/state.json")


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("paper_sim.run")

    cfg = Config.from_env()
    sim = PaperSimulator(
        cfg,
        state_path=os.environ.get("PAPER_STATE_PATH", _DEFAULT_STATE),
        start_equity=float(os.environ.get("PAPER_START_EQUITY", cfg.paper_balance)),
        fee_pct_per_side=float(os.environ.get("FEE_PCT_PER_SIDE", "0.05")),
        slippage_pct_per_side=float(os.environ.get("SLIPPAGE_PCT_PER_SIDE", "0.02")),
        decision_interval_min=float(os.environ.get("DECISION_INTERVAL_MIN", "40")),
    )
    monitor_interval = float(os.environ.get("MONITOR_INTERVAL_SEC", "60"))

    sim.connect()
    log.info(
        "paper-sim started | %s every %.0fmin, monitor every %.0fs | %s",
        cfg.symbol, sim.decision_interval / 60, monitor_interval, sim.summary(),
    )

    while True:
        try:
            now = time.time()
            if sim.has_open():
                sim.monitor()
            elif sim.decision_due(now):
                log.info("decision due (flat) — running analysis, this takes a few minutes...")
                sim.maybe_decide(now)
            sim.save()
            log.info("STATUS %s", sim.summary())
        except KeyboardInterrupt:
            log.info("stopping — state saved")
            sim.save()
            return 0
        except Exception as e:  # noqa: BLE001 - daemon must survive transient errors
            log.exception("loop error (continuing): %s", e)
        time.sleep(monitor_interval)


if __name__ == "__main__":
    raise SystemExit(main())
