"""Cron entry point: run one decision cycle for one instrument, then exit.

Flow (one cron tick):
  1. Connect to Kraken Futures.
  2. If a position is already open -> ensure it has SL/TP, then stop.
     (One position at a time: we never stack.)
  3. If flat -> cancel any orphan SL/TP, run the LLM analysis.
  4. Map the decision to a side; HOLD (or short while ALLOW_SHORT=false) -> stop.
  5. Pass pre-trade guards (leverage cap, equity floor).
  6. Open the position with a full-balance entry + SL/TP bracket.

Designed for Railway's Cron Schedule: starts, runs once, exits.
Set LIVE=true only when you want real orders; until then everything is logged.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from live.config import Config
from live import bridge, guards


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )


log = logging.getLogger("live.runner")


def _emit(event: str, **fields) -> None:
    """One-line structured log so Railway logs stay greppable / parseable."""
    log.info("EVENT %s %s", event, json.dumps(fields, default=str))


def run_once() -> dict:
    cfg = Config.from_env()
    _emit("config", **cfg.summary())

    api_key = os.environ.get("KRAKEN_API_KEY", "")
    api_secret = os.environ.get("KRAKEN_API_SECRET", "")
    if not cfg.no_broker and (not api_key or not api_secret):
        _emit("abort", reason="missing KRAKEN_API_KEY / KRAKEN_API_SECRET")
        return {"status": "abort", "reason": "missing kraken credentials"}

    # Imported here so a missing ccxt only fails when we actually trade,
    # not when importing config in tests.
    from live.kraken_client import KrakenClient

    kraken = KrakenClient(cfg, api_key, api_secret)
    kraken.connect()

    # 2. Already in a position -> do nothing but guarantee stops exist.
    position = kraken.get_open_position()
    if position:
        info = bridge.ensure_protective_orders(kraken, position, cfg)
        _emit(
            "position_open_skip",
            side=position.get("side"),
            contracts=position.get("contracts"),
            entry=position.get("entryPrice"),
            unrealized=position.get("unrealizedPnl"),
            liquidation=position.get("liquidationPrice"),
            reattach=info,
        )
        return {"status": "skip_position_open", "position": position}

    # 3. Flat -> clean up any stale protective orders, then analyse.
    cancelled = kraken.cancel_orphan_orders()
    if cancelled:
        _emit("orphan_cleanup", cancelled=cancelled)

    from live.analysis import run_analysis  # heavy import (LLM stack)
    rating, _state = run_analysis(cfg)

    # 4. Decision -> side.
    side = guards.map_decision_to_side(rating, cfg)
    if side is None:
        _emit("no_entry", rating=rating)
        return {"status": "no_entry", "rating": rating}

    # 5. Pre-trade guards.
    ok, reason = guards.check_pre_trade(kraken, cfg)
    if not ok:
        _emit("guard_blocked", rating=rating, side=side, reason=reason)
        return {"status": "guard_blocked", "reason": reason}

    # 6. Open.
    result = bridge.open_position(kraken, side, cfg)
    _emit("entry", rating=rating, **{k: v for k, v in result.items() if k != "orders"})
    return {"status": "opened" if result.get("opened") else "skipped", "result": result}


def main() -> int:
    _setup_logging()
    try:
        outcome = run_once()
        _emit("done", status=outcome.get("status"))
        return 0
    except Exception as e:  # noqa: BLE001 - top-level guard so cron exits cleanly
        log.exception("Run failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
