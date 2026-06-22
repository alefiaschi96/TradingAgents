"""Per-run structured logging.

Every process gets its OWN timestamped text log plus a machine-readable JSONL
event stream, under ``~/.tradingagents/<component>/logs/`` (override with
``PAPER_LOG_DIR``). Old runs are never overwritten, so the full history stays on
disk for later analysis. Two stable symlinks — ``latest.log`` and
``latest.jsonl`` — always point at the current run, so the dashboards can find
it without knowing the timestamp.

Granularity (``bucket``):
- ``"session"`` — one file per process (the paper-sim daemon: one file per run).
- ``"day"``     — one file per UTC day, appended across invocations (the live
                  cron runner, which starts/exits every tick — per-tick files
                  would be noise; a day bucket keeps each day's ticks together).

The text log keeps the exact human format the project already uses; the JSONL
carries one object per event (decision / open / close / …) with a UTC timestamp
and the run id, so a later pandas/jq pass can reconstruct every run.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


def _default_log_dir(component: str) -> str:
    return os.path.expanduser(f"~/.tradingagents/{component}/logs")


def _slug(value) -> str:
    """Filesystem-safe token (keep alnum / - / _, replace the rest)."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(value))


def default_text_log(component: str = "paper_sim") -> str:
    """Path the dashboards should tail by default.

    Returns the ``latest.log`` symlink in the run-log dir when it exists, so
    ``python -m paper_sim.watch`` always follows the current run. Falls back to
    the legacy ``paper_sim.log`` in the cwd for backward compatibility.
    """
    log_dir = os.environ.get("PAPER_LOG_DIR") or _default_log_dir(component)
    latest = os.path.join(log_dir, "latest.log")
    if os.path.exists(latest):
        return latest
    return "paper_sim.log"


class RunLogger:
    """Owns a run's text + JSONL files and the ``latest.*`` symlinks."""

    def __init__(
        self,
        component: str,
        *,
        symbol=None,
        bucket: str = "session",
        log_dir: str | None = None,
        level=None,
        meta: dict | None = None,
    ):
        self.component = component
        sym = _slug(symbol) if symbol else "run"
        now = datetime.now(timezone.utc)
        if bucket == "day":
            self.run_id = f"{sym}-{now.strftime('%Y%m%d')}"
        else:  # "session"
            self.run_id = f"{sym}-{now.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"

        self.dir = log_dir or os.environ.get("PAPER_LOG_DIR") or _default_log_dir(component)
        os.makedirs(self.dir, exist_ok=True)
        self.text_path = os.path.join(self.dir, f"{self.run_id}.log")
        self.jsonl_path = os.path.join(self.dir, f"{self.run_id}.jsonl")

        self._configure_root_logging(level)
        self._symlink("latest.log", self.text_path)
        self._symlink("latest.jsonl", self.jsonl_path)

        logging.getLogger(component).info(
            "RUN %s started | logs: %s", self.run_id, self.dir
        )
        self.event("run_start", component=component, **(meta or {}))

    # ------------------------------------------------------------- internals
    def _configure_root_logging(self, level) -> None:
        """Attach stdout + per-run file handlers to the root logger.

        Clears pre-existing handlers first so a stray ``basicConfig`` elsewhere
        cannot double-log, and so re-running in one process replaces handlers
        cleanly rather than stacking them.
        """
        level = level or os.environ.get("LOG_LEVEL", "INFO")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
        root = logging.getLogger()
        root.setLevel(level)
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        file_handler = logging.FileHandler(self.text_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(stream_handler)
        root.addHandler(file_handler)

    def _symlink(self, name: str, target: str) -> None:
        """Point ``<dir>/<name>`` at ``target`` (relative). Best-effort: some
        filesystems / sandboxes disallow symlinks — never fail the run for it."""
        link = os.path.join(self.dir, name)
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.basename(target), link)
        except OSError:
            pass

    # --------------------------------------------------------------- events
    def event(self, event_type: str, **fields) -> None:
        """Append one JSON object (a single line) to the run's JSONL stream."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event_type,
        }
        record.update(fields)
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:  # noqa: BLE001 - logging must never crash the run
            logging.getLogger(self.component).warning(
                "could not write JSONL event %s: %s", event_type, e
            )

    def close(self, **fields) -> None:
        self.event("run_stop", **fields)


def setup_run_logging(
    component: str,
    *,
    symbol=None,
    bucket: str = "session",
    meta: dict | None = None,
) -> RunLogger:
    """Configure per-run logging and return the :class:`RunLogger`.

    ``component`` is the log namespace / subdir (``"paper_sim"`` or ``"live"``).
    Pass the instrument as ``symbol`` so the filename carries it (handy when
    running several instruments side by side).
    """
    return RunLogger(component, symbol=symbol, bucket=bucket, meta=meta)
