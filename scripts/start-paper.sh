#!/usr/bin/env bash
#
# Start the paper-sim daemon AND the browser dashboard with a single command.
# The dashboard runs in the background; the daemon runs in the foreground and
# keeps the Mac awake (caffeinate). Ctrl-C stops BOTH.
#
# Usage:
#   ./scripts/start-paper.sh                 # dashboard on $DASHBOARD_PORT (default 8765)
#   DASHBOARD_PORT=9000 ./scripts/start-paper.sh
#
# Per-run logs are written automatically to logs/paper_sim/ (no tee needed).

# Always run from the repo root, regardless of where it is invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Activate the venv (heavy deps — langgraph/ccxt/yfinance — live there).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Load config (GOOGLE_API_KEY etc.). Prices are public; no Kraken key needed.
if [[ -f .env.live ]]; then
  set -a; source .env.live; set +a
else
  echo "warning: .env.live not found — the daemon needs GOOGLE_API_KEY" >&2
fi

PORT="${DASHBOARD_PORT:-8765}"

# Dashboard in the background (stdout muted; real errors still reach the terminal).
python -m paper_sim.dashboard "$PORT" 1>/dev/null &
DASH_PID=$!

# Stop the dashboard whenever this script exits — Ctrl-C, daemon stop, or error.
cleanup() { kill "$DASH_PID" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

echo "dashboard -> http://localhost:${PORT}   (logs: logs/paper_sim/)"
sleep 1
command -v open >/dev/null 2>&1 && open "http://localhost:${PORT}" 2>/dev/null || true

# Daemon in the foreground, Mac kept awake. Ctrl-C lands here -> the daemon
# saves state and exits -> the trap then stops the dashboard.
caffeinate -is python -m paper_sim.run
