#!/usr/bin/env bash
#
# Start a paper-sim instance (daemon + browser dashboard) for ONE instrument.
# Each instrument gets its OWN state, logs and dashboard port, so you can run
# several side by side (e.g. SOL and ADA at once).
#
# Usage:
#   ./scripts/start-paper.sh                    # symbol from .env.live, dashboard 8765
#   ./scripts/start-paper.sh PF_ADAUSD 8766     # ADA on its own dashboard
#
# The dashboard runs in the background and opens the browser itself; the daemon
# runs in the foreground under caffeinate. Ctrl-C stops BOTH.
# Per-run logs go to logs/paper_sim/<SLUG>/ ; state to ~/.tradingagents/paper_sim/<SLUG>/.

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

# Instrument + dashboard port (positional args override .env.live).
SYMBOL="${1:-${SYMBOL:-PF_SOLUSD}}"
PORT="${2:-${DASHBOARD_PORT:-8765}}"
# Readable per-instrument slug: PF_ADAUSD -> ADA, PF_SOLUSD -> SOL.
SLUG="$(printf '%s' "$SYMBOL" | sed 's/^PF_//; s/USD$//' | tr -cd 'A-Za-z0-9')"
[[ -n "$SLUG" ]] || SLUG="run"

# Isolate state + logs + port so multiple instruments never clash.
export SYMBOL
export DASHBOARD_PORT="$PORT"
export PAPER_STATE_PATH="$HOME/.tradingagents/paper_sim/$SLUG/state.json"
export PAPER_LOG_DIR="$PWD/logs/paper_sim/$SLUG"

# Dashboard in the background (it opens the browser itself; one tab).
python -m paper_sim.dashboard "$PORT" 1>/dev/null &
DASH_PID=$!

# Stop the dashboard whenever this script exits — Ctrl-C, daemon stop, or error.
cleanup() { kill "$DASH_PID" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

echo "instance $SLUG ($SYMBOL)  |  dashboard -> http://localhost:${PORT}  |  logs: $PAPER_LOG_DIR"

# Daemon in the foreground, Mac kept awake. Ctrl-C lands here -> the daemon
# saves state and exits -> the trap then stops the dashboard.
caffeinate -is python -m paper_sim.run
