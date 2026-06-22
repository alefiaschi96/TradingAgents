# paper_sim/ — live paper-trading simulator

Measures whether the bot's signals would make money on **real Kraken prices**,
with **zero risk** and **without touching** `tradingagents/` or `live/`.

## What it does

A single daemon loop:
- **Every minute** — if a simulated position is open, read the live 1-minute
  candle (high/low, so intra-minute wicks count) and close it if it hit the
  **stop-loss** or **take-profit**, recording the outcome and P&L.
- **Every 40 minutes** — if **flat**, run the analysis and (if it says
  long/short) open a **simulated** position with the same entry/SL/TP/size the
  live bridge would use. One position at a time.

No order is ever sent. Equity starts at €100 (~$108) and **compounds** with each
closed trade, so you get a real equity curve.

## Run

```bash
source .venv/bin/activate
set -a && source .env.live && set +a       # needs GOOGLE_API_KEY (prices are public, no Kraken key)
python -m paper_sim.run
```

Stop with Ctrl-C — state is saved and resumes next time.

No `tee` needed: each run writes its own log files automatically (see **Logs** below).

## Config (env vars, all optional)

| Var | Default | Meaning |
|-----|---------|---------|
| `DECISION_INTERVAL_MIN` | 40 | minutes between decisions (only fires when flat) |
| `MONITOR_INTERVAL_SEC` | 60 | how often to check SL/TP on an open position |
| `FEE_PCT_PER_SIDE` | 0.05 | taker fee per side (Kraken Futures ≈ 0.05%) |
| `PAPER_START_EQUITY` | 108 | starting paper equity (USD) |
| `PAPER_STATE_PATH` | `~/.tradingagents/paper_sim/state.json` | where state is stored |

Instrument, leverage, stop %, sizing all come from the same `.env.live` the live
bot uses (`SYMBOL`, `LEVERAGE`, `STOP_PCT`, `BALANCE_PCT`).

## Reading the results

Live log lines:
- `OPEN sell ... @ 73.79 SL 74.16 TP 73.42`
- `monitor open sell: 1m range [...] vs SL .. TP ..`
- `CLOSE sell via TP @ 73.42 | pnl +2.13 | equity 110.13 | W/L 1/0`
- `STATUS equity 110.13 | trades 1 (W 1/L 0) | pnl +2.13 | flat`

Full history (every closed trade with entry/exit/outcome/pnl) is in the state
JSON at `PAPER_STATE_PATH`.

## Logs (per-run, kept forever)

Every run writes its own files under `~/.tradingagents/paper_sim/logs/` — old
runs are never overwritten, so you always have the history to analyse:

- `PF_SOLUSD-<YYYYMMDD-HHMMSS>-<pid>.log` — the full human-readable log (same
  lines that print to the terminal).
- `PF_SOLUSD-<YYYYMMDD-HHMMSS>-<pid>.jsonl` — one JSON object per event
  (`run_start`, `decision`, `open`, `close`, `run_stop`) with a UTC timestamp,
  ready for `jq`/pandas analysis across runs.
- `latest.log` / `latest.jsonl` — symlinks to the current run, so the dashboards
  always find it.

Override the directory with `PAPER_LOG_DIR`. The dashboards default to the
current run; tail an older one with `python -m paper_sim.watch <path-to.log>`.
Analyse the structured stream, e.g.:

```bash
jq -c 'select(.event=="close") | {ts,side,outcome,pnl,equity}' \
  ~/.tradingagents/paper_sim/logs/latest.jsonl
```

## Notes

- **Cost**: each decision runs the full LLM analysis (~minutes, many Gemini
  calls). Every 40 min, 24/7 ≈ 36 analyses/day — watch your Gemini usage.
- **Timing**: the analysis blocks for a few minutes, but only when flat (nothing
  to monitor then), so SL/TP monitoring is never interrupted while a trade runs.
- **Conservative fills**: if a single 1m candle straddles both SL and TP, it's
  counted as **SL** (worst case) — paper results never flatter the strategy.
