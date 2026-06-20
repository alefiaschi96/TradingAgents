# live/ — Kraken Futures trading bot

Wires the TradingAgents analysis pipeline to a **Kraken Futures** account.
One cron tick = one decision cycle for **one instrument**.

## What it does (per run)

1. Connect to Kraken Futures (`ccxt.krakenfutures`).
2. **If a position is already open** → ensure it has SL/TP, then stop.
   Only ever **one position at a time**.
3. **If flat** → cancel any orphan SL/TP orders, then run the LLM analysis
   (`propagate(symbol, today, asset_type="crypto")`).
4. Map the 5-tier decision to a side:
   `Buy/Overweight → long`, `Sell/Underweight → short`, `Hold → nothing`.
5. Pass the pre-trade guards (leverage cap, equity floor).
6. Open the position with a **full-balance** entry, bracketed by a
   **stop-loss and take-profit at ±`STOP_PCT`%** of the fill price.

## Safety

- **`LIVE=false` by default** → orders are computed and logged but **never sent**.
  Run it for a few days, read the logs, then set `LIVE=true`.
- Isolated margin, leverage capped by `MAX_LEVERAGE`.
- Equity kill-switch (`EQUITY_FLOOR_USD`).
- Kraken API key must have **Futures Trading only, no withdrawal**.

> ⚠️ ±0.5% stops on a 5x perp is aggressive scalping: positions often close
> within minutes and fees (~0.1% round trip) eat into the 0.5% move. Watch the
> first runs.

## Run locally (dry-run)

```bash
pip install .                      # TradingAgents core
pip install -r requirements-live.txt
cp .env.live.example .env.live     # fill GOOGLE_API_KEY + Kraken keys
set -a && source .env.live && set +a
python -m live.runner
```

With `LIVE=false` you'll see `[DRY-RUN] ENTRY ...`, `[DRY-RUN] STOP-LOSS ...`,
`[DRY-RUN] TAKE-PROFIT ...` lines — the orders it *would* send.

## Deploy on Railway

1. New service from the `alefiaschi96/TradingAgents` repo, branch `ale`.
2. Build with `Dockerfile.live`.
3. Set the env vars from `.env.live.example` (keep `LIVE=false` to start).
4. **Settings → Cron Schedule**: `0 * * * *` (hourly). The container starts,
   runs one tick, exits.
5. To run more instruments, duplicate the service and change `SYMBOL`.

## Before going live (`LIVE=true`)

Validate against the installed `ccxt` version, since these are the only
order-shape assumptions in the code:

- `KrakenClient.create_stop_loss` / `create_take_profit` — the `stopLossPrice` /
  `takeProfitPrice` + `reduceOnly` params map to Kraken `stp` / `take_profit`.
- `KrakenClient.get_equity_usd` — reads the margin currency total from the
  multi-collateral (flex) wallet; confirm it reflects real available margin.
- `set_leverage` — confirm it sets **isolated** margin at the requested leverage.

## Files

| File | Role |
|------|------|
| `config.py` | env → `Config` dataclass |
| `kraken_client.py` | ccxt wrapper; writes gated by `LIVE` |
| `analysis.py` | TradingAgents config + `propagate` |
| `guards.py` | decision→side mapping + pre-trade checks |
| `bridge.py` | sizing + entry + SL/TP bracket |
| `runner.py` | cron entry point (`python -m live.runner`) |
