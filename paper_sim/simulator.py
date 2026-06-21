"""Paper-trading engine: decide → open (simulated) → monitor → close.

Reuses live/ components (analysis, decision mapping, bracket math, Kraken price
client) WITHOUT modifying them. No order is ever sent — outcomes are simulated
by comparing the live 1-minute candle range against the stop-loss / take-profit.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

from live.analysis import run_analysis
from live.bridge import _bracket_prices
from live.config import Config
from live.guards import map_decision_to_side
from live.kraken_client import KrakenClient

logger = logging.getLogger("paper_sim")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_hit(side: str, sl: float, tp: float, high: float, low: float):
    """Did this candle's range touch SL or TP? Pure, testable.

    Returns (reason, exit_price) or (None, None). If BOTH SL and TP are inside
    the same candle (a gap straddle), we resolve conservatively to SL — the
    pessimistic outcome — so paper results never flatter the strategy.
    """
    if side == "sell":  # short: SL above entry, TP below
        sl_hit, tp_hit = high >= sl, low <= tp
    else:               # long: SL below entry, TP above
        sl_hit, tp_hit = low <= sl, high >= tp
    if sl_hit:
        return "SL", sl
    if tp_hit:
        return "TP", tp
    return None, None


class PaperSimulator:
    def __init__(
        self,
        cfg: Config,
        *,
        state_path: str,
        start_equity: float,
        fee_pct_per_side: float,
        decision_interval_min: float,
        slippage_pct_per_side: float = 0.0,
    ):
        self.cfg = cfg
        self.state_path = state_path
        self.fee = fee_pct_per_side
        self.slippage = slippage_pct_per_side
        self.decision_interval = decision_interval_min * 60.0
        self.kraken = KrakenClient(cfg, "", "")  # no keys: public price/markets only
        self.state = self._load(start_equity)

    def connect(self) -> None:
        # Force public-only mode so price/markets need no API keys.
        self.cfg.no_broker = True
        self.kraken.connect()

    # ------------------------------------------------------------- state
    def _load(self, start_equity: float) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {
            "equity": start_equity,
            "open": None,
            "closed": [],
            "last_decision_at": None,
            "wins": 0,
            "losses": 0,
            "pnl_total": 0.0,
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def has_open(self) -> bool:
        return self.state.get("open") is not None

    def decision_due(self, now_ts: float) -> bool:
        last = self.state.get("last_decision_at")
        return last is None or (now_ts - last) >= self.decision_interval

    # ------------------------------------------------------------- prices
    def last_price(self) -> float:
        return self.kraken.get_last_price()

    def _fetch_ohlcv(self, timeframe: str, limit: int) -> list:
        """Fetch candles with retry — Kraken's charts endpoint drops often.

        Raises the last error if all attempts fail; callers decide the fallback.
        """
        last_err = None
        for attempt in range(3):
            try:
                return self.kraken.exchange.fetch_ohlcv(
                    self.kraken.ccxt_symbol, timeframe, limit=limit
                )
            except Exception as e:  # noqa: BLE001 - transient charts-endpoint blips
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise last_err

    def minute_range(self) -> tuple[float, float]:
        """(high, low) of the latest 1m candle — catches intra-minute wicks.

        On a persistent fetch failure, fall back to the last price (a different,
        more reliable endpoint) as a point check rather than skipping the tick.
        """
        try:
            ohlcv = self._fetch_ohlcv("1m", 2)
            last = ohlcv[-1]
            return float(last[2]), float(last[3])
        except Exception as e:  # noqa: BLE001 - all retries exhausted
            logger.warning(
                "minute_range: 1m candle fetch failed (%s); "
                "falling back to last price for this tick", e,
            )
            price = self.last_price()
            return price, price

    def _atr(self) -> float | None:
        """Average True Range over cfg.atr_period bars of cfg.atr_timeframe.

        Returns None if candles can't be fetched or there aren't enough, so the
        caller can fall back to the fixed-percent stop.
        """
        period = self.cfg.atr_period
        try:
            ohlcv = self._fetch_ohlcv(self.cfg.atr_timeframe, period + 1)
        except Exception as e:  # noqa: BLE001 - all retries exhausted
            logger.warning("ATR: candle fetch failed (%s); falling back to fixed stop", e)
            return None
        if len(ohlcv) < period + 1:
            return None
        trs = []
        for i in range(1, len(ohlcv)):
            high, low, prev_close = float(ohlcv[i][2]), float(ohlcv[i][3]), float(ohlcv[i - 1][4])
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        trs = trs[-period:]
        return sum(trs) / len(trs) if trs else None

    def _effective_stop_pct(self, entry: float) -> float:
        """Stop distance as a percent of entry: fixed, or scaled to ATR."""
        if self.cfg.stop_mode.lower() != "atr" or entry <= 0:
            return self.cfg.stop_pct
        atr = self._atr()
        if not atr:
            return self.cfg.stop_pct
        return (self.cfg.stop_atr_mult * atr / entry) * 100.0

    # ------------------------------------------------------------- decide
    def maybe_decide(self, now_ts: float) -> None:
        """Run the analysis; open a simulated position if it says long/short."""
        rating, _state = run_analysis(self.cfg)
        self.state["last_decision_at"] = now_ts
        side = map_decision_to_side(rating, self.cfg)
        if side is None:
            logger.info("decision %s -> no entry (stay flat)", rating)
            return
        self.open_position(side, rating)

    def _fill_price(self, price: float, fill_side: str) -> float:
        """Adverse-slippage fill for a market order: a buy fills higher, a sell
        fills lower. Always moves against us so paper results never flatter
        what live taker fills would actually give."""
        slip = self.slippage / 100.0
        return price * (1.0 + slip) if fill_side == "buy" else price * (1.0 - slip)

    def open_position(self, side: str, rating: str) -> None:
        ref_price = self.last_price()
        entry = self._fill_price(ref_price, side)  # market entry slips against us
        equity = self.state["equity"]
        margin = equity * self.cfg.balance_pct
        notional = margin * self.cfg.leverage
        size = self.kraken.amount_for_notional(notional, entry)
        # SL/TP are set relative to the real (slipped) entry, like live.
        # Stop width is fixed or ATR-scaled; TP is rr x the stop distance.
        stop_pct = self._effective_stop_pct(entry)
        rr = self.cfg.take_profit_rr
        sl, tp, _close_side = _bracket_prices(side, entry, stop_pct, rr)
        self.state["open"] = {
            "side": side,
            "rating": rating,
            "entry": entry,
            "size": size,
            "sl": sl,
            "tp": tp,
            "stop_pct": stop_pct,
            "rr": rr,
            "notional": notional,
            "opened_at": _now_iso(),
        }
        logger.info(
            "OPEN %s %s @ %.4f (ref %.4f, slip %.3f%%) size %.4f | SL %.4f TP %.4f "
            "(stop %.2f%% x rr %.1f, mode %s) | equity %.2f",
            side, self.cfg.symbol, entry, ref_price, self.slippage, size, sl, tp,
            stop_pct, rr, self.cfg.stop_mode, equity,
        )

    # ------------------------------------------------------------- monitor
    def monitor(self) -> None:
        pos = self.state["open"]
        high, low = self.minute_range()
        reason, exit_price = check_hit(pos["side"], pos["sl"], pos["tp"], high, low)
        if reason:
            self.close_position(exit_price, reason)
        else:
            logger.info(
                "monitor open %s: 1m range [%.4f, %.4f] vs SL %.4f TP %.4f",
                pos["side"], low, high, pos["sl"], pos["tp"],
            )

    def close_position(self, trigger_price: float, reason: str) -> None:
        pos = self.state["open"]
        side, entry, size = pos["side"], pos["entry"], pos["size"]
        # SL/TP are market-on-trigger orders, so the actual fill slips past the
        # trigger, against us (a short buys back higher, a long sells lower).
        close_side = "buy" if side == "sell" else "sell"
        exit_price = self._fill_price(trigger_price, close_side)
        gross = (entry - exit_price) * size if side == "sell" else (exit_price - entry) * size
        fees = (entry * size + exit_price * size) * (self.fee / 100.0)
        net = gross - fees
        self.state["equity"] += net
        self.state["pnl_total"] += net
        self.state["wins" if net >= 0 else "losses"] += 1
        self.state["closed"].append({
            **pos,
            "exit": exit_price,
            "exit_trigger": trigger_price,
            "outcome": reason,
            "gross": gross,
            "fees": fees,
            "pnl": net,
            "closed_at": _now_iso(),
            "equity_after": self.state["equity"],
        })
        self.state["open"] = None
        logger.info(
            "CLOSE %s via %s @ %.4f (trigger %.4f) | pnl %+.2f (gross %+.2f, fees %.2f) | equity %.2f | W/L %d/%d",
            side, reason, exit_price, trigger_price, net, gross, fees, self.state["equity"],
            self.state["wins"], self.state["losses"],
        )

    def summary(self) -> str:
        s = self.state
        pos = "flat" if not s["open"] else f"OPEN {s['open']['side']} @ {s['open']['entry']:.4f}"
        return (
            f"equity {s['equity']:.2f} | trades {len(s['closed'])} "
            f"(W {s['wins']}/L {s['losses']}) | pnl {s['pnl_total']:+.2f} | {pos}"
        )
