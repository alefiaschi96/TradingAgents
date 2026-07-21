"""Paper-trading engine: decide → open (simulated) → monitor → close.

Reuses live/ components (analysis, decision mapping, bracket math, Kraken price
client) WITHOUT modifying them. No order is ever sent — outcomes are simulated
by comparing the live 1-minute candle range against the stop-loss / take-profit.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from json import JSONDecodeError

from live.analysis import reasoning_from_state, run_analysis
from live.bridge import _bracket_prices
from live.config import Config
from live.guards import map_decision_to_side
from live.kraken_client import KrakenClient

logger = logging.getLogger("paper_sim")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_PRICE_TARGET_RE = re.compile(
    r"price\s*target\**\s*[:\-]\s*\**\s*[~≈]?\s*\$?\s*([0-9][\d,]*(?:\.\d+)?)",
    re.I,
)


def parse_price_target(pm_decision: str) -> float | None:
    """The PM report's explicit "**Price Target**: <n>" level, or None.

    Takes the last match so the structured field at the end of the report wins
    over incidental "price target" mentions inside the thesis text.
    """
    matches = _PRICE_TARGET_RE.findall(pm_decision or "")
    if not matches:
        return None
    try:
        value = float(matches[-1].replace(",", ""))
    except ValueError:
        return None
    return value if value > 0 else None


_EXEC_PLAN_RE = re.compile(
    r"\*\*Execution Plan\*\*:\s*```json\s*(\{.*?\})\s*```", re.I | re.S
)

# Structural-SL constants (see docs/structural-sl-plan.md).
_CONFIRM_TF = "15m"          # confirmation bar timeframe for close/hold
_CONFIRM_SEC = 900.0
_DEFAULT_CONFIRM_BARS = {"close": 1, "hold": 2}
_HARD_BUFFER_ATR = 0.4       # default hard = soft + this x ATR x confirm_bars
_HARD_ATR_RANGE = (0.2, 3.0)  # accepted hard distance, in ATRs from entry


def parse_execution_plan(pm_decision: str) -> dict | None:
    """The fenced-JSON stop contract rendered by ``render_pm_decision``, or None.

    This is a JSON round-trip of the PM's structured output — never a prose
    parse. Last match wins, mirroring ``parse_price_target``.
    """
    matches = _EXEC_PLAN_RE.findall(pm_decision or "")
    if not matches:
        return None
    try:
        plan = json.loads(matches[-1])
    except JSONDecodeError:
        return None
    return plan if isinstance(plan, dict) else None


def normalize_execution_plan(plan: dict) -> tuple[dict | None, str | None]:
    """Cascade steps 1 and 5: typed fields and confirm_bars range. Pure.

    Returns ``(normalized, None)`` or ``(None, reject_reason)``. Prices here
    are still in the PM's (spot) geometry — the caller translates them.
    """
    if not isinstance(plan, dict):
        return None, "block is not an object"
    try:
        soft = float(plan["invalidation_level"])
    except (KeyError, TypeError, ValueError):
        return None, "invalidation_level missing or not a number"
    if soft <= 0:
        return None, "invalidation_level not positive"
    semantics = plan.get("invalidation_semantics")
    if semantics not in ("touch", "close", "hold"):
        return None, f"invalidation_semantics {semantics!r} invalid"
    confirm = plan.get("confirm_bars")
    if confirm is None:
        confirm = _DEFAULT_CONFIRM_BARS.get(semantics, 1)
    else:
        try:
            confirm = int(confirm)
        except (TypeError, ValueError):
            return None, "confirm_bars not an integer"
        if not 1 <= confirm <= 4:
            return None, f"confirm_bars {confirm} outside [1, 4]"
    hard = plan.get("hard_level")
    if hard is not None:
        try:
            hard = float(hard)
        except (TypeError, ValueError):
            return None, "hard_level not a number"
        if hard <= 0:
            return None, "hard_level not positive"
    target = plan.get("price_target")
    if target is not None:
        try:
            target = float(target)
        except (TypeError, ValueError):
            return None, "price_target not a number"
        if target <= 0:
            target = None
    horizon = plan.get("horizon_minutes")
    if horizon is not None:
        try:
            horizon = float(horizon)
        except (TypeError, ValueError):
            return None, "horizon_minutes not a number"
        if horizon <= 0:
            horizon = None
    return {
        "soft": soft,
        "semantics": semantics,
        "confirm_bars": confirm,
        "hard": hard,
        "target": target,
        "horizon_minutes": horizon,
    }, None


def resolve_structural_levels(
    norm: dict, side: str, price: float, atr: float | None
) -> tuple[dict | None, str | None, bool]:
    """Cascade steps 2-4: geometry against the CURRENT execution price. Pure.

    All prices must already be in the same (perp) geometry. Returns
    ``(levels, reject_reason, vetoed)``: vetoed means the thesis is already
    invalidated at execution time — skip the trade, don't fall back.
    """
    soft, hard = norm["soft"], norm["hard"]
    semantics, confirm = norm["semantics"], norm["confirm_bars"]
    long = side == "buy"
    # Step 2: the invalidation level must sit on the adverse side of the
    # current price. Wrong side = price has already crossed it = born dead.
    if (long and soft >= price) or (not long and soft <= price):
        return None, None, True
    if semantics == "touch":
        hard = soft  # single stop at the declared level
    elif hard is not None:
        # Step 3: a declared hard must sit beyond the soft, adverse side.
        if (long and hard >= soft) or (not long and hard <= soft):
            return None, "hard_level not beyond invalidation_level", False
    else:
        if not atr or atr <= 0:
            return None, "no ATR available for the default hard buffer", False
        buffer = _HARD_BUFFER_ATR * atr * confirm
        hard = soft - buffer if long else soft + buffer
    # Step 4: hard distance from the execution price within the sane band.
    if not atr or atr <= 0:
        return None, "no ATR available to validate hard distance", False
    lo, hi = _HARD_ATR_RANGE
    dist = abs(price - hard)
    if not lo * atr <= dist <= hi * atr:
        return None, (
            f"hard distance {dist:.6g} outside [{lo:g}, {hi:g}]xATR ({atr:.6g})"
        ), False
    return {
        "soft": soft,
        "hard": hard,
        "semantics": semantics,
        "confirm_bars": confirm,
        "target": norm["target"],
        "horizon_minutes": norm["horizon_minutes"],
    }, None, False


def capped_tp(
    side: str, entry: float, tp_rr: float, analyst_target: float | None
) -> tuple[float, str]:
    """Effective take-profit: the PM's stated price target when it sits
    strictly between entry and the rr-based TP (profitable side, nearer than
    the mechanical level), else the rr TP unchanged. The rr level is a hard
    ceiling — the target can only pull the TP closer, never push it further.
    Returns (tp, source) with source "analyst_target" or "rr".
    """
    if analyst_target is not None and (
        (side == "buy" and entry < analyst_target < tp_rr)
        or (side == "sell" and tp_rr < analyst_target < entry)
    ):
        return analyst_target, "analyst_target"
    return tp_rr, "rr"


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
        analysis_min_gap_min: float = 30.0,
        event_sink=None,
        risk_pct_per_trade: float = 0.0,
        min_tp_cost_mult: float = 0.0,
        time_stop_hours: float = 0.0,
        regime_persist_ticks: int = 0,
        regime_exit_check: bool = False,
        min_analyst_rr: float = 0.0,
        structural_sl: bool = False,
    ):
        self.cfg = cfg
        self.state_path = state_path
        self.fee = fee_pct_per_side
        self.slippage = slippage_pct_per_side
        self.decision_interval = decision_interval_min * 60.0
        # Floor between two analyses when a regime *transition* fires early
        # (the periodic re-check still waits the full decision_interval).
        self.analysis_min_gap = analysis_min_gap_min * 60.0
        self.kraken = KrakenClient(cfg, "", "")  # no keys: public price/markets only
        self.state = self._load(start_equity)
        self._last_regime_log = None  # throttles the "regime ready/not ready" log line
        # Last regime observed while flat ("up"/"down"/"chop"/...). A tick whose
        # directional read differs from this baseline is a fresh transition —
        # the birth of a trend — and may trigger an early analysis.
        self._prev_regime: str | None = None
        # --- optional risk fixes, all off (0/False) by default so existing
        # sims keep byte-identical behaviour; enable per-instance via env.
        # Size positions off a fixed equity risk instead of full notional:
        # notional = equity*risk% / stop%, still capped by balance_pct*leverage.
        self.risk_pct_per_trade = risk_pct_per_trade
        # Skip entries whose expected TP distance is under N× the round-trip
        # cost (fees+slippage both sides) — no edge, pure churn.
        self.min_tp_cost_mult = min_tp_cost_mult
        # Close stagnant positions after N hours (neither SL nor TP hit).
        self.time_stop_hours = time_stop_hours
        # A regime transition must persist N consecutive ticks before it may
        # trigger an early analysis (debounce for chop↔trend flapping).
        self.regime_persist_ticks = regime_persist_ticks
        # While a position is open, a clean opposite regime closes it instead
        # of waiting for the stop.
        self.regime_exit_check = regime_exit_check
        # When the PM's own price target implies less than N x the stop
        # distance, skip the trade entirely (never trade a bad ratio: a
        # near target used to silently cap the TP and invert the win/loss
        # asymmetry the RR bracket is designed for).
        self.min_analyst_rr = min_analyst_rr
        # Structural SL: honour the PM's execution_plan stop contract (soft
        # invalidation confirmed on 15m closes + hard catastrophic stop on
        # touch) instead of the single ATR stop. Any invalid block falls back
        # to the ATR stop — never worse than today.
        self.structural_sl = structural_sl
        self._spot_exchange = None  # lazy public spot client (basis ratio)
        self._pending_regime: str | None = None
        self._pending_count = 0
        # Structured per-run event stream (JSONL). No-op when unset so the
        # simulator stays usable/testable without a run logger.
        self._emit_event = event_sink if event_sink is not None else (lambda *a, **k: None)

    def connect(self) -> None:
        # Force public-only mode so price/markets need no API keys.
        self.cfg.no_broker = True
        self.kraken.connect()

    # ------------------------------------------------------------- state
    @staticmethod
    def _new_state(start_equity: float) -> dict:
        return {
            "equity": start_equity,
            "open": None,
            "closed": [],
            "last_decision_at": None,
            "wins": 0,
            "losses": 0,
            "pnl_total": 0.0,
        }

    @staticmethod
    def _read_state_file(path: str, start_equity: float) -> dict:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        if not isinstance(state, dict):
            raise ValueError("state root must be a JSON object")

        defaults = PaperSimulator._new_state(start_equity)
        defaults.update(state)
        state = defaults

        if not isinstance(state["closed"], list):
            raise ValueError("state['closed'] must be a list")
        if state["open"] is not None and not isinstance(state["open"], dict):
            raise ValueError("state['open'] must be an object or null")

        state["equity"] = float(state["equity"])
        state["wins"] = int(state["wins"])
        state["losses"] = int(state["losses"])
        state["pnl_total"] = float(state["pnl_total"])
        return state

    def _load(self, start_equity: float) -> dict:
        if not os.path.exists(self.state_path):
            return self._new_state(start_equity)

        backup_path = f"{self.state_path}.bak"
        try:
            return self._read_state_file(self.state_path, start_equity)
        except (JSONDecodeError, OSError, TypeError, ValueError, KeyError) as exc:
            logger.error("Invalid paper-sim state %s: %s", self.state_path, exc)

        if os.path.exists(backup_path):
            try:
                recovered = self._read_state_file(backup_path, start_equity)
                shutil.copy2(backup_path, self.state_path)
                logger.warning(
                    "Recovered paper-sim state from backup %s", backup_path
                )
                return recovered
            except (JSONDecodeError, OSError, TypeError, ValueError, KeyError) as exc:
                logger.error("Invalid paper-sim backup %s: %s", backup_path, exc)

        raise RuntimeError(
            f"Paper-sim state is invalid and no valid backup exists: "
            f"{self.state_path}. Restore a known-good state file before restarting."
        )

    def save(self) -> None:
        state_dir = os.path.dirname(os.path.abspath(self.state_path))
        os.makedirs(state_dir, exist_ok=True)
        backup_path = f"{self.state_path}.bak"
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=state_dir,
                prefix=".paper-sim-state-",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name
                json.dump(self.state, tmp, indent=2, default=str)
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())

            if os.path.exists(self.state_path):
                try:
                    self._read_state_file(
                        self.state_path, float(self.state["equity"])
                    )
                except (JSONDecodeError, OSError, TypeError, ValueError, KeyError):
                    logger.warning(
                        "Existing state is invalid; not overwriting backup %s",
                        backup_path,
                    )
                else:
                    shutil.copy2(self.state_path, backup_path)

            os.replace(tmp_path, self.state_path)
            tmp_path = None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

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

    @staticmethod
    def _atr_from_ohlcv(ohlcv: list, period: int) -> float | None:
        """Average True Range over the last ``period`` bars of an OHLCV list."""
        if len(ohlcv) < period + 1:
            return None
        trs = []
        for i in range(1, len(ohlcv)):
            high, low, prev_close = float(ohlcv[i][2]), float(ohlcv[i][3]), float(ohlcv[i - 1][4])
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        trs = trs[-period:]
        return sum(trs) / len(trs) if trs else None

    @staticmethod
    def _ema_series(closes: list, period: int) -> list:
        """Exponential moving average series over ``closes``."""
        if not closes:
            return []
        k = 2.0 / (period + 1)
        ema = closes[0]
        out = [ema]
        for c in closes[1:]:
            ema = c * k + ema * (1.0 - k)
            out.append(ema)
        return out

    def _atr(self) -> float | None:
        """ATR over cfg.atr_period bars of cfg.atr_timeframe, or None on failure
        (caller falls back to the fixed-percent stop)."""
        period = self.cfg.atr_period
        try:
            ohlcv = self._fetch_ohlcv(self.cfg.atr_timeframe, period + 1)
        except Exception as e:  # noqa: BLE001 - all retries exhausted
            logger.warning("ATR: candle fetch failed (%s); falling back to fixed stop", e)
            return None
        return self._atr_from_ohlcv(ohlcv, period)

    def _regime_signals(self) -> dict | None:
        """Higher-timeframe trend read: EMA (now + a few bars back for slope),
        the latest price, and ATR — all from one regime-timeframe fetch.
        Returns None if data is unavailable (the pre-screen then fails open)."""
        period = self.cfg.regime_ema_period
        need = period + 5  # extra bars for the slope lookback and ATR
        try:
            ohlcv = self._fetch_ohlcv(self.cfg.regime_timeframe, need)
        except Exception as e:  # noqa: BLE001 - all retries exhausted
            logger.warning("regime: candle fetch failed (%s); gate will allow", e)
            return None
        if len(ohlcv) < period + 2:
            return None
        closes = [float(c[4]) for c in ohlcv]
        ema = self._ema_series(closes, period)
        return {
            "price": closes[-1],
            "ema": ema[-1],
            "ema_prev": ema[-4] if len(ema) >= 4 else ema[0],
            "atr": self._atr_from_ohlcv(ohlcv, min(period, len(ohlcv) - 1)),
        }

    def _regime_ready(self) -> tuple[bool, str, str]:
        """Cheap, direction-agnostic pre-screen: is there a clean tradeable
        regime *right now*? Used to decide whether it's worth spending an LLM
        analysis at all. No LLM, just candles. Fails open."""
        if not self.cfg.regime_filter:
            return True, "off", "filter off"
        sig = self._regime_signals()
        if sig is None:
            return True, "nodata", "no regime data (allow)"
        price, ema, ema_prev, atr = sig["price"], sig["ema"], sig["ema_prev"], sig["atr"]
        slope = ema - ema_prev
        if price > ema and slope > 0:
            regime = "up"
        elif price < ema and slope < 0:
            regime = "down"
        else:
            return False, "chop", "chop (no clear HTF trend)"
        if atr and atr > 0:
            stretch = abs(price - ema) / atr
            if stretch > self.cfg.regime_max_stretch_atr:
                return False, "stretched", f"overextended ({stretch:.1f} ATR from EMA)"
        return True, regime, f"{regime}-trend setup"

    def _analysis_gap_elapsed(self, now_ts: float) -> bool:
        """Rate-limit for transition-triggered analyses (anti flip-flop spam)."""
        last = self.state.get("last_decision_at")
        return last is None or (now_ts - last) >= self.analysis_min_gap

    def should_decide(self, now_ts: float) -> bool:
        """Trigger the (expensive) LLM only on a clean regime setup, either when
        the periodic cooldown has elapsed (re-check of a standing trend) or the
        moment the regime *transitions* into a directional state (chop→trend or
        trend flip). The transition path bypasses the cooldown so the analysis
        lands on the birth of the move instead of hours into it; it is
        rate-limited by analysis_min_gap so regime flip-flop can't spam LLM runs.
        """
        ready, regime, reason = self._regime_ready()
        prev = self._prev_regime
        if not ready:
            # chop/stretched/nodata: reset the baseline so the next clean
            # directional read counts as a fresh transition.
            self._log_regime(regime, reason, False)
            self._prev_regime = regime
            self._pending_regime, self._pending_count = None, 0
            return False
        if self.decision_due(now_ts):  # cooldown elapsed: periodic re-check
            self._log_regime(regime, reason, True)
            self._prev_regime = regime
            return True
        directional = regime in ("up", "down")
        if directional and regime != prev and self._analysis_gap_elapsed(now_ts):
            if self.regime_persist_ticks > 0:
                # Debounce: the same fresh regime must be read N ticks in a row
                # before it may spend an LLM run (chop↔trend flapping burned
                # multiple identical analyses per hour without it).
                if self._pending_regime == regime:
                    self._pending_count += 1
                else:
                    self._pending_regime, self._pending_count = regime, 1
                if self._pending_count < self.regime_persist_ticks:
                    return False
                self._pending_regime, self._pending_count = None, 0
            logger.info(
                "regime transition (%s -> %s) — early analysis, cooldown bypassed: %s",
                prev or "start", regime, reason,
            )
            self._prev_regime = regime
            return True
        # Ready but waiting (cooldown running, no fresh transition — or the
        # transition is inside the min gap: baseline stays put so it still
        # fires once the gap elapses).
        return False

    def _log_regime(self, regime: str, reason: str, ready: bool) -> None:
        """Log the regime read only when it changes, to avoid 60s spam."""
        key = (regime, ready)
        if key == self._last_regime_log:
            return
        self._last_regime_log = key
        if ready:
            logger.info("regime ready (%s) — running analysis: %s", regime, reason)
        else:
            logger.info("regime not ready — holding off (no LLM): %s", reason)

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
        rating, state = run_analysis(self.cfg)
        self.state["last_decision_at"] = now_ts
        # Persist the full agent reasoning (market/news/debate/PM) to the JSONL,
        # always — it is otherwise only printed to stdout and lost.
        self._emit_event("analysis", rating=rating, **reasoning_from_state(state))
        side = map_decision_to_side(rating, self.cfg)
        if side is None:
            logger.info("decision %s -> no entry (stay flat)", rating)
            self._emit_event("decision", rating=rating, side=None, action="flat")
            return
        self._emit_event("decision", rating=rating, side=side, action="open")
        final = (state or {}).get("final_trade_decision", "")
        analyst_target = parse_price_target(final)
        exec_plan = parse_execution_plan(final) if self.structural_sl else None
        self.open_position(
            side, rating, analyst_target=analyst_target, exec_plan=exec_plan
        )

    def _fill_price(self, price: float, fill_side: str) -> float:
        """Adverse-slippage fill for a market order: a buy fills higher, a sell
        fills lower. Always moves against us so paper results never flatter
        what live taker fills would actually give."""
        slip = self.slippage / 100.0
        return price * (1.0 + slip) if fill_side == "buy" else price * (1.0 - slip)

    def _spot_price(self) -> float | None:
        """Last SPOT price for the analysis symbol (basis-ratio anchor).

        The sim's Kraken client is perp-only, so this uses a tiny lazy public
        spot client. Returns None on any failure — the caller degrades to
        ratio 1 rather than rejecting an otherwise valid block.
        """
        try:
            if self._spot_exchange is None:
                import ccxt

                self._spot_exchange = ccxt.kraken({"enableRateLimit": True})
            pair = self.cfg.analysis_symbol.replace("-", "/")
            ticker = self._spot_exchange.fetch_ticker(pair)
            price = ticker.get("last") or ticker.get("close")
            return float(price) if price else None
        except Exception as exc:  # noqa: BLE001 - price feed, degrade gracefully
            logger.warning("structural SL: spot price fetch failed (%s)", exc)
            return None

    def _prepare_structural(
        self, exec_plan: dict | None, side: str, entry: float, ref_price: float
    ) -> tuple[dict | None, bool]:
        """Validate the PM's block and translate it into perp geometry.

        Returns ``(levels, vetoed)``. ``(None, False)`` = block rejected, fall
        back to the ATR stop; ``(None, True)`` = thesis already invalidated at
        execution — skip the trade entirely.
        """
        def _reject(reason: str) -> tuple[None, bool]:
            logger.warning(
                "invalidation block rejected: %s — falling back to ATR stop",
                reason,
            )
            self._emit_event("structural_rejected", reason=reason)
            return None, False

        if exec_plan is None:
            return _reject("execution_plan block missing")
        norm, err = normalize_execution_plan(exec_plan)
        if err:
            return _reject(err)
        # Basis: the PM's levels were born on SPOT data, execution happens on
        # the perp — freeze the ratio at open and translate every level.
        ratio = 1.0
        spot = self._spot_price()
        if spot and spot > 0:
            ratio = ref_price / spot
        else:
            logger.warning(
                "structural SL: spot unavailable, basis ratio defaults to 1"
            )
        for key in ("soft", "hard", "target"):
            if norm[key] is not None:
                norm[key] *= ratio
        # Contract rule: with touch semantics the declared level IS the stop;
        # a different hard_level is ignored, not honoured and not fatal.
        if (
            norm["semantics"] == "touch"
            and norm["hard"] is not None
            and norm["hard"] != norm["soft"]
        ):
            logger.warning(
                "structural SL: touch semantics — declared hard_level %.6g "
                "ignored (the invalidation level is the stop)", norm["hard"],
            )
            # Telemetry for the A/B verdict: every occurrence is a decision
            # where the PM de-facto disabled the soft/hard dual-track.
            self._emit_event(
                "structural_touch_divergent_hard",
                invalidation_level=norm["soft"],
                declared_hard=norm["hard"],
            )
            norm["hard"] = None
        levels, err, vetoed = resolve_structural_levels(
            norm, side, entry, self._atr()
        )
        if vetoed:
            return None, True
        if err:
            return _reject(err)
        levels.update(
            ratio=ratio,
            count=0,
            last_eval_bar_ts=None,
            soft_touched=False,
            opened_ts=time.time(),
        )
        return levels, False

    def open_position(
        self,
        side: str,
        rating: str,
        analyst_target: float | None = None,
        exec_plan: dict | None = None,
    ) -> None:
        ref_price = self.last_price()
        entry = self._fill_price(ref_price, side)  # market entry slips against us
        equity = self.state["equity"]
        structural = None
        if self.structural_sl:
            structural, vetoed = self._prepare_structural(
                exec_plan, side, entry, ref_price
            )
            if vetoed:
                logger.info(
                    "entry VETOED by invalidation gate: thesis already "
                    "invalidated at execution (price %.4f vs invalidation)",
                    entry,
                )
                self._emit_event("veto", gate="invalidation", entry=entry)
                return
            if structural is not None:
                # The block is the single source of truth: its (translated)
                # target replaces the prose-parsed one, even when null.
                analyst_target = structural.pop("target")
        # SL/TP are set relative to the real (slipped) entry, like live.
        # Stop width is fixed or ATR-scaled; TP is rr x the stop distance,
        # capped at the PM's stated price target when that is nearer.
        # Structural: every downstream gate and the sizing run on the HARD
        # distance — the worst case is what the size must price in.
        if structural is not None:
            stop_pct = abs(entry - structural["hard"]) / entry * 100.0
        else:
            stop_pct = self._effective_stop_pct(entry)
        rr = self.cfg.take_profit_rr
        cost_pct = 2.0 * (self.fee + self.slippage)  # full round trip, % of notional
        if self.min_tp_cost_mult > 0 and stop_pct * rr < self.min_tp_cost_mult * cost_pct:
            logger.info(
                "entry VETOED by cost gate: expected TP %.3f%% < %.1fx round-trip cost %.3f%%",
                stop_pct * rr, self.min_tp_cost_mult, cost_pct,
            )
            self._emit_event(
                "veto", gate="cost", tp_pct=stop_pct * rr, cost_pct=cost_pct
            )
            return
        if self.min_analyst_rr > 0 and analyst_target is not None and stop_pct > 0:
            tgt_dist = analyst_target - entry if side == "buy" else entry - analyst_target
            implied_rr = (tgt_dist / entry * 100.0) / stop_pct
            if implied_rr < self.min_analyst_rr:
                logger.info(
                    "entry VETOED by target gate: analysts' own price target %.4f "
                    "implies %.2fx the stop (floor %.1fx) — not enough room to run",
                    analyst_target, implied_rr, self.min_analyst_rr,
                )
                self._emit_event(
                    "veto", gate="target", analyst_target=analyst_target,
                    implied_rr=implied_rr, floor=self.min_analyst_rr,
                )
                return
        margin = equity * self.cfg.balance_pct
        notional = margin * self.cfg.leverage
        if self.risk_pct_per_trade > 0 and stop_pct > 0:
            # Fixed-fractional risk: losing this trade at the stop costs
            # ~risk% of equity, however wide the (ATR) stop is. The leveraged
            # notional above remains the hard ceiling.
            risk_notional = equity * self.risk_pct_per_trade / stop_pct
            notional = min(notional, risk_notional)
        size = self.kraken.amount_for_notional(notional, entry)
        sl, tp_rr, _close_side = _bracket_prices(side, entry, stop_pct, rr)
        tp, tp_source = capped_tp(side, entry, tp_rr, analyst_target)
        if tp_source == "analyst_target":
            logger.info(
                "TP capped to analyst price target %.4f (rr bracket wanted %.4f)",
                tp, tp_rr,
            )
        if structural is not None:
            sl = structural["hard"]  # exact declared level, no pct round-trip
        self.state["open"] = {
            "side": side,
            "rating": rating,
            "entry": entry,
            "size": size,
            "sl": sl,
            "tp": tp,
            "tp_source": tp_source,
            "stop_pct": stop_pct,
            "rr": rr,
            "notional": notional,
            "opened_at": _now_iso(),
        }
        if structural is not None:
            self.state["open"]["structural"] = structural
            logger.info(
                "structural SL armed: soft %.4f (%s, %d bar/s 15m) | hard %.4f "
                "(touch) | basis ratio %.6f",
                structural["soft"], structural["semantics"],
                structural["confirm_bars"], structural["hard"],
                structural["ratio"],
            )
        logger.info(
            "OPEN %s %s @ %.4f (ref %.4f, slip %.3f%%) size %.4f | SL %.4f TP %.4f "
            "(stop %.2f%% x rr %.1f, mode %s, tp %s) | equity %.2f",
            side, self.cfg.symbol, entry, ref_price, self.slippage, size, sl, tp,
            stop_pct, rr, self.cfg.stop_mode, tp_source, equity,
        )
        structural_fields = (
            {
                "sl_mode": "structural",
                "soft_level": structural["soft"],
                "hard_level": structural["hard"],
                "semantics": structural["semantics"],
                "confirm_bars": structural["confirm_bars"],
                "basis_ratio": structural["ratio"],
            }
            if structural is not None
            else {}
        )
        self._emit_event(
            "open", symbol=self.cfg.symbol, side=side, rating=rating, entry=entry,
            ref_price=ref_price, slip_pct=self.slippage, size=size, sl=sl, tp=tp,
            tp_source=tp_source, stop_pct=stop_pct, rr=rr, stop_mode=self.cfg.stop_mode,
            notional=notional, equity=equity, **structural_fields,
        )

    # ------------------------------------------------------------- monitor
    def monitor(self) -> None:
        pos = self.state["open"]
        high, low = self.minute_range()
        # Hard stop (structural: pos["sl"] IS the hard level) and TP, on the
        # 1m range. check_hit resolves same-candle straddles to SL first —
        # pessimistic, per the plan's precedence rule.
        reason, exit_price = check_hit(pos["side"], pos["sl"], pos["tp"], high, low)
        if reason:
            self.close_position(exit_price, reason)
            return
        st = pos.get("structural")
        if st and st["semantics"] in ("close", "hold"):
            # Wick beyond the soft with no exit: the old at-touch stop would
            # have closed here. Flag once; the trade's outcome tells us later
            # whether the save was worth it (wick-save metric).
            wicked = (
                low <= st["soft"] if pos["side"] == "buy" else high >= st["soft"]
            )
            if wicked and not st.get("soft_touched"):
                st["soft_touched"] = True
                logger.info(
                    "soft touched, not confirmed: 1m range crossed %.4f — "
                    "old stop would have exited here", st["soft"],
                )
                self._emit_event(
                    "soft_touched_not_confirmed",
                    soft_level=st["soft"], high=high, low=low,
                )
            if self._soft_confirmed(pos, st):
                logger.info(
                    "soft invalidation CONFIRMED: %d consecutive 15m close(s) "
                    "beyond %.4f", st["count"], st["soft"],
                )
                self.close_position(self.last_price(), "SOFT")
                return
        # Time stop: a declared horizon overrides the profile default — the
        # thesis was given ~2x its own clock to play out.
        time_limit_h = self.time_stop_hours
        if st and st.get("horizon_minutes"):
            time_limit_h = st["horizon_minutes"] * 2.0 / 60.0
        if time_limit_h > 0 and self._position_age_hours(pos) >= time_limit_h:
            logger.info(
                "time stop: position open for %.1fh (limit %.1fh) with neither SL nor TP hit",
                self._position_age_hours(pos), time_limit_h,
            )
            self.close_position(self.last_price(), "TIME")
            return
        if self.regime_exit_check:
            ready, regime, reason_txt = self._regime_ready()
            against = (regime == "down" and pos["side"] == "buy") or (
                regime == "up" and pos["side"] == "sell"
            )
            if ready and against:
                logger.info(
                    "regime flipped against open %s (%s) — closing early instead of "
                    "riding to the stop: %s", pos["side"], regime, reason_txt,
                )
                self.close_position(self.last_price(), "REGIME")
                return
        logger.info(
            "monitor open %s: 1m range [%.4f, %.4f] vs SL %.4f TP %.4f",
            pos["side"], low, high, pos["sl"], pos["tp"],
        )

    def _soft_confirmed(self, pos: dict, st: dict) -> bool:
        """Advance the soft-invalidation count over every FINAL 15m bar not yet
        evaluated; True when confirm_bars consecutive closes sit beyond the soft.

        Catch-up by design: bars are keyed off ``last_eval_bar_ts``, so a
        daemon restart or a run of failed fetches replays every bar closed in
        the gap instead of silently skipping it. Rules (see the plan):
        - only bars that OPEN after the position opened (the straddling bar
          was half-written, possibly on our own entry wick);
        - only final bars (open <= now - 15m), never the forming candle;
        - hold at N = N *consecutive*; a close back inside resets the count.
        """
        now = time.time()
        opened_ts = float(st.get("opened_ts") or 0.0)
        last_eval = float(st.get("last_eval_bar_ts") or 0.0)
        span_start = max(opened_ts, last_eval)
        limit = max(4, min(500, int((now - span_start) // _CONFIRM_SEC) + 3))
        try:
            ohlcv = self._fetch_ohlcv(_CONFIRM_TF, limit)
        except Exception as e:  # noqa: BLE001 - all retries exhausted
            logger.warning(
                "soft check: 15m fetch failed (%s); catch-up resumes next tick", e
            )
            return False
        long = pos["side"] == "buy"
        soft = st["soft"]
        for candle in ohlcv:
            ts = float(candle[0]) / 1000.0
            if ts <= opened_ts or ts <= last_eval:
                continue
            if ts + _CONFIRM_SEC > now:
                break  # bar still forming — finality only
            close = float(candle[4])
            beyond = close < soft if long else close > soft
            st["count"] = (st.get("count") or 0) + 1 if beyond else 0
            st["last_eval_bar_ts"] = ts
            if st["count"] >= st["confirm_bars"]:
                return True
        return False

    @staticmethod
    def _position_age_hours(pos: dict) -> float:
        try:
            opened = datetime.fromisoformat(pos["opened_at"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - opened).total_seconds() / 3600.0

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
        # Cooldown counts from the exit too, so we don't re-fire the LLM the very
        # next tick after a stop-out.
        self.state["last_decision_at"] = time.time()
        logger.info(
            "CLOSE %s via %s @ %.4f (trigger %.4f) | pnl %+.2f (gross %+.2f, fees %.2f) | equity %.2f | W/L %d/%d",
            side, reason, exit_price, trigger_price, net, gross, fees, self.state["equity"],
            self.state["wins"], self.state["losses"],
        )
        structural_fields = {}
        st = pos.get("structural")
        if st:
            structural_fields = {
                "sl_mode": "structural",
                "soft_level": st.get("soft"),
                "hard_level": st.get("hard"),
                "semantics": st.get("semantics"),
                "confirm_count": st.get("count"),
                "soft_touched": bool(st.get("soft_touched")),
                "basis_ratio": st.get("ratio"),
            }
            if reason == "SOFT":
                # Insurance premium of waiting for the 15m close: systematic
                # slippage past the declared level. Without this column the
                # wick-save tally lies (plan §5, metric 2).
                structural_fields["soft_delay"] = exit_price - st["soft"]
            elif reason == "SL":
                structural_fields["hard_delay"] = exit_price - st["hard"]
                # Full loser: straight to the hard stop without a single soft
                # confirmation — where the dual-track loses more than today.
                structural_fields["hard_hit_direct"] = not st.get("count")
        self._emit_event(
            "close", symbol=self.cfg.symbol, side=side, rating=pos.get("rating"),
            outcome=reason, exit=exit_price, trigger=trigger_price, pnl=net,
            gross=gross, fees=fees, equity=self.state["equity"],
            wins=self.state["wins"], losses=self.state["losses"],
            **structural_fields,
        )

    def summary(self) -> str:
        s = self.state
        pos = "flat" if not s["open"] else f"OPEN {s['open']['side']} @ {s['open']['entry']:.4f}"
        return (
            f"equity {s['equity']:.2f} | trades {len(s['closed'])} "
            f"(W {s['wins']}/L {s['losses']}) | pnl {s['pnl_total']:+.2f} | {pos}"
        )
