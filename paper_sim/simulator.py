"""Paper-trading engine: decide → (pend) → open (simulated) → monitor → close.

Reuses live/ components (analysis, decision mapping, bracket math, Kraken price
client) WITHOUT modifying them. No order is ever sent — outcomes are simulated
by comparing the live 1-minute candle range against the stop-loss / take-profit.

When ENTRY_MODE=plan (the default) and the Portfolio Manager's decision carries
an ``**Entry Plan**`` block, the simulator does not enter at market: it parks
the legs as pending conditional orders (OCO — the first leg whose condition
triggers becomes the position, the other is cancelled) and watches the 1m tape
until one fills, the plan expires, or the market flushes through it.
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


_ENTRY_PLAN_HEADER_RE = re.compile(r"\*\*Entry Plan\*\*\s*:", re.I)
_ENTRY_NUM = r"([0-9][\d,]*(?:\.\d+)?)"
_ENTRY_FIELD_RES = {
    "zone": re.compile(rf"zone\s+{_ENTRY_NUM}\s*-\s*{_ENTRY_NUM}", re.I),
    "trigger": re.compile(rf"trigger\s+{_ENTRY_NUM}", re.I),
    "target": re.compile(rf"target\s+{_ENTRY_NUM}", re.I),
    "invalidation": re.compile(rf"invalidation\s+{_ENTRY_NUM}", re.I),
}


def _entry_num(match_groups) -> list[float]:
    return [float(g.replace(",", "")) for g in match_groups]


def parse_entry_plan(pm_decision: str) -> list[dict]:
    """The PM's ``**Entry Plan**`` block as a list of leg dicts, or [].

    Reads the deterministic lines ``render_entry_leg`` produces (this is our
    own rendered format, not LLM prose):

        **Entry Plan**:
        - pullback | zone 1837.0-1840.0 | target 1849.5 | invalidation 1832.0
        - breakout | trigger 1850.0 | target 1858.5

    Each leg dict has kind, zone_low, zone_high, trigger, target, invalidation
    (None where absent). Malformed legs are skipped, never raised on.
    """
    m = _ENTRY_PLAN_HEADER_RE.search(pm_decision or "")
    if not m:
        return []
    legs: list[dict] = []
    for line in pm_decision[m.end():].splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("-"):
            break  # end of the block (next section header or prose)
        body = line.lstrip("- ").strip()
        kind = body.split("|", 1)[0].strip().lower()
        if kind not in ("pullback", "breakout"):
            continue
        leg = {"kind": kind, "zone_low": None, "zone_high": None,
               "trigger": None, "target": None, "invalidation": None}
        zm = _ENTRY_FIELD_RES["zone"].search(body)
        if zm:
            lo, hi = sorted(_entry_num(zm.groups()))
            leg["zone_low"], leg["zone_high"] = lo, hi
        for field in ("trigger", "target", "invalidation"):
            fm = _ENTRY_FIELD_RES[field].search(body)
            if fm:
                leg[field] = _entry_num(fm.groups())[0]
        if kind == "pullback" and leg["zone_low"] is None:
            continue  # a pullback leg without a zone is unusable
        if kind == "breakout" and leg["trigger"] is None:
            continue  # a breakout leg without a trigger is unusable
        legs.append(leg)
    return legs


def validate_legs(side: str, legs: list[dict], ref_price: float) -> tuple[list[dict], list[str]]:
    """Keep only legs whose levels make geometric sense for this side.

    For a buy: the pullback zone must sit below the current price and the
    breakout trigger above it; invalidation below the entry, target above.
    (Mirrored for a sell.) At most one leg per kind — first valid one wins.
    Returns (valid_legs, reasons_for_dropped) and never raises: a nonsense
    plan just degrades to fewer (or zero) legs.
    """
    sign = 1.0 if side == "buy" else -1.0  # +1: profits up; -1: profits down

    def below(a, b):  # "a is on the losing side of b"
        return sign * (a - b) < 0

    valid: list[dict] = []
    reasons: list[str] = []
    seen: set[str] = set()
    for leg in legs:
        kind = leg["kind"]
        if kind in seen:
            reasons.append(f"duplicate {kind} leg dropped")
            continue
        if kind == "pullback":
            edge = leg["zone_high"] if side == "buy" else leg["zone_low"]
            far = leg["zone_low"] if side == "buy" else leg["zone_high"]
            if not below(edge, ref_price):
                reasons.append(
                    f"pullback zone {leg['zone_low']}-{leg['zone_high']} is not "
                    f"on the retracement side of price {ref_price:.4f}")
                continue
            anchor = far
        else:  # breakout
            if not below(ref_price, leg["trigger"]):
                reasons.append(
                    f"breakout trigger {leg['trigger']} is not beyond "
                    f"price {ref_price:.4f}")
                continue
            anchor = leg["trigger"]
        if leg["invalidation"] is not None and not below(leg["invalidation"], anchor):
            reasons.append(
                f"{kind} invalidation {leg['invalidation']} is not on the stop "
                f"side of {anchor}; ignored")
            leg = {**leg, "invalidation": None}
        entry_ref = anchor if kind == "breakout" else (
            leg["zone_high"] if side == "buy" else leg["zone_low"])
        if leg["target"] is not None and not below(entry_ref, leg["target"]):
            reasons.append(
                f"{kind} target {leg['target']} is not on the profit side of "
                f"entry {entry_ref}; ignored")
            leg = {**leg, "target": None}
        seen.add(kind)
        valid.append(leg)
    return valid, reasons


def check_entry(side: str, legs: list[dict], o: float, h: float, low: float, c: float):
    """Which pending leg (if any) does this 1m candle fill? Pure, testable.

    A pullback leg fills at the near edge of its zone (a resting limit order:
    no adverse slippage); a breakout leg fills at its trigger (a stop-market:
    the caller applies slippage). When ONE candle triggers both legs, its
    direction decides which side of the range traded first: a rising candle
    (close >= open) is assumed to have gone open→low→high, so the retracement
    leg filled first; a falling candle the reverse. Whether the same candle
    then also flushed to the stop is the caller's job to check (it reuses
    ``check_hit`` on the same candle after placing the bracket).

    Returns (leg, fill_price) or (None, None).
    """
    pull = next((leg for leg in legs if leg["kind"] == "pullback"), None)
    brk = next((leg for leg in legs if leg["kind"] == "breakout"), None)
    if side == "buy":
        pull_hit = pull is not None and low <= pull["zone_high"]
        brk_hit = brk is not None and h >= brk["trigger"]
        pull_fill = pull["zone_high"] if pull else None
        retrace_first = c >= o  # rising candle: open→low→high
    else:
        pull_hit = pull is not None and h >= pull["zone_low"]
        brk_hit = brk is not None and low <= brk["trigger"]
        pull_fill = pull["zone_low"] if pull else None
        retrace_first = c <= o  # falling candle: open→high→low
    if pull_hit and brk_hit:
        if retrace_first:
            return pull, pull_fill
        return brk, brk["trigger"]
    if pull_hit:
        return pull, pull_fill
    if brk_hit:
        return brk, brk["trigger"]
    return None, None


def profit_lock_level(side: str, tp: float, pct: float) -> float:
    """The profit-lock floor: the price ``pct`` percent short of the TP."""
    frac = pct / 100.0
    return tp * (1.0 - frac) if side == "buy" else tp * (1.0 + frac)


def maybe_arm_profit_lock(
    side: str, sl: float, tp: float, pct: float,
    high: float, low: float, already: bool = False,
):
    """Floor to lock in when this candle first reaches the near-TP band, else None.

    Once price trades within ``pct`` percent of the TP, that band edge becomes
    a protective floor (the position's new SL): a later return to it closes
    the trade there, in profit, instead of riding the reversal all the way
    back to the original stop. The TP itself stays active above/below, so a
    clean push through still exits at the full target. Pure, testable;
    ``pct`` <= 0 disables the feature, an armed position never re-arms.
    """
    if already or pct <= 0:
        return None
    level = profit_lock_level(side, tp, pct)
    if side == "buy":
        return level if high >= level and level > sl else None
    return level if low <= level and level < sl else None


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
            "pending": None,
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
        if state["pending"] is not None and not isinstance(state["pending"], dict):
            raise ValueError("state['pending'] must be an object or null")

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

    def has_pending(self) -> bool:
        return self.state.get("pending") is not None

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

    def minute_candle(self) -> tuple[float, float, float, float]:
        """(open, high, low, close) of the latest 1m candle.

        On a persistent fetch failure, fall back to the last price (a different,
        more reliable endpoint) as a point check rather than skipping the tick.
        """
        try:
            ohlcv = self._fetch_ohlcv("1m", 2)
            last = ohlcv[-1]
            return (float(last[1]), float(last[2]), float(last[3]), float(last[4]))
        except Exception as e:  # noqa: BLE001 - all retries exhausted
            logger.warning(
                "minute_candle: 1m candle fetch failed (%s); "
                "falling back to last price for this tick", e,
            )
            price = self.last_price()
            return price, price, price, price

    def minute_range(self) -> tuple[float, float]:
        """(high, low) of the latest 1m candle — catches intra-minute wicks."""
        _o, high, low, _c = self.minute_candle()
        return high, low

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
            return False
        if self.decision_due(now_ts):  # cooldown elapsed: periodic re-check
            self._log_regime(regime, reason, True)
            self._prev_regime = regime
            return True
        directional = regime in ("up", "down")
        if directional and regime != prev and self._analysis_gap_elapsed(now_ts):
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
        """Run the analysis; act on a long/short call.

        With ENTRY_MODE=plan and a usable ``**Entry Plan**`` in the PM
        decision, the trade is parked as pending conditional legs instead of
        being opened at market — the executor then *waits for the PM's own
        entry conditions* rather than chasing the current price.
        """
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
        pm_text = (state or {}).get("final_trade_decision", "")
        analyst_target = parse_price_target(pm_text)
        if self.cfg.entry_mode.lower() == "plan":
            legs = parse_entry_plan(pm_text)
            if legs:
                ref_price = self.last_price()
                legs, dropped = validate_legs(side, legs, ref_price)
                for reason in dropped:
                    logger.info("entry leg dropped: %s", reason)
                if legs:
                    self._emit_event(
                        "decision", rating=rating, side=side, action="pending")
                    self._create_pending(
                        side, rating, legs, analyst_target, ref_price, now_ts)
                    return
                logger.info("entry plan unusable after validation -> market entry")
            else:
                logger.info("no entry plan in PM decision -> market entry")
        self._emit_event("decision", rating=rating, side=side, action="open")
        self.open_position(side, rating, analyst_target=analyst_target)

    # ------------------------------------------------------------- pending
    @staticmethod
    def _leg_desc(side: str, leg: dict) -> str:
        """Compact human description of one leg for logs, e.g.
        ``pullback 1837.00-1840.00 (target 1849.50, inval 1832.00)``."""
        if leg["kind"] == "pullback":
            head = f"pullback {leg['zone_low']:.2f}-{leg['zone_high']:.2f}"
        else:
            arrow = ">=" if side == "buy" else "<="
            head = f"breakout {arrow}{leg['trigger']:.2f}"
        extras = []
        if leg["target"] is not None:
            extras.append(f"target {leg['target']:.2f}")
        if leg["invalidation"] is not None:
            extras.append(f"inval {leg['invalidation']:.2f}")
        return head + (f" ({', '.join(extras)})" if extras else "")

    def _create_pending(
        self, side: str, rating: str, legs: list[dict],
        analyst_target: float | None, ref_price: float, now_ts: float,
    ) -> None:
        ttl_min = self.cfg.entry_ttl_min
        self.state["pending"] = {
            "side": side,
            "rating": rating,
            "legs": legs,
            "analyst_target": analyst_target,
            "ref_price": ref_price,
            "created_at": _now_iso(),
            "created_ts": now_ts,
            "expires_ts": now_ts + ttl_min * 60.0,
        }
        desc = " OR ".join(self._leg_desc(side, leg) for leg in legs)
        logger.info(
            "PENDING %s %s: %s | valid %.0fmin | ref %.4f",
            side, self.cfg.symbol, desc, ttl_min, ref_price,
        )
        self._emit_event(
            "pending", symbol=self.cfg.symbol, side=side, rating=rating,
            legs=legs, analyst_target=analyst_target, ref_price=ref_price,
            ttl_min=ttl_min,
        )

    def check_pending(self, now_ts: float) -> None:
        """One watch tick for the pending conditional legs: expire, fill, or wait.

        A fill immediately re-checks the SAME candle against the new bracket
        (via ``check_hit``): if the minute that filled us also flushed through
        the stop, the position opens and closes in one tick — the pessimistic
        reading of an ambiguous candle, consistent with ``check_hit``'s own
        SL-first rule.
        """
        p = self.state["pending"]
        if now_ts >= p["expires_ts"]:
            age_min = (now_ts - p["created_ts"]) / 60.0
            logger.info(
                "PENDING EXPIRED after %.0fmin without a fill (was: %s)",
                age_min,
                " OR ".join(self._leg_desc(p["side"], leg) for leg in p["legs"]),
            )
            self._emit_event(
                "pending_expired", symbol=self.cfg.symbol, side=p["side"],
                rating=p["rating"], legs=p["legs"], age_min=age_min,
            )
            self.state["pending"] = None
            return
        o, high, low, close = self.minute_candle()
        leg, fill_price = check_entry(p["side"], p["legs"], o, high, low, close)
        if leg is None:
            logger.info(
                "pending watch %s: 1m range [%.4f, %.4f] vs %s",
                p["side"], low, high,
                " / ".join(self._leg_desc(p["side"], x) for x in p["legs"]),
            )
            return
        logger.info(
            "PENDING FILLED (%s) %s @ %.4f — other leg cancelled",
            leg["kind"], p["side"], fill_price,
        )
        self._emit_event(
            "pending_fill", symbol=self.cfg.symbol, side=p["side"],
            rating=p["rating"], leg=leg, fill_price=fill_price,
        )
        self.state["pending"] = None
        self.open_position(
            p["side"], p["rating"], analyst_target=p["analyst_target"],
            entry_kind=leg["kind"], leg=leg, trigger_price=fill_price,
        )
        pos = self.state["open"]
        reason, exit_price = check_hit(pos["side"], pos["sl"], pos["tp"], high, low)
        if reason:
            logger.info(
                "same-candle flush: the minute that filled the entry also ran "
                "through %s — closing immediately", reason,
            )
            self.close_position(exit_price, reason)

    def _fill_price(self, price: float, fill_side: str) -> float:
        """Adverse-slippage fill for a market order: a buy fills higher, a sell
        fills lower. Always moves against us so paper results never flatter
        what live taker fills would actually give."""
        slip = self.slippage / 100.0
        return price * (1.0 + slip) if fill_side == "buy" else price * (1.0 - slip)

    def _sl_with_invalidation(
        self, side: str, entry: float, sl_atr: float, leg: dict | None
    ) -> tuple[float, float, str]:
        """Effective SL when a plan leg carries an invalidation level.

        The PM's invalidation is the *structural* stop area; when it sits
        beyond the ATR stop we widen the SL to it — capped at
        ``entry_max_stop_widen`` × the ATR distance — and hand back a size
        scale (<1) so the dollar risk of the widened stop stays what the ATR
        stop would have risked. The SL is never tightened below the ATR stop:
        stops clipped inside normal volatility were the observed failure mode.
        Returns (sl, size_scale, sl_source).
        """
        inval = (leg or {}).get("invalidation")
        atr_dist = abs(entry - sl_atr)
        if inval is None or atr_dist <= 0:
            return sl_atr, 1.0, "atr"
        inv_dist = (entry - inval) if side == "buy" else (inval - entry)
        if inv_dist <= atr_dist:
            return sl_atr, 1.0, "atr"
        max_dist = atr_dist * max(1.0, self.cfg.entry_max_stop_widen)
        source = "invalidation"
        if inv_dist > max_dist:
            inv_dist, source = max_dist, "invalidation_capped"
        sl = entry - inv_dist if side == "buy" else entry + inv_dist
        return sl, atr_dist / inv_dist, source

    def open_position(
        self, side: str, rating: str, analyst_target: float | None = None,
        *, entry_kind: str = "market", leg: dict | None = None,
        trigger_price: float | None = None,
    ) -> None:
        """Open the simulated position.

        ``entry_kind`` records how we got in: "market" (immediate, adverse
        slippage), "pullback" (resting limit — fills at its own price, no
        slippage) or "breakout" (stop-market on the trigger — slips like a
        market order). ``leg`` carries the filled plan leg, whose target caps
        the TP and whose invalidation may widen the SL (size scaled down to
        hold dollar risk constant).
        """
        ref_price = trigger_price if trigger_price is not None else self.last_price()
        # A pullback is a resting limit order (fills at its own price); market
        # and breakout entries slip against us like live taker fills.
        entry = (
            ref_price if entry_kind == "pullback"
            else self._fill_price(ref_price, side)
        )
        equity = self.state["equity"]
        margin = equity * self.cfg.balance_pct
        # SL/TP are set relative to the real (slipped) entry, like live.
        # Stop width is fixed or ATR-scaled; TP is rr x the stop distance,
        # capped at the leg's own price target (or the PM's overall one)
        # when that is nearer.
        stop_pct = self._effective_stop_pct(entry)
        rr = self.cfg.take_profit_rr
        sl_atr, tp_rr, _close_side = _bracket_prices(side, entry, stop_pct, rr)
        sl, size_scale, sl_source = self._sl_with_invalidation(side, entry, sl_atr, leg)
        if sl_source != "atr":
            logger.info(
                "SL widened to plan invalidation: %.4f (ATR stop was %.4f), "
                "size scaled x%.2f to hold dollar risk%s",
                sl, sl_atr, size_scale,
                " [capped]" if sl_source == "invalidation_capped" else "",
            )
        notional = margin * self.cfg.leverage * size_scale
        size = self.kraken.amount_for_notional(notional, entry)
        leg_target = (leg or {}).get("target")
        tp, tp_source = capped_tp(
            side, entry, tp_rr, leg_target if leg_target is not None else analyst_target
        )
        if tp_source == "analyst_target":
            tp_source = "leg_target" if leg_target is not None else tp_source
            logger.info(
                "TP capped to %s price target %.4f (rr bracket wanted %.4f)",
                "leg" if leg_target is not None else "analyst", tp, tp_rr,
            )
        self.state["open"] = {
            "side": side,
            "rating": rating,
            "entry": entry,
            "entry_kind": entry_kind,
            "size": size,
            "sl": sl,
            "sl_source": sl_source,
            "tp": tp,
            "tp_source": tp_source,
            "stop_pct": stop_pct,
            "rr": rr,
            "notional": notional,
            "opened_at": _now_iso(),
        }
        logger.info(
            "OPEN %s %s @ %.4f (ref %.4f, slip %.3f%%, entry %s) size %.4f | "
            "SL %.4f TP %.4f (stop %.2f%% x rr %.1f, mode %s, sl %s, tp %s) | equity %.2f",
            side, self.cfg.symbol, entry, ref_price, self.slippage, entry_kind,
            size, sl, tp, stop_pct, rr, self.cfg.stop_mode, sl_source, tp_source,
            equity,
        )
        self._emit_event(
            "open", symbol=self.cfg.symbol, side=side, rating=rating, entry=entry,
            entry_kind=entry_kind, ref_price=ref_price, slip_pct=self.slippage,
            size=size, sl=sl, sl_source=sl_source, tp=tp, tp_source=tp_source,
            stop_pct=stop_pct, rr=rr, stop_mode=self.cfg.stop_mode,
            notional=notional, equity=equity,
        )

    # ------------------------------------------------------------- monitor
    def monitor(self) -> None:
        pos = self.state["open"]
        _o, high, low, close = self.minute_candle()
        reason, exit_price = check_hit(pos["side"], pos["sl"], pos["tp"], high, low)
        if reason:
            self.close_position(exit_price, reason)
            return
        floor = maybe_arm_profit_lock(
            pos["side"], pos["sl"], pos["tp"], self.cfg.profit_lock_pct,
            high, low, already=pos.get("lock_armed", False),
        )
        if floor is not None:
            pos["sl"] = floor
            pos["lock_armed"] = True
            pos["sl_source"] = "profit_lock"
            logger.info(
                "PROFIT LOCK armed: price within %.2f%% of TP %.4f — floor set "
                "at %.4f, a return there closes in gain",
                self.cfg.profit_lock_pct, pos["tp"], floor,
            )
            self._emit_event(
                "profit_lock", symbol=self.cfg.symbol, side=pos["side"],
                floor=floor, tp=pos["tp"], pct=self.cfg.profit_lock_pct,
            )
            # The arming minute may itself contain the reversal: touched the
            # band, then closed back beyond the floor. Pessimistic reading
            # (peak first, drop after): collect the profit at the floor now.
            reversed_already = (
                close < floor if pos["side"] == "buy" else close > floor
            )
            if reversed_already:
                logger.info(
                    "reversal within the arming minute (1m close %.4f beyond "
                    "floor %.4f) — closing at the floor", close, floor,
                )
                self.close_position(floor, "LOCK")
            return
        logger.info(
            "monitor open %s: 1m range [%.4f, %.4f] vs SL %.4f TP %.4f",
            pos["side"], low, high, pos["sl"], pos["tp"],
        )

    def close_position(self, trigger_price: float, reason: str) -> None:
        pos = self.state["open"]
        # Once the profit lock is armed the SL *is* the floor: a stop-out is a
        # protected profit-taking, not a loss — label it so logs/dashboard/
        # stats tell the two apart.
        if reason == "SL" and pos.get("lock_armed"):
            reason = "LOCK"
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
        self._emit_event(
            "close", symbol=self.cfg.symbol, side=side, rating=pos.get("rating"),
            outcome=reason, exit=exit_price, trigger=trigger_price, pnl=net,
            gross=gross, fees=fees, equity=self.state["equity"],
            wins=self.state["wins"], losses=self.state["losses"],
        )

    def summary(self) -> str:
        s = self.state
        if s["open"]:
            pos = f"OPEN {s['open']['side']} @ {s['open']['entry']:.4f}"
        elif s.get("pending"):
            p = s["pending"]
            pos = f"PENDING {p['side']} ({len(p['legs'])} leg{'s' if len(p['legs']) > 1 else ''})"
        else:
            pos = "flat"
        return (
            f"equity {s['equity']:.2f} | trades {len(s['closed'])} "
            f"(W {s['wins']}/L {s['losses']}) | pnl {s['pnl_total']:+.2f} | {pos}"
        )
