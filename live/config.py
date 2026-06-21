"""Configuration for the live Kraken Futures bot, sourced from env vars.

The same Docker image is deployed once per instrument on Railway by changing
only the environment. Defaults match the agreed test setup:

    instrument : Solana perpetual (PF_SOLUSD)
    leverage   : 5x, isolated margin
    stops      : +/-0.5% of entry price (both take-profit and stop-loss)
    entry      : ~98% of account balance as margin
    direction  : long + short (BUY->long, SELL->short, HOLD->nothing)
    mode       : DRY-RUN until LIVE=true is set explicitly
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _s(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _f(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _i(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _b(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Kraken quotes Bitcoin as XBT; map it back to the BTC base that yfinance /
# TradingAgents expect. Other bases are identical on both sides.
_KRAKEN_BASE_ALIASES = {"XBT": "BTC"}


def _derive_analysis_symbol(kraken_symbol: str) -> str:
    """Turn a Kraken perpetual id (``PF_XBTUSD``) into a ``BASE-USD`` symbol.

    TradingAgents / yfinance want ``SOL-USD``, ``BTC-USD`` etc. We strip the
    ``PF_`` prefix and ``USD`` quote, then de-alias the base (XBT -> BTC).
    """
    s = kraken_symbol.upper()
    for prefix in ("PF_", "PI_", "FI_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith("USD"):
        s = s[: -len("USD")]
    base = _KRAKEN_BASE_ALIASES.get(s, s)
    return f"{base}-USD"


@dataclass
class Config:
    # --- instrument -------------------------------------------------------
    symbol: str            # Kraken Futures native id, e.g. "PF_SOLUSD"
    analysis_symbol: str   # symbol fed to TradingAgents/yfinance, e.g. "SOL-USD"

    # --- intraday analysis (Path 1) --------------------------------------
    intraday: bool             # analyse intraday bars instead of daily candles
    intraday_timeframe: str    # ccxt timeframe for the analysis bars, e.g. "15m"
    intraday_bars: int         # how many recent bars to fetch for the analysis

    # --- sizing / risk ----------------------------------------------------
    leverage: float        # target leverage (isolated margin)
    max_leverage: float    # hard cap; bot refuses to exceed this
    stop_pct: float        # +/- percent of entry price for SL and TP
    balance_pct: float     # fraction of balance committed as margin (0-1)
    margin_currency: str    # collateral currency to read from the wallet
    equity_floor_usd: float  # kill-switch: no new trades below this equity

    # --- behaviour --------------------------------------------------------
    allow_short: bool      # if False, SELL/Underweight => stay flat
    live: bool             # if False, orders are logged but NOT sent
    no_broker: bool        # if True, skip Kraken auth entirely (offline test)
    paper_balance: float   # fake equity (USD) used when no_broker
    reattach_stops: bool   # if a live position lacks SL/TP, re-place them
    trigger_signal: str    # "mark" | "index" | "last" for stop triggers
    long_signals: frozenset[str]
    short_signals: frozenset[str]

    # --- LLM / analysis ---------------------------------------------------
    llm_provider: str
    deep_model: str
    quick_model: str
    temperature: float
    debate_rounds: int
    analysts: tuple[str, ...]
    google_thinking_level: str       # "" leaves the model default; "low"/"high" tune latency
    analysis_max_attempts: int       # retries on transient network drops during analysis
    stream_llm: bool                 # stream Gemini calls (avoids ~60s non-streaming cutoff)
    debug: bool                      # print every analyst/agent node output as it runs

    @classmethod
    def from_env(cls) -> "Config":
        symbol = _s("SYMBOL", "PF_SOLUSD")
        analysis_symbol = _s("ANALYSIS_SYMBOL", "") or _derive_analysis_symbol(symbol)

        long_sigs = frozenset(
            x.strip() for x in _s("LONG_SIGNALS", "Buy,Overweight").split(",") if x.strip()
        )
        short_sigs = frozenset(
            x.strip() for x in _s("SHORT_SIGNALS", "Sell,Underweight").split(",") if x.strip()
        )
        analysts = tuple(
            x.strip() for x in _s("ANALYSTS", "market,social,news").split(",") if x.strip()
        )

        # Offline mode can never send orders: it has no authenticated session.
        no_broker = _b("NO_BROKER", False)
        live = _b("LIVE", False) and not no_broker

        return cls(
            symbol=symbol,
            analysis_symbol=analysis_symbol,
            intraday=_b("INTRADAY", True),
            intraday_timeframe=_s("INTRADAY_TIMEFRAME", "15m"),
            intraday_bars=_i("INTRADAY_BARS", 300),
            leverage=_f("LEVERAGE", 5.0),
            max_leverage=_f("MAX_LEVERAGE", 5.0),
            stop_pct=_f("STOP_PCT", 0.5),
            balance_pct=_f("BALANCE_PCT", 0.98),
            margin_currency=_s("MARGIN_CURRENCY", "USD"),
            equity_floor_usd=_f("EQUITY_FLOOR_USD", 0.0),
            allow_short=_b("ALLOW_SHORT", True),
            live=live,
            no_broker=no_broker,
            paper_balance=_f("PAPER_BALANCE", 108.0),
            reattach_stops=_b("REATTACH_STOPS", True),
            trigger_signal=_s("TRIGGER_SIGNAL", "mark"),
            long_signals=long_sigs,
            short_signals=short_sigs,
            llm_provider=_s("LLM_PROVIDER", "google"),
            deep_model=_s("DEEP_MODEL", "gemini-2.5-pro"),
            quick_model=_s("QUICK_MODEL", "gemini-2.5-flash"),
            temperature=_f("TEMPERATURE", 0.0),
            debate_rounds=_i("DEBATE_ROUNDS", 1),
            analysts=analysts,
            google_thinking_level=_s("GOOGLE_THINKING_LEVEL", ""),
            analysis_max_attempts=_i("ANALYSIS_MAX_ATTEMPTS", 3),
            stream_llm=_b("STREAM_LLM", True),
            debug=_b("DEBUG", True),
        )

    def summary(self) -> dict:
        """Loggable, secret-free snapshot of the active configuration."""
        return {
            "symbol": self.symbol,
            "analysis_symbol": self.analysis_symbol,
            "leverage": self.leverage,
            "stop_pct": self.stop_pct,
            "balance_pct": self.balance_pct,
            "allow_short": self.allow_short,
            "live": self.live,
            "no_broker": self.no_broker,
            "paper_balance": self.paper_balance,
            "deep_model": self.deep_model,
            "quick_model": self.quick_model,
            "analysts": list(self.analysts),
        }
