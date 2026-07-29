"""Thin wrapper over ccxt's ``krakenfutures`` exchange.

All *reads* (balance, positions, price, open orders) always hit the real API —
we want true account state even in dry-run. All *writes* (orders, cancels,
leverage changes) are gated by ``cfg.live``: when False they are logged and
return a ``{"dry_run": True, ...}`` stub instead of being sent.

NOTE on going live: the exact ccxt param names for stop/take-profit on
krakenfutures (``stopLossPrice`` / ``takeProfitPrice`` vs ``triggerPrice`` +
``reduceOnly``) can vary by ccxt version. They are isolated to
``create_stop_loss`` / ``create_take_profit`` below — validate there against the
installed ccxt before flipping LIVE=true.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class KrakenClient:
    def __init__(self, cfg, api_key: str, api_secret: str):
        self.cfg = cfg
        self._api_key = api_key
        self._api_secret = api_secret
        self.exchange = None
        self.market: dict[str, Any] | None = None
        self.ccxt_symbol: str | None = None

    # ------------------------------------------------------------------ setup
    def connect(self) -> None:
        """Instantiate the exchange, load markets, resolve our symbol."""
        import ccxt  # imported here so config/tests don't require ccxt installed

        if self.cfg.no_broker:
            # Offline test: public endpoints only (real prices + market specs),
            # no API key, no authenticated calls. Balance is faked.
            self.exchange = ccxt.krakenfutures({"enableRateLimit": True})
            logger.info("NO_BROKER mode: public data only, paper balance=%.2f", self.cfg.paper_balance)
        else:
            self.exchange = ccxt.krakenfutures(
                {
                    "apiKey": self._api_key,
                    "secret": self._api_secret,
                    "enableRateLimit": True,
                }
            )
        self.exchange.load_markets()
        self.market = self._resolve_market(self.cfg.symbol)
        self.ccxt_symbol = self.market["symbol"]
        logger.info(
            "Connected to Kraken Futures: %s -> ccxt symbol %s (min size %s)",
            self.cfg.symbol, self.ccxt_symbol, self.min_amount(),
        )

    def _resolve_market(self, symbol: str) -> dict[str, Any]:
        """Find the ccxt market whose native id or unified symbol matches.

        Accepts both the Kraken id (``PF_SOLUSD``) and the ccxt unified symbol
        (``SOL/USD:USD``) so config can use whichever is handier.
        """
        for m in self.exchange.markets.values():
            if symbol in (m.get("id"), m.get("symbol")):
                return m
        raise ValueError(
            f"Symbol {symbol!r} not found among Kraken Futures markets. "
            "Check the id via GET /instruments (e.g. PF_SOLUSD)."
        )

    def min_amount(self) -> float:
        try:
            return float(self.market["limits"]["amount"]["min"] or 0.0)
        except (KeyError, TypeError):
            return 0.0

    # ------------------------------------------------------------------ reads
    def get_equity_usd(self) -> float:
        """Best-effort account equity in the configured margin currency.

        Multi-collateral (flex) wallets report per-currency totals; we read the
        margin currency total and fall back to free balance. Logged raw in
        dry-run so we can tune this once real keys are connected.
        """
        if self.cfg.no_broker:
            return float(self.cfg.paper_balance)
        bal = self.exchange.fetch_balance()
        cur = self.cfg.margin_currency
        total = (bal.get("total") or {}).get(cur)
        free = (bal.get("free") or {}).get(cur)
        equity = total if total not in (None, 0) else free
        if equity in (None, 0):
            logger.warning(
                "Could not read %s equity from wallet; raw balance info=%s",
                cur, bal.get("info"),
            )
            return 0.0
        return float(equity)

    def get_last_price(self) -> float:
        ticker = self.exchange.fetch_ticker(self.ccxt_symbol)
        price = ticker.get("last") or ticker.get("mark") or ticker.get("close")
        if not price:
            raise RuntimeError(f"No price for {self.ccxt_symbol}: {ticker}")
        return float(price)

    def get_open_position(self) -> dict[str, Any] | None:
        """Return the open position dict for our symbol, or None if flat."""
        if self.cfg.no_broker:
            return None  # offline: no account to read, assume flat
        positions = self.exchange.fetch_positions([self.ccxt_symbol])
        for p in positions:
            contracts = p.get("contracts")
            if contracts and float(contracts) != 0.0:
                return p
        return None

    def get_open_orders(self) -> list[dict[str, Any]]:
        if self.cfg.no_broker:
            return []  # offline: no authenticated order book to read
        return self.exchange.fetch_open_orders(self.ccxt_symbol)

    def amount_for_notional(self, notional_usd: float, price: float) -> float:
        """Convert a USD notional into a precision-rounded contract size."""
        from ccxt.base.errors import InvalidOrder

        raw = notional_usd / price
        try:
            return float(self.exchange.amount_to_precision(self.ccxt_symbol, raw))
        except InvalidOrder:
            # raw amount is below the exchange minimum precision → return 0
            # so the caller's min-size veto can handle it gracefully.
            return 0.0

    # ----------------------------------------------------------------- writes
    def set_leverage_isolated(self, leverage: float) -> dict[str, Any]:
        if not self.cfg.live:
            logger.info("[DRY-RUN] set leverage %sx isolated on %s", leverage, self.ccxt_symbol)
            return {"dry_run": True, "leverage": leverage}
        # maxLeverage present => Kraken sets ISOLATED margin (see leveragepreferences).
        return self.exchange.set_leverage(leverage, self.ccxt_symbol)

    def create_market_entry(self, side: str, amount: float) -> dict[str, Any]:
        if not self.cfg.live:
            logger.info("[DRY-RUN] ENTRY market %s %s %s", side, amount, self.ccxt_symbol)
            return {"dry_run": True, "side": side, "amount": amount, "type": "market"}
        return self.exchange.create_order(self.ccxt_symbol, "market", side, amount)

    def create_stop_loss(self, close_side: str, amount: float, stop_price: float) -> dict[str, Any]:
        params = {
            "reduceOnly": True,
            "stopLossPrice": stop_price,   # ccxt unified SL trigger -> Kraken 'stp'
            "triggerSignal": self.cfg.trigger_signal,
        }
        if not self.cfg.live:
            logger.info("[DRY-RUN] STOP-LOSS %s %s @ trigger %s", close_side, amount, stop_price)
            return {"dry_run": True, **params, "side": close_side, "amount": amount}
        return self.exchange.create_order(
            self.ccxt_symbol, "market", close_side, amount, None, params
        )

    def create_take_profit(self, close_side: str, amount: float, tp_price: float) -> dict[str, Any]:
        params = {
            "reduceOnly": True,
            "takeProfitPrice": tp_price,   # ccxt unified TP trigger -> Kraken 'take_profit'
            "triggerSignal": self.cfg.trigger_signal,
        }
        if not self.cfg.live:
            logger.info("[DRY-RUN] TAKE-PROFIT %s %s @ trigger %s", close_side, amount, tp_price)
            return {"dry_run": True, **params, "side": close_side, "amount": amount}
        return self.exchange.create_order(
            self.ccxt_symbol, "market", close_side, amount, None, params
        )

    def cancel_orphan_orders(self) -> int:
        """Cancel leftover reduce-only SL/TP orders when the account is flat.

        Kraken does not guarantee OCO: when one protective order fills and
        closes the position, the sibling remains open. We clear them before
        opening a new position so a stale trigger can't fire on it.
        """
        orders = self.get_open_orders()
        if not orders:
            return 0
        if not self.cfg.live:
            logger.info("[DRY-RUN] would cancel %d orphan order(s) on %s", len(orders), self.ccxt_symbol)
            return len(orders)
        cancelled = 0
        for o in orders:
            try:
                self.exchange.cancel_order(o["id"], self.ccxt_symbol)
                cancelled += 1
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Failed to cancel order %s: %s", o.get("id"), e)
        return cancelled
