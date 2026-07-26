"""Kraken intraday OHLCV: transient drops are retried, not reported as no-data.

One failed attempt used to blind the market analyst for the whole cycle
("data unavailable ... can't make a reliable call") while the paper-sim regime
filter — which retries — kept seeing the same candles. These tests exercise the
retry, the empty-response case, the typed failure after exhaustion, and the
exchange-singleton initialisation. No network anywhere.
"""

from unittest.mock import MagicMock

import pytest

import tradingagents.dataflows.crypto_intraday as ci
from tradingagents.dataflows.symbol_utils import NoMarketDataError

BAR = [1752537600000, 1870.0, 1875.0, 1868.0, 1873.0, 42.0]


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Stub the backoff sleep and isolate the exchange singleton per test."""
    monkeypatch.setattr(ci, "_sleep", lambda s: None)
    monkeypatch.setattr(ci, "_exchange", None)


def _with_exchange(monkeypatch, fetch_side_effect):
    exchange = MagicMock()
    exchange.fetch_ohlcv.side_effect = fetch_side_effect
    monkeypatch.setattr(ci, "_exchange", exchange)
    return exchange


@pytest.mark.unit
def test_transient_error_is_retried(monkeypatch):
    ex = _with_exchange(monkeypatch, [ConnectionError("503"), [BAR]])
    raw = ci._fetch_ohlcv_with_retry("ETH-USD", "ETH/USD", "15m", 300)
    assert raw == [BAR]
    assert ex.fetch_ohlcv.call_count == 2


@pytest.mark.unit
def test_empty_response_is_retried(monkeypatch):
    ex = _with_exchange(monkeypatch, [[], [BAR]])
    raw = ci._fetch_ohlcv_with_retry("ETH-USD", "ETH/USD", "15m", 300)
    assert raw == [BAR]
    assert ex.fetch_ohlcv.call_count == 2


@pytest.mark.unit
def test_exhausted_attempts_raise_typed_no_data(monkeypatch):
    ex = _with_exchange(monkeypatch, ConnectionError("kraken down"))
    with pytest.raises(NoMarketDataError) as err:
        ci._fetch_ohlcv_with_retry("ETH-USD", "ETH/USD", "15m", 300)
    assert ex.fetch_ohlcv.call_count == ci._OHLCV_ATTEMPTS
    assert "kraken down" in str(err.value)


@pytest.mark.unit
def test_fetch_intraday_ohlcv_survives_one_drop(monkeypatch):
    _with_exchange(monkeypatch, [ConnectionError("blip"), [BAR]])
    df = ci.fetch_intraday_ohlcv("ETH-USD")
    assert list(df["Close"]) == [1873.0]
    assert "vwap" in df.columns


@pytest.mark.unit
def test_failed_load_markets_does_not_poison_singleton(monkeypatch):
    """A transient failure while loading markets must leave the singleton
    unset, so the next call can retry the initialisation from scratch."""
    calls = {"n": 0}
    exchange = MagicMock()

    def load_markets():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("first load fails")

    exchange.load_markets.side_effect = load_markets
    fake_ccxt = MagicMock()
    fake_ccxt.kraken.return_value = exchange
    monkeypatch.setitem(__import__("sys").modules, "ccxt", fake_ccxt)

    with pytest.raises(ConnectionError):
        ci._get_exchange()
    assert ci._exchange is None  # not cached half-initialised

    assert ci._get_exchange() is exchange  # second attempt succeeds
    assert ci._exchange is exchange
