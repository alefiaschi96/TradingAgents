"""Kraken Futures market-structure tools (funding / OI / order book).

Ported from the ale branch (2026-07): the intraday market analyst binds these
so its report carries derivatives positioning, not just spot OHLCV. Contract
under test: correct symbol mapping, readable one-line summaries, graceful
degradation to 'data unavailable' on ANY error (a flaky public futures feed
must never crash an analysis), and the retry policy: transient drops get 3
attempts with backoff — like the core OHLCV fetch — while structural errors
(ccxt NotSupported) fail fast to their fallback path.
"""

import pytest

import tradingagents.agents.utils.futures_data_tools as fdt
from tradingagents.agents.utils.futures_data_tools import (
    _futures_symbol,
    get_funding_rate,
    get_open_interest,
    get_orderbook_imbalance,
)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch, tmp_path):
    """Stub the retry backoff and sandbox the OI history store per test."""
    monkeypatch.setattr(fdt, "_sleep", lambda s: None)
    monkeypatch.setattr(fdt, "_OI_HISTORY_PATH", str(tmp_path / "oi_history.json"))


def test_symbol_mapping():
    assert _futures_symbol("ETH-USD") == "ETH/USD:USD"
    assert _futures_symbol("sol-usd") == "SOL/USD:USD"
    assert _futures_symbol("ETH/USD:USD") == "ETH/USD:USD"  # ccxt style passes through


class _FakeExchange:
    """Canned public-endpoint responses; no network."""

    def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.0003}

    def fetch_funding_rate_history(self, symbol, limit=8):
        return [{"fundingRate": 0.0001}, {"fundingRate": 0.0003}]

    def fetch_open_interest(self, symbol):
        return {"openInterestAmount": 125_000.0}

    def fetch_open_interest_history(self, symbol, timeframe="1h", limit=8):
        return [{"openInterestAmount": 100_000.0}, {"openInterestAmount": 125_000.0}]

    def fetch_order_book(self, symbol, limit=10):
        return {
            "bids": [[1840.0, 7.0]] * 10,   # 70 total
            "asks": [[1841.0, 3.0]] * 10,   # 30 total -> bid-heavy
        }


class _BrokenExchange:
    def __getattr__(self, name):
        raise RuntimeError("public endpoint down")


@pytest.fixture
def fake_exchange():
    fdt._exchange = _FakeExchange()
    yield
    fdt._exchange = None


@pytest.fixture
def broken_exchange():
    fdt._exchange = _BrokenExchange()
    yield
    fdt._exchange = None


@pytest.mark.unit
def test_funding_rate_reads_crowded_long(fake_exchange):
    out = get_funding_rate.invoke({"symbol": "ETH-USD"})
    assert "ETH/USD:USD" in out and "longs pay shorts" in out
    assert "rising" in out  # 0.0001 -> 0.0003 history

@pytest.mark.unit
def test_open_interest_reads_rising(fake_exchange):
    out = get_open_interest.invoke({"symbol": "ETH-USD"})
    assert "125,000" in out and "rising" in out and "+25.00%" in out


@pytest.mark.unit
def test_orderbook_reads_bid_heavy(fake_exchange):
    out = get_orderbook_imbalance.invoke({"symbol": "ETH-USD"})
    assert "bids 70.0% vs asks 30.0%" in out and "bid-heavy" in out


class _TickerOnlyExchange(_FakeExchange):
    """ccxt krakenfutures reality: no fetchOpenInterest endpoint — the live OI
    comes from the public ticker instead."""

    def fetch_open_interest(self, symbol):
        raise RuntimeError("krakenfutures fetchOpenInterest() is not supported yet")

    def fetch_open_interest_history(self, symbol, timeframe="1h", limit=8):
        raise RuntimeError("not supported")

    def fetch_ticker(self, symbol):
        return {"info": {"openInterest": 27715.935}}


@pytest.mark.unit
def test_open_interest_falls_back_to_ticker():
    fdt._exchange = _TickerOnlyExchange()
    try:
        out = get_open_interest.invoke({"symbol": "ETH-USD"})
        assert "27,716" in out and "unavailable" not in out
    finally:
        fdt._exchange = None


@pytest.mark.unit
def test_all_tools_degrade_to_unavailable(broken_exchange):
    for tool in (get_funding_rate, get_open_interest, get_orderbook_imbalance):
        assert tool.invoke({"symbol": "ETH-USD"}) == "data unavailable"


# ------------------------------------------------------------- retry policy

class _FlakyExchange(_FakeExchange):
    """First two funding calls drop, the third succeeds: retries absorb it."""

    def __init__(self):
        self.calls = 0

    def fetch_funding_rate(self, symbol):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("connection dropped")
        return {"fundingRate": 0.0003}


@pytest.mark.unit
def test_transient_drops_are_retried_and_absorbed():
    ex = _FlakyExchange()
    fdt._exchange = ex
    try:
        out = get_funding_rate.invoke({"symbol": "ETH-USD"})
        assert "longs pay shorts" in out
        assert ex.calls == 3  # two drops + the successful third attempt
    finally:
        fdt._exchange = None


class _AlwaysDownExchange(_FakeExchange):
    def __init__(self):
        self.calls = 0

    def fetch_order_book(self, symbol, limit=10):
        self.calls += 1
        raise RuntimeError("endpoint down")


@pytest.mark.unit
def test_degrades_only_after_three_attempts():
    ex = _AlwaysDownExchange()
    fdt._exchange = ex
    try:
        assert get_orderbook_imbalance.invoke({"symbol": "ETH-USD"}) == "data unavailable"
        assert ex.calls == 3
    finally:
        fdt._exchange = None


class NotSupported(Exception):
    """Same class NAME as ccxt's — the fail-fast check matches by name."""


class _NotSupportedOIExchange(_FakeExchange):
    def __init__(self):
        self.oi_calls = 0

    def fetch_open_interest(self, symbol):
        self.oi_calls += 1
        raise NotSupported("krakenfutures fetchOpenInterest() is not supported yet")

    def fetch_open_interest_history(self, symbol, timeframe="1h", limit=8):
        raise NotSupported("not supported")

    def fetch_ticker(self, symbol):
        return {"info": {"openInterest": 27715.935}}


@pytest.mark.unit
def test_not_supported_fails_fast_to_the_ticker_fallback():
    ex = _NotSupportedOIExchange()
    fdt._exchange = ex
    try:
        out = get_open_interest.invoke({"symbol": "ETH-USD"})
        assert "27,716" in out
        assert ex.oi_calls == 1  # structural error: no pointless retries
    finally:
        fdt._exchange = None


# ------------------------------------------------- local OI change series

NOW = 1_800_000_000.0


@pytest.mark.unit
def test_oi_history_first_reading_has_no_priors(tmp_path):
    p = str(tmp_path / "oi.json")
    assert fdt._oi_history_update("ETH/USD:USD", 26_000.0, now=NOW, path=p) == []
    prior = fdt._oi_history_update("ETH/USD:USD", 26_300.0, now=NOW + 2400, path=p)
    assert prior == [[NOW, 26_000.0]]  # second call sees the first reading


@pytest.mark.unit
def test_oi_history_prunes_old_points_and_tolerates_corruption(tmp_path):
    p = str(tmp_path / "oi.json")
    (tmp_path / "oi.json").write_text("{not json")  # corrupt store: fresh start
    assert fdt._oi_history_update("X", 1.0, now=NOW, path=p) == []
    # a reading older than the window is dropped on the next update
    prior = fdt._oi_history_update("X", 2.0, now=NOW + fdt._OI_HISTORY_WINDOW_SEC + 1, path=p)
    assert prior == []


@pytest.mark.unit
def test_oi_change_text_rising_and_falling():
    prior = [[NOW - 2400, 26_000.0]]  # one reading, 40 min ago
    up = fdt._oi_change_text(prior, 26_312.0, now=NOW)
    assert "rising" in up and "+1.20%" in up and "40 min" in up
    down = fdt._oi_change_text(prior, 25_688.0, now=NOW)
    assert "falling" in down and "-1.20%" in down


@pytest.mark.unit
def test_oi_change_needs_minimum_span():
    # back-to-back readings inside one analysis: no meaningless trend claim
    prior = [[NOW - 60, 26_000.0]]
    assert fdt._oi_change_text(prior, 26_300.0, now=NOW) == ""
    assert fdt._oi_change_text([], 26_300.0, now=NOW) == ""


class _MutableTickerExchange(_TickerOnlyExchange):
    def __init__(self, oi):
        super().__init__()
        self.oi = oi

    def fetch_ticker(self, symbol):
        return {"info": {"openInterest": self.oi}}


@pytest.mark.unit
def test_tool_reports_change_across_calls(tmp_path, monkeypatch):
    """End-to-end: first call records silently, second call states the change."""
    import json as _json
    store = tmp_path / "oi_history.json"
    monkeypatch.setattr(fdt, "_OI_HISTORY_PATH", str(store))
    fdt._exchange = _MutableTickerExchange(26_000.0)
    try:
        first = get_open_interest.invoke({"symbol": "ETH-USD"})
        assert "26,000" in first and "Recent OI" not in first  # first reading
        # age the stored reading by 40 minutes, then move the OI
        data = _json.loads(store.read_text())
        data["ETH/USD:USD"][0][0] -= 2400
        store.write_text(_json.dumps(data))
        fdt._exchange.oi = 26_312.0
        second = get_open_interest.invoke({"symbol": "ETH-USD"})
        assert "26,312" in second and "Recent OI rising +1.20%" in second
        assert "(2 readings)" in second
    finally:
        fdt._exchange = None
