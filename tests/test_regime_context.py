"""Offline tests for live.regime.get_regime_context.

No network: the OHLCV fetch is injected with synthetic candles. We check that
a clear uptrend yields a sensible sentence and that any fetch error fails open
to "".
"""

from live.regime import get_regime_context


def _synthetic_uptrend(symbol, timeframe, limit):
    """A clean rising series. OHLCV rows: [ts, open, high, low, close, volume]."""
    rows = []
    price = 100.0
    for i in range(limit):
        close = price + i * 1.0  # steadily rising closes -> rising EMA, price above it
        high = close + 0.5
        low = close - 0.5
        rows.append([i * 3600_000, close, high, low, close, 10.0])
    return rows


def test_uptrend_returns_sensible_sentence():
    out = get_regime_context(
        "PF_SOLUSD", timeframe="1h", ema_period=20, fetch_ohlcv=_synthetic_uptrend
    )
    assert isinstance(out, str)
    assert out, "expected a non-empty regime sentence for a clean uptrend"
    assert "Higher-timeframe (1h) regime:" in out
    assert "UPTREND" in out
    assert "20-bar EMA" in out
    assert "ATR from it" in out


def test_fetch_error_fails_open_to_empty_string():
    def _boom(symbol, timeframe, limit):
        raise RuntimeError("charts endpoint down")

    out = get_regime_context(
        "PF_SOLUSD", timeframe="1h", ema_period=20, fetch_ohlcv=_boom
    )
    assert out == ""


def test_insufficient_data_returns_empty_string():
    def _too_few(symbol, timeframe, limit):
        return [[0, 100, 101, 99, 100, 1.0]]  # 1 candle, below period+2

    out = get_regime_context(
        "PF_SOLUSD", timeframe="1h", ema_period=20, fetch_ohlcv=_too_few
    )
    assert out == ""
