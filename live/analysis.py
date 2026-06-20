"""Run the TradingAgents pipeline and return its decision.

Wraps ``TradingAgentsGraph.propagate(symbol, date, asset_type="crypto")`` with
our Gemini configuration. The crypto pipeline is used so fundamentals (which
make no sense for a coin) are skipped; by default we run only the market,
social and news analysts.

``propagate`` returns ``(full_state, rating)`` where ``rating`` is one of
``Buy / Overweight / Hold / Underweight / Sell``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)


def build_config(cfg) -> dict:
    """Copy DEFAULT_CONFIG and apply our Gemini / debate settings."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = cfg.llm_provider
    config["deep_think_llm"] = cfg.deep_model
    config["quick_think_llm"] = cfg.quick_model
    config["temperature"] = cfg.temperature
    config["max_debate_rounds"] = cfg.debate_rounds
    config["max_risk_discuss_rounds"] = cfg.debate_rounds
    return config


def run_analysis(cfg, trade_date: str | None = None) -> tuple[str, dict]:
    """Run the agents on the configured instrument.

    Returns ``(rating, full_state)``. ``rating`` is the 5-tier Portfolio
    Manager decision string. ``trade_date`` defaults to today (UTC).
    """
    trade_date = trade_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    config = build_config(cfg)

    logger.info(
        "Running analysis: %s on %s (provider=%s deep=%s quick=%s analysts=%s)",
        cfg.analysis_symbol, trade_date, cfg.llm_provider,
        cfg.deep_model, cfg.quick_model, list(cfg.analysts),
    )

    graph = TradingAgentsGraph(
        selected_analysts=cfg.analysts,
        debug=False,
        config=config,
    )
    full_state, rating = graph.propagate(
        cfg.analysis_symbol, trade_date, asset_type="crypto"
    )
    logger.info("Analysis decision for %s: %s", cfg.analysis_symbol, rating)
    return rating, full_state
