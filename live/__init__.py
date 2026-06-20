"""Live trading layer on top of TradingAgents.

This package wires the TradingAgents analysis pipeline to a Kraken Futures
account. It is intentionally separate from the upstream ``tradingagents``
package so pulling updates from upstream never conflicts with our code.

Entry point: ``python -m live.runner`` (one cron cycle, then exits).
"""
