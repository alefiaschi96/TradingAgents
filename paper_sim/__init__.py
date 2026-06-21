"""Paper-trading simulator on live Kraken prices.

A standalone module that reuses the existing analysis + sizing/bracket logic
(read-only) but never sends orders: it tracks ONE simulated position and checks
the live price each minute against the stop-loss / take-profit to record the
outcome. Nothing in tradingagents/ or live/ is modified.

Entry point: ``python -m paper_sim.run``
"""
