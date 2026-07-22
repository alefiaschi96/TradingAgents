"""Single-pass decision pipeline: Signal Synthesizer -> Critic Manager -> Risk Manager.

Replaces the old bull/bear debate + Research Manager and the aggressive/
neutral/conservative risk debate + Portfolio Manager with three straight-line
agents (no back-and-forth rounds), scoped to intraday crypto trading.
"""

from .critic_manager import create_critic_manager
from .risk_manager import create_risk_manager
from .signal_synthesizer import create_signal_synthesizer

__all__ = [
    "create_critic_manager",
    "create_risk_manager",
    "create_signal_synthesizer",
]
