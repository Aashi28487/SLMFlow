"""Pluggable execution strategies, including SLM-MUX and the C4 optimisations."""

from slmflow.strategies.adaptive import (
    AdaptiveSelfConsistency,
    CascadeStrategy,
    SingleStrategy,
)
from slmflow.strategies.base import BaseStrategy, GenerationBatch, StrategyContext
from slmflow.strategies.slm_mux import (
    AgentForestStrategy,
    SLMMuxStrategy,
    select_model_set,
)

__all__ = [
    "AdaptiveSelfConsistency",
    "AgentForestStrategy",
    "BaseStrategy",
    "CascadeStrategy",
    "GenerationBatch",
    "SLMMuxStrategy",
    "SingleStrategy",
    "StrategyContext",
    "select_model_set",
]
