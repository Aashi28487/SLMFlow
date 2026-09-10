"""Per-query routing: heuristic, learned, and the measurement baselines."""

from slmflow.routing.features import FEATURE_NAMES, extract_features, feature_vector
from slmflow.routing.heuristic import (
    HeuristicRouter,
    HeuristicWeights,
    OracleRouter,
    RandomRouter,
    RouterRegret,
    StaticRouter,
    router_regret,
)

__all__ = [
    "FEATURE_NAMES",
    "HeuristicRouter",
    "HeuristicWeights",
    "OracleRouter",
    "RandomRouter",
    "RouterRegret",
    "StaticRouter",
    "extract_features",
    "feature_vector",
    "router_regret",
]
