"""Resource-aware routing. Rules first, then a learned router, then cascade.

Phase 7's contribution is that routing consults **live memory and latency
state**, not just a static capability table. A model that is already resident
costs no load; on this hardware that can dominate the quality difference
between two similar models, which is exactly what price-per-token routers
(RouteLLM, FrugalGPT) have no term for.

Every router here implements :class:`~slmflow.types.Router`. The two baselines
the brief requires for honest comparison are included:
:class:`RandomRouter` and :class:`StaticRouter`.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from slmflow.errors import ModelNotFound, ScheduleInfeasible
from slmflow.telemetry.trace import get_logger
from slmflow.types import ModelCard, SubTask, TaskType

__all__ = [
    "HeuristicRouter",
    "RandomRouter",
    "StaticRouter",
    "OracleRouter",
    "RouterRegret",
    "router_regret",
]

_LOG = get_logger("routing")


def _require(candidates: Sequence[ModelCard], subtask: SubTask) -> Sequence[ModelCard]:
    """Reject an empty candidate list with an actionable message."""
    if not candidates:
        raise ScheduleInfeasible(
            f"No candidate models for sub-task {subtask.name!r}. Either the memory budget "
            f"excludes every registered model, or candidate_models names an unregistered id."
        )
    return candidates


class RandomRouter:
    """Uniform random choice. Baseline 1: what routing must beat to exist."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        """Seeded for reproducibility (Ground Rule 11)."""
        self._rng = random.Random(seed)

    def select(
        self,
        subtask: SubTask,
        candidates: Sequence[ModelCard],
        *,
        resident: Sequence[str] = (),
    ) -> ModelCard:
        """Pick uniformly at random."""
        return self._rng.choice(list(_require(candidates, subtask)))


class StaticRouter:
    """Always the same model. Baseline 2: the single-model control.

    This is what every framework without a router effectively does, and it is
    the honest comparison for "does per-query routing pay for itself?".
    """

    name = "static"

    def __init__(self, model_id: str) -> None:
        """Args: model_id: the model to always return."""
        self.model_id = model_id
        self.name = f"static:{model_id}"

    def select(
        self,
        subtask: SubTask,
        candidates: Sequence[ModelCard],
        *,
        resident: Sequence[str] = (),
    ) -> ModelCard:
        """Return the pinned model.

        Raises:
            ModelNotFound: When the pinned model is not among the candidates,
                usually because it does not fit the budget.
        """
        for card in _require(candidates, subtask):
            if card.model_id == self.model_id:
                return card
        raise ModelNotFound(self.model_id, [c.model_id for c in candidates])


@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    """Tunable terms of the heuristic score.

    Defaults are deliberate but arbitrary; Phase 7 must report the router
    against baselines rather than assert that these numbers are good.
    """

    quality: float = 1.0
    warm_bonus: float = 0.05
    size_penalty: float = 0.02
    unprofiled_penalty: float = 0.5
    latency_penalty: float = 0.0


class HeuristicRouter:
    """Rules over the Phase 3 complementarity matrix plus live state.

    Score for each candidate:

        quality * measured_accuracy
        + warm_bonus        if already resident
        - size_penalty      * resident_GB
        - latency_penalty   * predicted_seconds
        - unprofiled_penalty if the card has no measurement for this task

    The ``warm_bonus`` term is the resource-aware part: it is what lets a
    slightly weaker resident model win, and it is the term no price-based
    router has.

    Example:
        >>> from slmflow.models.backends.mock import make_pool
        >>> from slmflow.types import SubTask, TaskSpec, TaskType
        >>> pool = make_pool([("a", 0.5, 400_000_000), ("b", 3.0, 2_000_000_000)],
        ...                  accuracy={"a": {"classify": 0.7}, "b": {"classify": 0.8}})
        >>> st = SubTask("s", TaskSpec("x", TaskType.CLASSIFY))
        >>> HeuristicRouter().select(st, pool).model_id
        'b'
        >>> HeuristicRouter(HeuristicWeights(warm_bonus=0.5)).select(
        ...     st, pool, resident=["a"]).model_id
        'a'
    """

    name = "heuristic"

    def __init__(
        self,
        weights: HeuristicWeights | None = None,
        *,
        prefer_small_for: Sequence[TaskType] = (TaskType.CLASSIFY, TaskType.EXTRACT),
    ) -> None:
        """Args:
        weights: Score term weights.
        prefer_small_for: Task types where the size penalty is doubled,
            encoding the Phase 3 finding that mechanical tasks do not
            reward capacity. Adjust once real complementarity data exists.
        """
        self.weights = weights or HeuristicWeights()
        self.prefer_small_for = tuple(prefer_small_for)

    def select(
        self,
        subtask: SubTask,
        candidates: Sequence[ModelCard],
        *,
        resident: Sequence[str] = (),
    ) -> ModelCard:
        """Pick the highest-scoring candidate."""
        pool = _require(candidates, subtask)
        if subtask.pinned_model:
            for card in pool:
                if card.model_id == subtask.pinned_model:
                    return card
            raise ModelNotFound(subtask.pinned_model, [c.model_id for c in pool])

        resident_set = set(resident)
        scored = [(self._score(subtask, c, resident_set), c) for c in pool]
        best = max(scored, key=lambda pair: pair[0])
        _LOG.info(
            "route selected",
            extra={
                "event": "route",
                "slmflow": {
                    "step": subtask.name,
                    "chosen": best[1].model_id,
                    "warm": best[1].model_id in resident_set,
                    "scores": {c.model_id: round(s, 4) for s, c in scored},
                },
            },
        )
        return best[1]

    def _score(self, subtask: SubTask, card: ModelCard, resident: set[str]) -> float:
        """Compute one candidate's score."""
        w = self.weights
        accuracy = card.accuracy_for(subtask.spec.task_type)
        score = w.quality * (accuracy if accuracy is not None else 0.0)
        if accuracy is None:
            score -= w.unprofiled_penalty
        if card.model_id in resident:
            score += w.warm_bonus
        size_gb = card.footprint.resident_bytes / 1e9
        multiplier = 2.0 if subtask.spec.task_type in self.prefer_small_for else 1.0
        score -= w.size_penalty * size_gb * multiplier
        if w.latency_penalty:
            seconds = card.expected_seconds(256, subtask.spec.max_output_tokens)
            if seconds is not None:
                score -= w.latency_penalty * seconds
        return score


class OracleRouter:
    """Picks the best model in hindsight. For measuring **router regret** only.

    Requires a per-(task, model) correctness table built from a completed run,
    so it is not usable online. Phase 7's DoD requires router regret against
    exactly this.
    """

    name = "oracle_hindsight"

    def __init__(self, correctness: Mapping[tuple[str, str], float]) -> None:
        """Args: correctness: ``{(task_id, model_id): score}`` from a real run."""
        self._correctness = dict(correctness)

    def select(
        self,
        subtask: SubTask,
        candidates: Sequence[ModelCard],
        *,
        resident: Sequence[str] = (),
    ) -> ModelCard:
        """Return the candidate that actually scored best on this exact task."""
        pool = _require(candidates, subtask)
        task_id = subtask.spec.task_id
        scored = [
            (self._correctness.get((task_id, c.model_id), float("-inf")), c) for c in pool
        ]
        best = max(scored, key=lambda pair: pair[0])
        if best[0] == float("-inf"):
            # No record for this task: fall back to the smallest, and say so.
            _LOG.warning(
                "oracle router has no record for this task",
                extra={
                    "event": "oracle_router_miss",
                    "slmflow": {"task_id": task_id, "step": subtask.name},
                },
            )
            return min(pool, key=lambda c: c.footprint.resident_bytes)
        return best[1]


@dataclass(frozen=True, slots=True)
class RouterRegret:
    """Accuracy the router gives up against hindsight-optimal selection."""

    router: str
    router_accuracy: float
    oracle_accuracy: float
    n_items: int

    @property
    def regret(self) -> float:
        """Accuracy points left on the table. Lower is better."""
        return self.oracle_accuracy - self.router_accuracy

    @property
    def relative_regret(self) -> float | None:
        """Regret as a fraction of the achievable ceiling."""
        if self.oracle_accuracy <= 0:
            return None
        return self.regret / self.oracle_accuracy

    def as_row(self) -> dict[str, object]:
        """Flat dict for a results table."""
        return {
            "router": self.router,
            "router_accuracy": round(self.router_accuracy, 4),
            "oracle_accuracy": round(self.oracle_accuracy, 4),
            "regret": round(self.regret, 4),
            "relative_regret": (
                None if self.relative_regret is None else round(self.relative_regret, 4)
            ),
            "n": self.n_items,
        }


def router_regret(
    router_name: str,
    chosen: Sequence[tuple[str, str]],
    correctness: Mapping[tuple[str, str], float],
    all_models: Sequence[str],
) -> RouterRegret:
    """Compute router regret from a completed run.

    Args:
        router_name: Label for the results table.
        chosen: ``(task_id, model_id)`` the router actually picked.
        correctness: ``{(task_id, model_id): score}`` for every pair evaluated.
        all_models: Model ids the oracle may choose from.

    Returns:
        A :class:`RouterRegret`.

    Raises:
        ValueError: When ``chosen`` is empty, since regret would be undefined.
    """
    if not chosen:
        raise ValueError("Cannot compute router regret from zero routed items.")
    router_total = 0.0
    oracle_total = 0.0
    for task_id, model_id in chosen:
        router_total += correctness.get((task_id, model_id), 0.0)
        oracle_total += max(
            (correctness.get((task_id, m), 0.0) for m in all_models), default=0.0
        )
    n = len(chosen)
    return RouterRegret(
        router=router_name,
        router_accuracy=router_total / n,
        oracle_accuracy=oracle_total / n,
        n_items=n,
    )
