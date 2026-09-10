"""A learned router that runs on a phone: pure-Python logistic regression.

``BUILD_BRIEF.md`` Phase 7: "a lightweight learned router (logistic regression /
gradient boosting over cheap query features) trained on your own Phase 2 data --
no neural router, it must run on T0."

So this is one-vs-rest multinomial logistic regression, trained by batch
gradient descent with L2 regularisation, implemented in stdlib ``math``. No
numpy anywhere in the import path. Inference is a dot product per model: tens
of microseconds, which is the only budget a router can justify when the
decision it makes costs seconds.

Trained weights serialise to JSON so the router that ships is the router that
was measured, and the training data provenance travels with it.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slmflow.errors import ConfigError, ModelNotFound
from slmflow.routing.features import FEATURE_NAMES, feature_vector, standardise
from slmflow.telemetry.trace import get_logger
from slmflow.types import ModelCard, SubTask

__all__ = ["LearnedRouter", "TrainingExample", "TrainReport", "train_router"]

_LOG = get_logger("routing.learned")


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One routing decision with its observed outcome.

    Produced by replaying Phase 2 profiling data: for each task, which models
    were tried and which were correct. ``reward`` is the label; use accuracy,
    or accuracy minus a cost term to train a cost-aware router.
    """

    features: tuple[float, ...]
    model_id: str
    reward: float


@dataclass(slots=True)
class TrainReport:
    """What training actually did. Recorded next to the weights."""

    n_examples: int
    n_models: int
    epochs: int
    final_loss: float
    train_accuracy: float
    holdout_accuracy: float | None
    feature_names: tuple[str, ...] = FEATURE_NAMES
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        """Flat dict for a results table."""
        return {
            "n_examples": self.n_examples,
            "n_models": self.n_models,
            "epochs": self.epochs,
            "final_loss": round(self.final_loss, 5),
            "train_accuracy": round(self.train_accuracy, 4),
            "holdout_accuracy": (
                None if self.holdout_accuracy is None else round(self.holdout_accuracy, 4)
            ),
        }


def _sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
    """Dot product."""
    return sum(w * x for w, x in zip(weights, features, strict=True))


class LearnedRouter:
    """One-vs-rest logistic regression over cheap query features.

    Implements :class:`~slmflow.types.Router`. Falls back to the smallest
    candidate when it is asked about a model it was never trained on, and logs
    that rather than silently guessing.

    Args:
        weights: ``{model_id: weight_vector}`` in :data:`FEATURE_NAMES` order.
        means: Per-feature standardisation means.
        stds: Per-feature standardisation standard deviations.
        warm_bonus: Added to the predicted score of an already-resident model.
            This keeps the learned router resource-aware; without it a learned
            router optimises quality alone and pays swap cost it cannot see.
        report: The training report, carried for provenance.
    """

    name = "learned_logreg"

    def __init__(
        self,
        weights: Mapping[str, Sequence[float]],
        means: Sequence[float],
        stds: Sequence[float],
        *,
        warm_bonus: float = 0.05,
        report: TrainReport | None = None,
    ) -> None:
        """Store weights and standardisation parameters."""
        width = len(FEATURE_NAMES)
        for model_id, vector in weights.items():
            if len(vector) != width:
                raise ConfigError(
                    f"Weight vector for {model_id!r} has {len(vector)} entries but "
                    f"FEATURE_NAMES has {width}. The model was trained against a different "
                    f"feature set; retrain it."
                )
        self.weights = {k: list(v) for k, v in weights.items()}
        self.means = list(means)
        self.stds = list(stds)
        self.warm_bonus = warm_bonus
        self.report = report

    # -- inference --------------------------------------------------------- #
    def _standardise(self, raw: Sequence[float]) -> list[float]:
        """Apply the training-time standardisation."""
        return [
            (x - m) / (s if s else 1.0)
            for x, m, s in zip(raw, self.means, self.stds, strict=True)
        ]

    def predict_scores(
        self, subtask: SubTask, context: Mapping[str, object] | None = None
    ) -> dict[str, float]:
        """Predicted success probability per known model."""
        features = self._standardise(feature_vector(subtask, context))
        return {
            model_id: _sigmoid(_dot(vector, features))
            for model_id, vector in self.weights.items()
        }

    def select(
        self,
        subtask: SubTask,
        candidates: Sequence[ModelCard],
        *,
        resident: Sequence[str] = (),
    ) -> ModelCard:
        """Pick the candidate with the highest predicted score plus warm bonus."""
        if not candidates:
            raise ModelNotFound("<any>", list(self.weights))
        if subtask.pinned_model:
            for card in candidates:
                if card.model_id == subtask.pinned_model:
                    return card
            raise ModelNotFound(subtask.pinned_model, [c.model_id for c in candidates])

        scores = self.predict_scores(subtask)
        resident_set = set(resident)
        unknown = [c.model_id for c in candidates if c.model_id not in scores]
        if unknown:
            _LOG.warning(
                "learned router has no weights for some candidates",
                extra={
                    "event": "router_unknown_models",
                    "slmflow": {"step": subtask.name, "unknown": unknown},
                },
            )

        def score(card: ModelCard) -> float:
            base = scores.get(card.model_id)
            if base is None:
                # Never invent a score for an untrained model: rank it below
                # every trained one, breaking ties by smaller footprint.
                return -1.0 - card.footprint.resident_bytes / 1e12
            return base + (self.warm_bonus if card.model_id in resident_set else 0.0)

        return max(candidates, key=score)

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """Serialise weights, standardisation and provenance."""
        return {
            "kind": "logreg_ovr",
            "feature_names": list(FEATURE_NAMES),
            "means": self.means,
            "stds": self.stds,
            "warm_bonus": self.warm_bonus,
            "weights": self.weights,
            "report": self.report.as_row() if self.report else None,
        }

    def save(self, path: str | Path) -> None:
        """Write the router to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LearnedRouter:
        """Load a trained router.

        Raises:
            ConfigError: When the file is missing, malformed, or was trained
                against a different feature set.
        """
        p = Path(path)
        if not p.exists():
            raise ConfigError(
                f"No trained router at {p}. Train one with "
                f"`python -m benchmarks.train_router`.",
                path=str(p),
            )
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{p} is not valid JSON: {exc}", path=str(p)) from exc

        stored = list(payload.get("feature_names") or [])
        if stored != list(FEATURE_NAMES):
            raise ConfigError(
                f"{p} was trained on a different feature set. Stored {len(stored)} features, "
                f"code defines {len(FEATURE_NAMES)}. Retrain the router.",
                path=str(p),
            )
        return cls(
            weights=payload["weights"],
            means=payload["means"],
            stds=payload["stds"],
            warm_bonus=float(payload.get("warm_bonus", 0.05)),
        )


def train_router(
    examples: Sequence[TrainingExample],
    *,
    epochs: int = 300,
    learning_rate: float = 0.3,
    l2: float = 1e-3,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    warm_bonus: float = 0.05,
) -> LearnedRouter:
    """Train a one-vs-rest logistic-regression router.

    One binary classifier per model answers "would this model succeed on this
    query?". At inference the highest probability wins. This is deliberately
    simpler than a ranker: it trains on the sparse, per-model outcome data that
    Phase 2 profiling actually produces.

    Args:
        examples: Observed routing outcomes.
        epochs: Batch gradient-descent iterations.
        learning_rate: Step size.
        l2: L2 penalty, excluding the bias term.
        holdout_fraction: Fraction held out to report honest accuracy. Set to
            0 to train on everything (and then report no holdout number).
        seed: Shuffle seed (Ground Rule 11).
        warm_bonus: Resource-awareness term for the returned router.

    Returns:
        A trained :class:`LearnedRouter` carrying its :class:`TrainReport`.

    Raises:
        ValueError: When there are too few examples, or only one model, to
            train anything meaningful.
    """
    if len(examples) < 4:
        raise ValueError(
            f"Need at least 4 training examples, got {len(examples)}. "
            f"Run more Phase 2 profiling before training a router."
        )
    model_ids = sorted({e.model_id for e in examples})
    if len(model_ids) < 2:
        raise ValueError(
            f"Routing requires at least 2 models; the training data only covers "
            f"{model_ids}. A single-model 'router' is StaticRouter."
        )

    rows = [list(e.features) for e in examples]
    means, stds = standardise(rows)
    scaled = [
        [(x - m) / (s if s else 1.0) for x, m, s in zip(row, means, stds, strict=True)]
        for row in rows
    ]

    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    n_holdout = int(len(order) * holdout_fraction)
    holdout_idx = set(order[:n_holdout])
    train_idx = [i for i in order if i not in holdout_idx]

    width = len(FEATURE_NAMES)
    bias_index = FEATURE_NAMES.index("bias")
    weights: dict[str, list[float]] = {m: [0.0] * width for m in model_ids}
    final_loss = 0.0

    for model_id in model_ids:
        subset = [i for i in train_idx if examples[i].model_id == model_id]
        if not subset:
            continue
        w = weights[model_id]
        for _ in range(epochs):
            gradient = [0.0] * width
            loss = 0.0
            for i in subset:
                x = scaled[i]
                y = max(0.0, min(1.0, examples[i].reward))
                p = _sigmoid(_dot(w, x))
                error = p - y
                for j in range(width):
                    gradient[j] += error * x[j]
                loss += -(
                    y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12))
                )
            n = len(subset)
            for j in range(width):
                reg = 0.0 if j == bias_index else l2 * w[j]
                w[j] -= learning_rate * (gradient[j] / n + reg)
            final_loss = loss / n
        weights[model_id] = w

    router = LearnedRouter(weights, means, stds, warm_bonus=warm_bonus)

    def accuracy(indices: Sequence[int]) -> float | None:
        """Fraction of held-out decisions where the argmax model was rewarded."""
        by_task: dict[tuple[float, ...], list[int]] = {}
        for i in indices:
            by_task.setdefault(tuple(examples[i].features), []).append(i)
        hits = 0
        total = 0
        for group in by_task.values():
            if len(group) < 2:
                continue
            x = scaled[group[0]]
            predicted = max(
                (examples[i].model_id for i in group),
                key=lambda m: _sigmoid(_dot(weights[m], x)),
            )
            best_reward = max(examples[i].reward for i in group)
            picked = next(examples[i].reward for i in group if examples[i].model_id == predicted)
            hits += 1 if picked >= best_reward else 0
            total += 1
        return hits / total if total else None

    router.report = TrainReport(
        n_examples=len(examples),
        n_models=len(model_ids),
        epochs=epochs,
        final_loss=final_loss,
        train_accuracy=accuracy(train_idx) or 0.0,
        holdout_accuracy=accuracy(sorted(holdout_idx)) if holdout_idx else None,
        notes=(
            "one-vs-rest logistic regression, pure Python; accuracy is the fraction of "
            "multi-model tasks where the argmax model achieved the best observed reward"
        ),
    )
    _LOG.info(
        "router trained",
        extra={"event": "router_trained", "slmflow": router.report.as_row()},
    )
    return router


@dataclass(slots=True)
class ExampleBuilder:
    """Accumulates training examples from evaluated runs.

    Usage: for each benchmark item, call :meth:`observe` once per model tried,
    then :meth:`build`.
    """

    examples: list[TrainingExample] = field(default_factory=list)

    def observe(self, subtask: SubTask, model_id: str, reward: float) -> None:
        """Record one (task, model, outcome) triple."""
        self.examples.append(
            TrainingExample(
                features=tuple(feature_vector(subtask)),
                model_id=model_id,
                reward=reward,
            )
        )

    def build(self) -> list[TrainingExample]:
        """Return the accumulated examples."""
        return list(self.examples)
