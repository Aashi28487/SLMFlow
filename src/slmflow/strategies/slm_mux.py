"""SLM-MUX, implemented faithfully as one strategy among several.

``BUILD_BRIEF.md`` Phase 6: independent generation at ``T > 0``,
self-consistency confidence, validation-accuracy tie-break, and the
``UnionAcc - lambda * Contradiction`` model-selection search.

Two things to be clear about in the writeup:

* This is a **re-implementation from the method description**, with different
  models on different hardware, so numbers will differ from the paper. Phase 6's
  DoD is reproducing the *qualitative* pattern -- confidence-based selection
  beating naive majority voting when models disagree -- and stating plainly if
  it does not reproduce.
* SLM-MUX is a **fixed static ensemble**. Its cost is ``k * |M|`` generations
  per query with no early exit, which is what :mod:`slmflow.strategies.adaptive`
  attacks (C4). Keeping the faithful version here is what makes that comparison
  meaningful.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from slmflow.reliability.confidence import (
    SelfConsistency,
    contradiction_rate,
    modal_answer,
    normalise_answer,
)
from slmflow.strategies.base import BaseStrategy, GenerationBatch, StrategyContext
from slmflow.telemetry.trace import get_logger
from slmflow.types import Generation, SubTask

__all__ = ["SLMMuxStrategy", "AgentForestStrategy", "select_model_set", "SelectionScore"]

_LOG = get_logger("strategy.slm_mux")


class SLMMuxStrategy(BaseStrategy):
    """Independent sampling per model, then confidence-based selection.

    Algorithm, per the paper's description:

    1. Draw ``k`` samples from each of ``|M|`` models independently at ``T > 0``.
    2. Per model, compute the modal answer and its self-consistency share.
    3. Return the answer of the model with the highest self-consistency,
       breaking ties by that model's validation accuracy on this task type.

    Step 3 is what distinguishes it from flat majority voting
    (:class:`AgentForestStrategy`): a single confident model outranks a larger
    number of mutually inconsistent ones.

    Args:
        k: Samples per model.
        temperature: Sampling temperature. Must be > 0 or every sample is
            identical and self-consistency is vacuously 1.0.
        models: Restrict to these model ids. ``None`` uses the sub-task's
            candidates, which is what the config-driven path supplies.
    """

    name = "slm_mux"

    def __init__(
        self,
        *,
        k: int = 5,
        temperature: float = 0.7,
        models: Sequence[str] | None = None,
        use_grammar: bool = True,
    ) -> None:
        """Configure the ensemble."""
        if temperature <= 0.0:
            raise ValueError(
                "SLM-MUX needs temperature > 0; at T=0 every sample is identical and "
                "self-consistency is always 1.0, which makes the confidence signal useless."
            )
        if k < 2:
            raise ValueError(f"SLM-MUX needs k >= 2 samples per model, got {k}.")
        super().__init__(
            confidence=SelfConsistency(), use_grammar=use_grammar, temperature=temperature
        )
        self.k = k
        self.models = tuple(models) if models else None

    def execute(self, subtask: SubTask, context: StrategyContext) -> GenerationBatch:
        """Draw ``k`` samples from each model, grouped so each loads once.

        The grouping matters: sampling round-robin across models would pay
        ``k * |M|`` model loads instead of ``|M|``. On this hardware that is the
        difference between an expensive strategy and an unusable one, and it is
        an optimisation no cloud implementation would need.
        """
        candidates = self.candidates(subtask, context)
        if self.models:
            allowed = set(self.models)
            candidates = [c for c in candidates if c.model_id in allowed]
        if not candidates:
            raise ValueError(
                f"SLM-MUX has no candidate models for step {subtask.name!r}. "
                f"Check candidate_models against the registry and the memory budget."
            )

        batch = GenerationBatch()
        for card in candidates:
            for sample_index in range(self.k):
                batch.extend(
                    self.sample_once(
                        subtask,
                        card,
                        context,
                        seed_offset=sample_index,
                    )
                )
        return batch

    def select(
        self, generations: Sequence[Generation], context: StrategyContext, subtask: SubTask
    ) -> tuple[str, float, dict[str, float]]:
        """Apply SLM-MUX selection to already-drawn generations.

        Returns:
            ``(answer, confidence, per_model_confidence)``.
        """
        by_model: dict[str, list[Generation]] = {}
        for gen in generations:
            by_model.setdefault(gen.model_id, []).append(gen)

        accuracy_by_model = {
            c.model_id: (c.accuracy_for(subtask.spec.task_type) or 0.0) for c in context.pool
        }

        scored: list[tuple[float, float, str, str]] = []
        per_model: dict[str, float] = {}
        for model_id, model_gens in by_model.items():
            answer, count, total = modal_answer(model_gens)
            consistency = count / total if total else 0.0
            per_model[model_id] = consistency
            scored.append((consistency, accuracy_by_model.get(model_id, 0.0), model_id, answer))

        if not scored:
            return "", 0.0, {}

        # Highest self-consistency, tie-broken by validation accuracy.
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        best = scored[0]
        _LOG.info(
            "slm_mux selection",
            extra={
                "event": "mux_select",
                "slmflow": {
                    "step": subtask.name,
                    "winner": best[2],
                    "consistency": round(best[0], 4),
                    "contradiction_rate": round(contradiction_rate(generations), 4),
                    "per_model": {k: round(v, 4) for k, v in per_model.items()},
                },
            },
        )
        # Return the original (un-normalised) text of the winning answer.
        for gen in by_model[best[2]]:
            if normalise_answer(gen.text) == best[3]:
                return gen.text, best[0], per_model
        return best[3], best[0], per_model


class AgentForestStrategy(BaseStrategy):
    """Flat majority vote over all samples from all models.

    The contrasting ensemble the brief asks for (Agent-Forest style). Every
    sample counts equally, so a model sampled ``k`` times has ``k`` votes. This
    is the baseline SLM-MUX claims to beat when models disagree, and it is
    cheap to state precisely why: it cannot distinguish one confident model
    from several inconsistent ones.
    """

    name = "agent_forest"

    def __init__(
        self, *, k: int = 5, temperature: float = 0.7, use_grammar: bool = True
    ) -> None:
        """Configure the vote."""
        if temperature <= 0.0:
            raise ValueError("Agent-Forest majority voting needs temperature > 0.")
        super().__init__(
            confidence=SelfConsistency(), use_grammar=use_grammar, temperature=temperature
        )
        self.k = k

    def execute(self, subtask: SubTask, context: StrategyContext) -> GenerationBatch:
        """Draw ``k`` samples from every candidate model."""
        candidates = self.candidates(subtask, context)
        if not candidates:
            raise ValueError(f"Agent-Forest has no candidates for step {subtask.name!r}.")
        batch = GenerationBatch()
        for card in candidates:
            for sample_index in range(self.k):
                batch.extend(
                    self.sample_once(subtask, card, context, seed_offset=sample_index)
                )
        return batch


# --------------------------------------------------------------------------- #
# Model-set selection: UnionAcc - lambda * Contradiction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SelectionScore:
    """One candidate model set and its SLM-MUX selection score."""

    models: tuple[str, ...]
    union_accuracy: float
    contradiction: float
    score: float
    peak_bytes: int = 0

    def as_row(self) -> dict[str, object]:
        """Flat dict for ``docs/MODEL_SELECTION.md``."""
        return {
            "models": ", ".join(self.models),
            "union_accuracy": round(self.union_accuracy, 4),
            "contradiction": round(self.contradiction, 4),
            "score": round(self.score, 4),
            "peak_gb": round(self.peak_bytes / 1e9, 3) if self.peak_bytes else None,
        }


def select_model_set(
    per_model_correct: Mapping[str, Sequence[bool]],
    per_model_answers: Mapping[str, Sequence[str]],
    *,
    lambda_contradiction: float = 1.0,
    max_size: int = 3,
    footprints: Mapping[str, int] | None = None,
    budget_bytes: int | None = None,
) -> list[SelectionScore]:
    """Search model sets by ``UnionAcc - lambda * Contradiction``.

    SLM-MUX's model-selection objective. ``UnionAcc`` is the fraction of items
    at least one member answers correctly -- the ensemble's ceiling.
    ``Contradiction`` is the fraction of items where members disagree, which
    penalises sets whose ceiling is only reachable by luck.

    The addition here is ``budget_bytes``: a set whose members cannot be
    co-resident is not free on this hardware even if its objective is best,
    because it forces a swap per query. Sets exceeding the budget are still
    scored and returned, flagged by ``peak_bytes``, so the cost of the paper's
    preferred set is visible rather than hidden.

    Args:
        per_model_correct: ``{model_id: [correct?, ...]}`` aligned across models.
        per_model_answers: ``{model_id: [answer, ...]}`` aligned across models.
        lambda_contradiction: Penalty weight.
        max_size: Largest set size to consider.
        footprints: ``{model_id: resident_bytes}`` for the co-residency note.
        budget_bytes: If given, sets above this are marked in ``peak_bytes``.

    Returns:
        Every candidate set, best score first.

    Raises:
        ValueError: When models have differing item counts, which would make
            union accuracy meaningless.
    """
    models = sorted(per_model_correct)
    if not models:
        return []
    lengths = {len(v) for v in per_model_correct.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"Model correctness lists must be aligned and equal length, got lengths {lengths}."
        )
    n_items = lengths.pop()
    if n_items == 0:
        return []

    results: list[SelectionScore] = []
    for size in range(1, min(max_size, len(models)) + 1):
        for combo in itertools.combinations(models, size):
            union_hits = sum(
                1 for i in range(n_items) if any(per_model_correct[m][i] for m in combo)
            )
            union_accuracy = union_hits / n_items

            if size == 1:
                contradiction = 0.0
            else:
                disagreements = 0
                for i in range(n_items):
                    answers = {normalise_answer(per_model_answers[m][i]) for m in combo}
                    if len(answers) > 1:
                        disagreements += 1
                contradiction = disagreements / n_items

            peak = sum((footprints or {}).get(m, 0) for m in combo)
            results.append(
                SelectionScore(
                    models=combo,
                    union_accuracy=union_accuracy,
                    contradiction=contradiction,
                    score=union_accuracy - lambda_contradiction * contradiction,
                    peak_bytes=peak,
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    if budget_bytes:
        over = [r for r in results if r.peak_bytes > budget_bytes]
        if over:
            _LOG.info(
                "some model sets exceed the co-residency budget",
                extra={
                    "event": "selection_over_budget",
                    "slmflow": {
                        "budget_bytes": budget_bytes,
                        "over_budget_sets": [", ".join(r.models) for r in over[:5]],
                    },
                },
            )
    return results
