"""C4: cost-adaptive self-consistency, and the cascade.

SLM-MUX multiplies cost by ``k`` samples times ``|M|`` models with no early
exit. On CPU that is brutal, and it is the optimisation most likely to produce a
clean, quotable result (``BUILD_BRIEF.md`` C4).

Two mechanisms:

* :class:`AdaptiveSelfConsistency` -- draw samples **one at a time** and stop as
  soon as the posterior on the modal answer crosses a threshold. Identical
  answers early means the remaining samples are near-certain to agree, so
  drawing them buys nothing.
* :class:`CascadeStrategy` -- try the cheapest model first and escalate only
  when its confidence is low. The expensive model is then paid for on the
  fraction of queries that need it, not on all of them.

Both report the cost they saved (``samples_drawn`` versus ``k``), so the
Phase 11 before/after number is measured from the trace, not asserted.

Stopping rule. With ``m`` of ``n`` samples agreeing, the posterior probability
that the modal answer is the true majority is estimated with a Beta-Binomial
posterior under a uniform prior: ``p ~ Beta(m + 1, n - m + 1)``. Stop when
``P(p > 0.5)`` exceeds ``posterior_threshold``. That is a well-defined
criterion, cheap to evaluate, and it degrades gracefully at small ``n`` where a
normal approximation would stop far too early.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from slmflow.reliability.calibration import regularised_incomplete_beta
from slmflow.reliability.confidence import SelfConsistency, modal_answer
from slmflow.strategies.base import BaseStrategy, GenerationBatch, StrategyContext
from slmflow.telemetry.trace import get_logger
from slmflow.types import ModelCard, SubTask

__all__ = [
    "AdaptiveSelfConsistency",
    "CascadeStrategy",
    "SingleStrategy",
    "posterior_majority",
    "SamplingBudgetReport",
]

_LOG = get_logger("strategy.adaptive")


def posterior_majority(agree: int, total: int) -> float:
    """``P(true majority | agree of total agreed)`` under a uniform prior.

    Beta-Binomial: with a ``Beta(1, 1)`` prior the posterior is
    ``Beta(agree + 1, total - agree + 1)`` and the quantity wanted is
    ``P(p > 0.5) = 1 - I_0.5(agree + 1, total - agree + 1)``.

    Args:
        agree: Samples agreeing with the modal answer.
        total: Samples drawn.

    Returns:
        Posterior probability in ``[0, 1]``. ``0.5`` for zero samples, since
        nothing has been observed.

    Example:
        >>> posterior_majority(3, 3) > posterior_majority(2, 3)
        True
    """
    if total <= 0:
        return 0.5
    agree = max(0, min(agree, total))
    return 1.0 - regularised_incomplete_beta(0.5, agree + 1, total - agree + 1)


@dataclass(slots=True)
class SamplingBudgetReport:
    """How many samples were drawn versus how many were allowed.

    The C4 cost number. Recorded per step so the harness can report mean
    samples saved at matched accuracy.
    """

    max_samples: int
    drawn: int
    stopped_early: bool
    final_posterior: float
    per_model_drawn: dict[str, int] = field(default_factory=dict)

    @property
    def saved(self) -> int:
        """Samples not drawn."""
        return max(0, self.max_samples - self.drawn)

    @property
    def saved_fraction(self) -> float | None:
        """Fraction of the budget left unspent."""
        if self.max_samples <= 0:
            return None
        return self.saved / self.max_samples

    def as_row(self) -> dict[str, object]:
        """Flat dict for a results table."""
        return {
            "max_samples": self.max_samples,
            "drawn": self.drawn,
            "saved": self.saved,
            "saved_fraction": (
                None if self.saved_fraction is None else round(self.saved_fraction, 4)
            ),
            "stopped_early": self.stopped_early,
            "final_posterior": round(self.final_posterior, 4),
        }


class SingleStrategy(BaseStrategy):
    """One model, one greedy sample. The trivial baseline.

    Every cost claim is relative to this, so it must exist and be measured, not
    assumed to be the obvious floor.
    """

    name = "single"

    def __init__(self, *, model_id: str | None = None, use_grammar: bool = True) -> None:
        """Args: model_id: pin a model; otherwise the first candidate is used."""
        super().__init__(confidence=SelfConsistency(), use_grammar=use_grammar, temperature=0.0)
        self.model_id = model_id

    def execute(self, subtask: SubTask, context: StrategyContext) -> GenerationBatch:
        """Draw exactly one generation."""
        candidates = self.candidates(subtask, context)
        if self.model_id:
            candidates = [c for c in candidates if c.model_id == self.model_id]
        if not candidates:
            raise ValueError(
                f"SingleStrategy has no candidate model for step {subtask.name!r}"
                + (f" (wanted {self.model_id!r})" if self.model_id else "")
                + "."
            )
        return self.sample_once(subtask, candidates[0], context)


class AdaptiveSelfConsistency(BaseStrategy):
    """Sequential self-consistency with early stopping. The C4 mechanism.

    Draws one sample at a time from one model and stops as soon as the
    Beta-Binomial posterior on the modal answer exceeds
    ``posterior_threshold``, or the sample budget ``k`` is exhausted.

    On easy items the first two or three samples agree and it stops there; on
    hard items it spends the full budget. That is the point: cost tracks
    difficulty instead of being constant.

    Args:
        k: Maximum samples (matches SLM-MUX's ``k`` for a fair comparison).
        temperature: Sampling temperature, must be > 0.
        posterior_threshold: Stop when the posterior clears this.
        min_samples: Never stop before this many samples, however unanimous.
            Two identical samples are weak evidence; this prevents the
            degenerate "always stop at 1" behaviour.
    """

    name = "adaptive_self_consistency"

    def __init__(
        self,
        *,
        k: int = 5,
        temperature: float = 0.7,
        posterior_threshold: float = 0.95,
        min_samples: int = 2,
        model_id: str | None = None,
        use_grammar: bool = True,
    ) -> None:
        """Configure the stopping rule."""
        if temperature <= 0.0:
            raise ValueError("Adaptive self-consistency needs temperature > 0.")
        if not 0.5 < posterior_threshold < 1.0:
            raise ValueError(
                f"posterior_threshold must be in (0.5, 1.0), got {posterior_threshold}. "
                f"Below 0.5 the rule would stop on a minority answer."
            )
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        super().__init__(
            confidence=SelfConsistency(), use_grammar=use_grammar, temperature=temperature
        )
        self.k = k
        self.posterior_threshold = posterior_threshold
        self.min_samples = min_samples
        self.model_id = model_id
        self.last_report: SamplingBudgetReport | None = None

    def execute(self, subtask: SubTask, context: StrategyContext) -> GenerationBatch:
        """Draw samples until the posterior is satisfied or the budget runs out."""
        candidates = self.candidates(subtask, context)
        if self.model_id:
            candidates = [c for c in candidates if c.model_id == self.model_id]
        if not candidates:
            raise ValueError(
                f"AdaptiveSelfConsistency has no candidate for step {subtask.name!r}."
            )
        card = candidates[0]

        batch = GenerationBatch()
        posterior = 0.5
        for index in range(self.k):
            batch.extend(self.sample_once(subtask, card, context, seed_offset=index))
            _, agree, total = modal_answer(batch.generations)
            posterior = posterior_majority(agree, total)
            if total >= self.min_samples and posterior >= self.posterior_threshold:
                break

        self.last_report = SamplingBudgetReport(
            max_samples=self.k,
            drawn=len(batch.generations),
            stopped_early=len(batch.generations) < self.k,
            final_posterior=posterior,
            per_model_drawn={card.model_id: len(batch.generations)},
        )
        _LOG.info(
            "adaptive sampling finished",
            extra={
                "event": "adaptive_sampling",
                "slmflow": {"step": subtask.name, **self.last_report.as_row()},
            },
        )
        return batch


class CascadeStrategy(BaseStrategy):
    """Cheapest model first; escalate only when confidence is low.

    The routing policy form of C4. The escalation ladder is cheapest-first by
    measured resident bytes, so the expensive model is paid for only on the
    queries that need it.

    Escalation is where the two-tier budget bites: a 7B model may be
    GPU-resident while a 0.5B model sits in RAM, so escalating is not merely
    slower, it may force an eviction. The trace records the swap so that cost
    appears in the results rather than vanishing.

    Args:
        threshold: Confidence below which the strategy escalates.
        k: Samples per rung (1 keeps the first rung genuinely cheap).
        temperature: Sampling temperature; ``> 0`` needed if ``k > 1``.
        ladder: Explicit model order. ``None`` sorts candidates cheapest-first.
        max_rungs: Stop escalating after this many models.
    """

    name = "cascade"

    def __init__(
        self,
        *,
        threshold: float = 0.7,
        k: int = 1,
        temperature: float = 0.0,
        ladder: Sequence[str] | None = None,
        max_rungs: int = 3,
        use_grammar: bool = True,
    ) -> None:
        """Configure the ladder."""
        if k > 1 and temperature <= 0.0:
            raise ValueError(
                "A cascade with k > 1 needs temperature > 0, or every sample on a rung "
                "is identical and the confidence signal is vacuous."
            )
        super().__init__(
            confidence=SelfConsistency(), use_grammar=use_grammar, temperature=temperature
        )
        self.threshold = threshold
        self.k = k
        self.ladder = tuple(ladder) if ladder else None
        self.max_rungs = max_rungs
        self.escalations = 0
        self.last_rung: str | None = None

    def _rungs(self, subtask: SubTask, context: StrategyContext) -> list[ModelCard]:
        """Resolve the escalation ladder, cheapest first."""
        candidates = self.candidates(subtask, context)
        if self.ladder:
            index = {model_id: i for i, model_id in enumerate(self.ladder)}
            chosen = [c for c in candidates if c.model_id in index]
            chosen.sort(key=lambda c: index[c.model_id])
            return chosen[: self.max_rungs]
        candidates.sort(
            key=lambda c: (
                c.footprint.resident_bytes == 0,
                c.footprint.resident_bytes,
                -(c.accuracy_for(subtask.spec.task_type) or 0.0),
            )
        )
        return candidates[: self.max_rungs]

    def execute(self, subtask: SubTask, context: StrategyContext) -> GenerationBatch:
        """Walk the ladder, stopping at the first confident rung."""
        rungs = self._rungs(subtask, context)
        if not rungs:
            raise ValueError(f"Cascade has no rungs for step {subtask.name!r}.")

        ctx = context.as_mapping()
        accumulated = GenerationBatch()
        for rung_index, card in enumerate(rungs):
            rung_batch = GenerationBatch()
            for sample_index in range(self.k):
                rung_batch.extend(
                    self.sample_once(subtask, card, context, seed_offset=sample_index)
                )
            accumulated.extend(rung_batch)
            self.last_rung = card.model_id

            confidence = float(self.confidence.score(rung_batch.generations, ctx))  # type: ignore[attr-defined]
            if confidence >= self.threshold or rung_index == len(rungs) - 1:
                if rung_index:
                    self.escalations += rung_index
                _LOG.info(
                    "cascade settled",
                    extra={
                        "event": "cascade",
                        "slmflow": {
                            "step": subtask.name,
                            "rung": card.model_id,
                            "rung_index": rung_index,
                            "confidence": round(confidence, 4),
                            "escalated": rung_index > 0,
                        },
                    },
                )
                # Report only the winning rung's generations, so the answer is
                # not diluted by the rejected cheaper attempt. Cost still
                # counts every sample drawn.
                return GenerationBatch(
                    generations=rung_batch.generations,
                    swaps=accumulated.swaps,
                    swap_seconds=accumulated.swap_seconds,
                    peak_vram_bytes=accumulated.peak_vram_bytes,
                    peak_ram_bytes=accumulated.peak_ram_bytes,
                )

        return accumulated  # pragma: no cover - the loop always returns
