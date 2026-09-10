"""Abstention, verification and fallback.

Phase 10 assembles these into the C3 contract:

* :class:`AbstentionPolicy` -- decide whether to answer, given a confidence and
  a (possibly auto-tuned) threshold.
* :class:`TinyVerifier` -- a 0.5B model checking a 3B generator's output, the
  pattern from ``context.md`` section D. Selective: only low-confidence claims
  are verified, so the verifier's cost is paid where it can change the outcome.
* :class:`FallbackChain` -- re-route on resource error or failure.

Design note. Abstention is *not* an error. It is a first-class outcome carried
on :class:`~slmflow.types.Result`, and only :meth:`Result.unwrap` turns it into
an exception. A framework that raised on abstention would push every caller
into try/except and make the safe path the inconvenient one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from slmflow.errors import BackendUnavailable, ResourceExhausted
from slmflow.reliability.calibration import ThresholdPolicy
from slmflow.reliability.confidence import contradiction_rate, normalise_answer
from slmflow.telemetry.trace import get_logger
from slmflow.types import AbstentionReason, Generation, ModelCard, StepTrace, SubTask

__all__ = [
    "AbstentionDecision",
    "AbstentionPolicy",
    "TinyVerifier",
    "FallbackChain",
    "VerificationResult",
]

_LOG = get_logger("reliability.abstain")


@dataclass(frozen=True, slots=True)
class AbstentionDecision:
    """Whether to answer, and why not."""

    answer: bool
    confidence: float
    threshold: float
    reason: AbstentionReason | None = None
    detail: str = ""

    @property
    def abstained(self) -> bool:
        """True when the policy declined."""
        return not self.answer


@dataclass(slots=True)
class AbstentionPolicy:
    """Decides whether a step's output may be returned.

    Three independent grounds for declining, checked in order:

    1. **Schema failure.** Output that does not parse is never an answer.
    2. **Contradiction.** Models disagreeing above ``max_contradiction`` means
       the ensemble has no consensus to report, even if one member is
       internally confident.
    3. **Low confidence.** Below the threshold, which is either declared by the
       caller or tuned by :func:`~slmflow.reliability.calibration.tune_threshold`.

    Args:
        threshold: Static confidence floor. Ignored when ``policy`` is set.
        policy: A tuned :class:`ThresholdPolicy` from the C3 calibration step.
            Takes precedence, because a measured threshold beats a guessed one.
        max_contradiction: Cross-model disagreement above which to abstain.
        require_retrieval_support: Minimum groundedness on the RAG path.
    """

    threshold: float = 0.5
    policy: ThresholdPolicy | None = None
    max_contradiction: float = 1.0
    require_retrieval_support: float | None = None

    @property
    def effective_threshold(self) -> float:
        """The threshold actually applied."""
        return self.policy.threshold if self.policy else self.threshold

    def decide(
        self,
        confidence: float,
        *,
        generations: Sequence[Generation] = (),
        signals: Mapping[str, float] | None = None,
        schema_valid: bool | None = None,
    ) -> AbstentionDecision:
        """Apply the policy.

        Example:
            >>> AbstentionPolicy(threshold=0.8).decide(0.6).abstained
            True
        """
        threshold = self.effective_threshold

        if schema_valid is False:
            return AbstentionDecision(
                answer=False,
                confidence=confidence,
                threshold=threshold,
                reason=AbstentionReason.SCHEMA_FAILURE,
                detail="output did not satisfy the declared schema",
            )

        if generations and self.max_contradiction < 1.0:
            rate = contradiction_rate(generations)
            if rate > self.max_contradiction:
                return AbstentionDecision(
                    answer=False,
                    confidence=confidence,
                    threshold=threshold,
                    reason=AbstentionReason.CONTRADICTION,
                    detail=f"cross-model contradiction {rate:.2f} > {self.max_contradiction:.2f}",
                )

        if self.require_retrieval_support is not None:
            support = (signals or {}).get("retrieval_support", 0.0)
            if support < self.require_retrieval_support:
                return AbstentionDecision(
                    answer=False,
                    confidence=confidence,
                    threshold=threshold,
                    reason=AbstentionReason.NO_RETRIEVAL_SUPPORT,
                    detail=(
                        f"groundedness {support:.2f} < "
                        f"{self.require_retrieval_support:.2f}"
                    ),
                )

        if confidence < threshold:
            return AbstentionDecision(
                answer=False,
                confidence=confidence,
                threshold=threshold,
                reason=AbstentionReason.LOW_CONFIDENCE,
                detail=f"confidence {confidence:.3f} < threshold {threshold:.3f}",
            )

        return AbstentionDecision(answer=True, confidence=confidence, threshold=threshold)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of a verifier pass over one candidate answer."""

    accepted: bool
    verifier_model: str
    raw: str
    cost_seconds: float
    was_selective_skip: bool = False

    def as_row(self) -> dict[str, object]:
        """Flat dict for a results table."""
        return {
            "accepted": self.accepted,
            "verifier": self.verifier_model,
            "cost_s": round(self.cost_seconds, 4),
            "skipped": self.was_selective_skip,
        }


class TinyVerifier:
    """A small model checking a larger model's output.

    ``context.md`` section D: a 0.5B verifier checking a 3B generator. The bet
    is that verification is easier than generation, so a model too weak to
    answer can still catch a wrong answer.

    **Selective** by construction: :meth:`verify` returns immediately when
    confidence already exceeds ``skip_above``. Verifying a confident answer
    costs a model load and buys almost nothing, and on this hardware that load
    may evict the generator.

    Args:
        verifier_model: Model id of the verifier.
        skip_above: Confidence above which verification is skipped.
        prompt_template: Must contain ``{answer}`` and may contain
            ``{question}`` and ``{evidence}``.
    """

    name = "tiny_verifier"

    DEFAULT_TEMPLATE = (
        "{evidence}"
        "Question: {question}\n"
        "Proposed answer: {answer}\n\n"
        "Is the proposed answer supported? Reply with exactly one word, "
        "YES or NO.\n"
    )

    def __init__(
        self,
        verifier_model: str,
        *,
        skip_above: float = 0.9,
        prompt_template: str | None = None,
    ) -> None:
        """Configure the verifier."""
        self.verifier_model = verifier_model
        self.skip_above = skip_above
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE

    def verify(
        self,
        answer: str,
        subtask: SubTask,
        context: object,
        *,
        confidence: float,
    ) -> VerificationResult:
        """Check one answer, skipping when confidence is already high.

        Args:
            answer: The candidate answer.
            subtask: The sub-task, for the question text.
            context: A :class:`~slmflow.strategies.base.StrategyContext`.
            confidence: Confidence from the generating step.

        Returns:
            A :class:`VerificationResult`. ``accepted`` is ``True`` on a skip,
            since a skip is not a rejection.
        """
        if confidence >= self.skip_above:
            return VerificationResult(
                accepted=True,
                verifier_model=self.verifier_model,
                raw="",
                cost_seconds=0.0,
                was_selective_skip=True,
            )

        manager = getattr(context, "manager", None)
        pool: Sequence[ModelCard] = getattr(context, "pool", ())
        card = next((c for c in pool if c.model_id == self.verifier_model), None)
        if manager is None or card is None:
            _LOG.warning(
                "verifier model unavailable; accepting without verification",
                extra={
                    "event": "verifier_missing",
                    "slmflow": {"verifier": self.verifier_model},
                },
            )
            return VerificationResult(
                accepted=True,
                verifier_model=self.verifier_model,
                raw="",
                cost_seconds=0.0,
                was_selective_skip=True,
            )

        evidence = getattr(context, "retrieved", ())
        prompt = self.prompt_template.format(
            question=subtask.spec.prompt,
            answer=answer,
            evidence=(
                "Evidence:\n" + "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(evidence)) + "\n\n"
                if evidence
                else ""
            ),
        )

        with manager.resident(card) as handle:
            generation = handle.backend.generate(
                card, prompt, max_tokens=8, temperature=0.0, seed=None
            )
        verdict = normalise_answer(generation.text)
        accepted = verdict.startswith("yes")
        _LOG.info(
            "verifier ruled",
            extra={
                "event": "verify",
                "slmflow": {
                    "verifier": self.verifier_model,
                    "accepted": accepted,
                    "raw": generation.text[:40],
                    "cost_s": round(generation.total_s, 4),
                },
            },
        )
        return VerificationResult(
            accepted=accepted,
            verifier_model=self.verifier_model,
            raw=generation.text,
            cost_seconds=generation.total_s,
        )


@dataclass(slots=True)
class FallbackChain:
    """Retry a failed step on alternate models.

    Recovers from :class:`ResourceExhausted` and :class:`BackendUnavailable` --
    an out-of-memory error on a laptop is an ordinary operational event, not a
    bug, so the framework treats it as routable rather than fatal.

    Args:
        alternates: Model ids to try, in order.
        on_schema_failure: Also retry when output fails schema validation.
    """

    alternates: tuple[str, ...] = ()
    on_schema_failure: bool = True
    attempts: list[tuple[str, str]] = field(default_factory=list)

    def recoverable(self, exc: BaseException) -> bool:
        """True when this exception is worth retrying elsewhere."""
        return isinstance(exc, (ResourceExhausted, BackendUnavailable))

    def next_model(self, tried: Sequence[str]) -> str | None:
        """The next untried alternate, or ``None`` when exhausted."""
        for model_id in self.alternates:
            if model_id not in tried:
                return model_id
        return None

    def record(self, model_id: str, outcome: str) -> None:
        """Note an attempt, for the trace."""
        self.attempts.append((model_id, outcome))

    def should_retry(self, trace: StepTrace) -> bool:
        """True when a completed step should be retried on another model."""
        if trace.error and self.alternates:
            return True
        return bool(
            self.on_schema_failure
            and trace.abstained
            and trace.abstention_reason is AbstentionReason.SCHEMA_FAILURE
            and self.alternates
        )
