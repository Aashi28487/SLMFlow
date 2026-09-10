"""Confidence signals beyond bare self-consistency.

SLM-MUX states its own weakness plainly: confidence is self-consistency alone.
``BUILD_BRIEF.md`` Phase 10 attacks exactly that, so this module implements four
independent signals and a combiner:

* :class:`SelfConsistency` -- modal-answer share across ``k`` samples. The
  SLM-MUX signal, kept as the baseline to beat.
* :class:`LogProbMargin` -- mean top-1 minus top-2 log-prob. Costs **one**
  sample, so it is the only signal that is free on CPU.
* :class:`CrossModelAgreement` -- do independent models concur? Its inverse is
  SLM-MUX's contradiction rate.
* :class:`RetrievalSupport` -- are the answer's claims present in retrieved
  evidence? The groundedness signal for the RAG path.

Each implements :class:`~slmflow.types.ReliabilityCheck`. Phase 10 measures ECE
per signal, so they must stay independently addressable rather than being fused
into one opaque score.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from slmflow.types import Generation

__all__ = [
    "normalise_answer",
    "SelfConsistency",
    "LogProbMargin",
    "CrossModelAgreement",
    "RetrievalSupport",
    "CombinedConfidence",
    "answer_distribution",
    "contradiction_rate",
]

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_ARTICLES = {"a", "an", "the"}


def normalise_answer(text: str) -> str:
    """Canonicalise an answer for exact-match comparison.

    Lowercases, strips punctuation and articles, collapses whitespace. Applied
    consistently to predictions and gold labels so agreement measures agreement
    about content rather than about formatting.

    Example:
        >>> normalise_answer("  The Investigational. ")
        'investigational'
    """
    lowered = _PUNCT.sub(" ", text.lower())
    words = [w for w in _WS.split(lowered) if w and w not in _ARTICLES]
    return " ".join(words)


def answer_distribution(generations: Sequence[Generation]) -> dict[str, int]:
    """Count normalised answers across samples."""
    counts: dict[str, int] = {}
    for gen in generations:
        key = normalise_answer(gen.text)
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def modal_answer(generations: Sequence[Generation]) -> tuple[str, int, int]:
    """Return ``(modal_answer, count, total)``. ``("", 0, 0)`` when empty."""
    counts = answer_distribution(generations)
    if not counts:
        return "", 0, 0
    best = max(sorted(counts.items()), key=lambda kv: kv[1])
    return best[0], best[1], sum(counts.values())


class SelfConsistency:
    """Share of samples agreeing with the modal answer.

    The SLM-MUX confidence signal. With ``k`` samples at temperature ``T > 0``,
    confidence is ``count(mode) / k``. Perfectly calibrated only if sampling
    spread happens to track correctness, which is precisely the assumption
    Phase 10 tests rather than accepts.
    """

    name = "self_consistency"

    def score(self, generations: Sequence[Generation], context: Mapping[str, object]) -> float:
        """Modal share in ``[0, 1]``. Zero samples scores 0.0."""
        _, count, total = modal_answer(generations)
        return count / total if total else 0.0


class LogProbMargin:
    """Mean top-1 minus top-2 log-probability, squashed to ``[0, 1]``.

    Needs only one generation, so unlike self-consistency it does not multiply
    cost by ``k``. Requires a backend that reports log-probs
    (``ModelCard.supports_logprobs``); returns 0.0 when unavailable rather than
    fabricating a value.
    """

    name = "logprob_margin"

    def __init__(self, scale: float = 2.0) -> None:
        """Args: scale: logistic temperature for squashing the raw margin."""
        self.scale = scale

    def score(self, generations: Sequence[Generation], context: Mapping[str, object]) -> float:
        """Squashed mean margin, or 0.0 when no generation reported one."""
        margins = [g.top_token_margin for g in generations if g.top_token_margin is not None]
        if not margins:
            return 0.0
        mean = sum(margins) / len(margins)
        return 1.0 / (1.0 + math.exp(-mean / self.scale))


class CrossModelAgreement:
    """Agreement between generations from *different* models.

    Where :class:`SelfConsistency` samples one model repeatedly, this compares
    independent models, so it detects the failure mode where a model is
    confidently and consistently wrong. Its complement is the contradiction
    rate SLM-MUX penalises when selecting an ensemble.
    """

    name = "cross_model_agreement"

    def score(self, generations: Sequence[Generation], context: Mapping[str, object]) -> float:
        """Largest share of models agreeing on one answer.

        Each model votes once (its own modal answer), so a model sampled ten
        times does not outvote a model sampled once.
        """
        by_model: dict[str, list[Generation]] = {}
        for gen in generations:
            by_model.setdefault(gen.model_id, []).append(gen)
        if len(by_model) < 2:
            return 0.0
        votes: dict[str, int] = {}
        for model_gens in by_model.values():
            answer, _, _ = modal_answer(model_gens)
            if answer:
                votes[answer] = votes.get(answer, 0) + 1
        if not votes:
            return 0.0
        return max(votes.values()) / len(by_model)


def contradiction_rate(generations: Sequence[Generation]) -> float:
    """Fraction of cross-model pairs that disagree.

    SLM-MUX's model-selection objective is ``UnionAcc - lambda * Contradiction``;
    this is that second term.
    """
    by_model: dict[str, str] = {}
    for gen in generations:
        by_model.setdefault(gen.model_id, normalise_answer(gen.text))
    answers = [a for a in by_model.values() if a]
    if len(answers) < 2:
        return 0.0
    pairs = 0
    disagreements = 0
    for i, first in enumerate(answers):
        for second in answers[i + 1 :]:
            pairs += 1
            if first != second:
                disagreements += 1
    return disagreements / pairs if pairs else 0.0


class RetrievalSupport:
    """Does the answer's content appear in the retrieved evidence?

    Token-overlap groundedness: the fraction of the answer's content words that
    occur in the retrieved passages. Crude but objective, cheap, and it needs
    no judge model -- which matters because Ground Rule 10 forbids a small model
    judging comparable outputs.

    Reads ``context["retrieved"]``, a sequence of passage strings.
    """

    name = "retrieval_support"

    def __init__(self, min_word_length: int = 4) -> None:
        """Args: min_word_length: ignore shorter words as function words."""
        self.min_word_length = min_word_length

    def score(self, generations: Sequence[Generation], context: Mapping[str, object]) -> float:
        """Mean per-generation support in ``[0, 1]``, or 0.0 with no evidence."""
        passages = context.get("retrieved")
        if not isinstance(passages, Sequence) or isinstance(passages, str):
            return 0.0
        evidence = normalise_answer(" ".join(str(p) for p in passages))
        evidence_words = set(evidence.split())
        if not evidence_words:
            return 0.0

        scores: list[float] = []
        for gen in generations:
            words = [
                w for w in normalise_answer(gen.text).split() if len(w) >= self.min_word_length
            ]
            if not words:
                continue
            supported = sum(1 for w in words if w in evidence_words)
            scores.append(supported / len(words))
        return sum(scores) / len(scores) if scores else 0.0


@dataclass(frozen=True, slots=True)
class CombinedConfidence:
    """Weighted mean of several signals, reporting each component too.

    Phase 10 must compare signals, so the combiner keeps the per-signal scores
    rather than collapsing them. Weights are a declared preference; the ECE
    comparison in ``docs/RESULTS.md`` is what justifies any particular choice.
    """

    checks: tuple[tuple[object, float], ...]
    name: str = "combined"

    @classmethod
    def default(cls) -> CombinedConfidence:
        """Self-consistency plus log-prob margin, equally weighted.

        Deliberately excludes retrieval support, which is only meaningful on
        the RAG path; add it explicitly there.
        """
        return cls(checks=((SelfConsistency(), 0.5), (LogProbMargin(), 0.5)))

    def components(
        self, generations: Sequence[Generation], context: Mapping[str, object]
    ) -> dict[str, float]:
        """Per-signal scores, keyed by signal name."""
        out: dict[str, float] = {}
        for check, _ in self.checks:
            out[check.name] = float(check.score(generations, context))  # type: ignore[attr-defined]
        return out

    def score(self, generations: Sequence[Generation], context: Mapping[str, object]) -> float:
        """Weighted mean of the component signals."""
        total_weight = sum(w for _, w in self.checks)
        if total_weight <= 0:
            return 0.0
        parts = self.components(generations, context)
        weighted = sum(parts[check.name] * w for check, w in self.checks)  # type: ignore[attr-defined]
        return weighted / total_weight
