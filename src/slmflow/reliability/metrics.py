"""Evaluation metrics, including the significance tests Phase 12 requires.

``BUILD_BRIEF.md`` Phase 12: "mean +/- std over >=3 seeds, and a significance
test (paired bootstrap or McNemar) wherever you claim a difference". A claimed
difference without one of these is not reportable, so both are here and the
harness calls them automatically.

Pure stdlib. numpy would be faster but would breach the zero-dependency rule,
and these run on hundreds of items, not millions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "expected_calibration_error",
    "brier_score",
    "mean_std",
    "paired_bootstrap",
    "mcnemar",
    "BootstrapResult",
    "McNemarResult",
    "CalibrationBins",
    "accuracy",
    "cost_normalised_accuracy",
]


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def accuracy(predictions: Sequence[str], gold: Sequence[str]) -> float:
    """Exact-match accuracy after normalisation.

    Raises:
        ValueError: On length mismatch, which would silently skew the metric.
    """
    from slmflow.reliability.confidence import normalise_answer

    if len(predictions) != len(gold):
        raise ValueError(
            f"predictions ({len(predictions)}) and gold ({len(gold)}) differ in length."
        )
    if not predictions:
        return 0.0
    hits = sum(
        1 for p, g in zip(predictions, gold, strict=True)
        if normalise_answer(p) == normalise_answer(g)
    )
    return hits / len(predictions)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Sample mean and standard deviation (``ddof=1``).

    Returns ``(value, 0.0)`` for a single observation, and ``(0.0, 0.0)`` for
    none, so a results table never crashes on a partial run.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


def cost_normalised_accuracy(
    accuracy_value: float, seconds: float | None, peak_bytes: int | None
) -> dict[str, float | None]:
    """Accuracy per second and per GB of peak memory.

    The metric that makes "small beats big on a laptop" concrete
    (``context.md`` section 8). ``None`` propagates when the cost was not
    measured, so an unmeasured denominator can never masquerade as free.
    """
    return {
        "accuracy": accuracy_value,
        "accuracy_per_second": (
            accuracy_value / seconds if seconds and seconds > 0 else None
        ),
        "accuracy_per_gb_peak": (
            accuracy_value / (peak_bytes / 1e9) if peak_bytes and peak_bytes > 0 else None
        ),
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CalibrationBins:
    """Per-bin calibration detail behind an ECE number."""

    edges: tuple[float, ...]
    counts: tuple[int, ...]
    mean_confidence: tuple[float, ...]
    mean_accuracy: tuple[float, ...]

    def as_rows(self) -> list[dict[str, float | int]]:
        """One row per non-empty bin, for a reliability diagram."""
        rows: list[dict[str, float | int]] = []
        for i, count in enumerate(self.counts):
            if not count:
                continue
            rows.append(
                {
                    "bin_low": round(self.edges[i], 3),
                    "bin_high": round(self.edges[i + 1], 3),
                    "n": count,
                    "mean_confidence": round(self.mean_confidence[i], 4),
                    "mean_accuracy": round(self.mean_accuracy[i], 4),
                    "gap": round(self.mean_confidence[i] - self.mean_accuracy[i], 4),
                }
            )
        return rows


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 10
) -> tuple[float, CalibrationBins]:
    """Expected Calibration Error with equal-width bins.

    ECE is the mean absolute gap between confidence and accuracy, weighted by
    bin population. Phase 10's DoD requires it for at least one signal, and
    ideally compared across signals -- so this returns the bins too, because an
    ECE scalar hides *where* a signal is miscalibrated.

    Args:
        confidences: Confidence per item, in ``[0, 1]``.
        correct: Whether each item was correct.
        n_bins: Number of equal-width bins.

    Returns:
        ``(ece, bins)``.

    Raises:
        ValueError: On length mismatch.

    Example:
        >>> ece, _ = expected_calibration_error([0.9, 0.9], [True, True])
        >>> round(ece, 3)
        0.1
    """
    if len(confidences) != len(correct):
        raise ValueError(
            f"confidences ({len(confidences)}) and correct ({len(correct)}) differ in length."
        )
    if not confidences:
        return 0.0, CalibrationBins((), (), (), ())

    edges = tuple(i / n_bins for i in range(n_bins + 1))
    sums = [0.0] * n_bins
    hits = [0.0] * n_bins
    counts = [0] * n_bins

    for conf, is_correct in zip(confidences, correct, strict=True):
        clamped = min(max(conf, 0.0), 1.0)
        index = min(int(clamped * n_bins), n_bins - 1)
        counts[index] += 1
        sums[index] += clamped
        hits[index] += 1.0 if is_correct else 0.0

    total = len(confidences)
    ece = 0.0
    mean_conf: list[float] = []
    mean_acc: list[float] = []
    for i in range(n_bins):
        if counts[i]:
            conf = sums[i] / counts[i]
            acc = hits[i] / counts[i]
            ece += (counts[i] / total) * abs(conf - acc)
        else:
            conf = acc = 0.0
        mean_conf.append(conf)
        mean_acc.append(acc)

    return ece, CalibrationBins(
        edges=edges,
        counts=tuple(counts),
        mean_confidence=tuple(mean_conf),
        mean_accuracy=tuple(mean_acc),
    )


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of the confidence against the outcome.

    A proper scoring rule, so it penalises both miscalibration and lack of
    discrimination. Reported alongside ECE because ECE alone can be gamed by a
    signal that always outputs the base rate.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct differ in length.")
    if not confidences:
        return 0.0
    return sum(
        (c - (1.0 if k else 0.0)) ** 2 for c, k in zip(confidences, correct, strict=True)
    ) / len(confidences)


# --------------------------------------------------------------------------- #
# Significance tests
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Paired bootstrap outcome for a claimed difference between two systems."""

    observed_difference: float
    p_value: float
    ci_low: float
    ci_high: float
    n_items: int
    n_resamples: int
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def as_row(self) -> dict[str, object]:
        """Flat dict for a results table."""
        return {
            "difference": round(self.observed_difference, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "p_value": round(self.p_value, 4),
            "significant": self.significant,
            "n": self.n_items,
            "resamples": self.n_resamples,
        }

    def sentence(self, name_a: str, name_b: str) -> str:
        """A results-table caption that cannot be misread as stronger than it is."""
        verdict = "significant" if self.significant else "NOT significant"
        return (
            f"{name_a} - {name_b} = {self.observed_difference:+.4f} "
            f"(95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], p={self.p_value:.4f}, "
            f"n={self.n_items}): {verdict}"
        )


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Paired bootstrap test on per-item scores.

    Paired because both systems are evaluated on the *same* items, so item
    difficulty is common noise that pairing removes -- with a few hundred
    benchmark items that is the difference between detecting a real effect and
    not.

    Args:
        scores_a: Per-item score for system A.
        scores_b: Per-item score for system B, same order.
        n_resamples: Bootstrap resamples.
        alpha: Two-sided significance level for the interval.
        seed: RNG seed (Ground Rule 11).

    Returns:
        A :class:`BootstrapResult`.

    Raises:
        ValueError: On length mismatch or empty input.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"paired bootstrap needs equal lengths, got {len(scores_a)} and {len(scores_b)}."
        )
    if not scores_a:
        raise ValueError("paired bootstrap needs at least one item.")

    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True)]
    n = len(diffs)
    observed = sum(diffs) / n

    rng = random.Random(seed)
    samples: list[float] = []
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        mean = total / n
        samples.append(mean)
        # Centred null: does a resample deviate from the observed mean by at
        # least as much as the observed mean deviates from zero?
        if abs(mean - observed) >= abs(observed):
            at_least_as_extreme += 1

    samples.sort()
    lo_index = max(0, int((alpha / 2) * n_resamples) - 1)
    hi_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return BootstrapResult(
        observed_difference=observed,
        p_value=(at_least_as_extreme + 1) / (n_resamples + 1),
        ci_low=samples[lo_index],
        ci_high=samples[hi_index],
        n_items=n,
        n_resamples=n_resamples,
        alpha=alpha,
    )


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """McNemar's test for two classifiers on the same items."""

    b: int
    c: int
    statistic: float
    p_value: float
    exact: bool

    @property
    def significant(self) -> bool:
        """True at the conventional 0.05 level."""
        return self.p_value < 0.05

    def as_row(self) -> dict[str, object]:
        """Flat dict for a results table."""
        return {
            "a_only_correct": self.b,
            "b_only_correct": self.c,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 5),
            "test": "exact binomial" if self.exact else "chi-square (Yates)",
            "significant": self.significant,
        }


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """McNemar's test on paired binary outcomes.

    The right test for "is classifier A better than classifier B on this exact
    task set", because it conditions on the discordant pairs and ignores items
    both systems get right or wrong. Uses the exact binomial when discordant
    pairs are few (``b + c < 25``), where the chi-square approximation is
    unreliable.

    Args:
        correct_a: Per-item correctness for system A.
        correct_b: Per-item correctness for system B, same order.

    Returns:
        A :class:`McNemarResult`.

    Raises:
        ValueError: On length mismatch.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"mcnemar needs equal lengths, got {len(correct_a)} and {len(correct_b)}."
        )
    b = sum(1 for a, bb in zip(correct_a, correct_b, strict=True) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b, strict=True) if bb and not a)
    n = b + c

    if n == 0:
        return McNemarResult(b=b, c=c, statistic=0.0, p_value=1.0, exact=True)

    if n < 25:
        # Exact two-sided binomial with p = 0.5.
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
        return McNemarResult(b=b, c=c, statistic=float(k), p_value=min(1.0, 2.0 * tail), exact=True)

    statistic = (abs(b - c) - 1.0) ** 2 / n  # Yates continuity correction
    return McNemarResult(
        b=b, c=c, statistic=statistic, p_value=_chi2_sf_1df(statistic), exact=False
    )


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 degree of freedom.

    ``P(X > x) = erfc(sqrt(x / 2))`` for 1 df, which ``math.erfc`` gives
    exactly -- no table and no scipy.
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))
