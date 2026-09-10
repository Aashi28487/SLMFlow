"""C3: abstention as a declarative contract.

The user writes::

    pipe = Pipeline.from_config("qa.yaml", guarantee=Precision(0.90))

and the framework tunes its own confidence threshold on a calibration split so
that precision on the *answered* subset meets the target, then reports the
coverage it achieved. Nobody exposes this in an orchestration API.

The mechanism is Learn-then-Test (Angelopoulos et al.), which is the right tool
because the guarantee must hold on unseen data, not merely on the split it was
tuned on:

1. Enumerate candidate thresholds (every observed confidence value).
2. For each, test the null hypothesis "precision at this threshold is below the
   target" using an exact binomial (Clopper-Pearson) lower bound.
3. Apply a fixed-sequence correction, walking thresholds from permissive to
   strict, and select the first that rejects the null at level ``delta``.

That yields a distribution-free guarantee: precision on the answered subset is
at least ``target`` with probability at least ``1 - delta``. The cost is
coverage, which is measured and reported rather than assumed.

Everything is stdlib ``math``. No scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from slmflow.errors import GuaranteeUnmet
from slmflow.telemetry.trace import get_logger

__all__ = [
    "Precision",
    "CalibrationPoint",
    "ThresholdPolicy",
    "tune_threshold",
    "binomial_lower_bound",
    "risk_coverage_curve",
    "regularised_incomplete_beta",
]

_LOG = get_logger("reliability.calibration")


@dataclass(frozen=True, slots=True)
class Precision:
    """A declarative precision guarantee on the answered subset.

    Args:
        target: Required precision, in ``(0, 1)``.
        delta: Allowed failure probability of the guarantee itself.

    Example:
        >>> Precision(0.90).target
        0.9
    """

    target: float
    delta: float = 0.1

    def __post_init__(self) -> None:
        """Reject targets that cannot be interpreted."""
        if not 0.0 < self.target < 1.0:
            raise ValueError(
                f"Precision target must be in (0, 1), got {self.target}. "
                f"Use 0.9 for 90 percent precision."
            )
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}.")

    def __str__(self) -> str:
        """Human label for a results table."""
        return f"Precision(target={self.target:.2f}, delta={self.delta:.2f})"


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """One calibration item: a confidence and whether the answer was correct."""

    confidence: float
    correct: bool


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    """A tuned threshold and the measured behaviour that justified it."""

    threshold: float
    target_precision: float
    delta: float
    achieved_precision: float
    achieved_coverage: float
    precision_lower_bound: float
    n_calibration: int
    n_answered: int
    signal: str = "combined"

    @property
    def abstention_rate(self) -> float:
        """Fraction of items the policy declines."""
        return 1.0 - self.achieved_coverage

    def accepts(self, confidence: float) -> bool:
        """True when this confidence clears the threshold."""
        return confidence >= self.threshold

    def as_row(self) -> dict[str, object]:
        """Flat dict for ``docs/RESULTS.md``."""
        return {
            "signal": self.signal,
            "target_precision": round(self.target_precision, 4),
            "delta": self.delta,
            "threshold": round(self.threshold, 4),
            "precision_lower_bound": round(self.precision_lower_bound, 4),
            "achieved_precision": round(self.achieved_precision, 4),
            "achieved_coverage": round(self.achieved_coverage, 4),
            "n_calibration": self.n_calibration,
            "n_answered": self.n_answered,
        }


# --------------------------------------------------------------------------- #
# Exact binomial machinery
# --------------------------------------------------------------------------- #
def _log_beta(a: float, b: float) -> float:
    """Log of the Beta function."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betainc_regularised(x: float, a: float, b: float, *, iterations: int = 300) -> float:
    """Regularised incomplete beta function ``I_x(a, b)``.

    Continued-fraction expansion (Lentz's algorithm), which is the standard
    stdlib-only route to an exact binomial tail.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use the symmetry relation where the fraction converges faster.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc_regularised(1.0 - x, b, a, iterations=iterations)

    tiny = 1e-30
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b)) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(iterations):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


#: Public alias. Other modules (adaptive sampling) need the incomplete beta
#: for their own posterior arithmetic and must not import a private name.
def regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``. See :func:`_betainc_regularised`."""
    return _betainc_regularised(x, a, b)


def binomial_lower_bound(successes: int, trials: int, delta: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound on a proportion.

    Exact rather than normal-approximate, because abstention decisions are made
    on small answered subsets where the normal approximation is wrong in the
    optimistic direction -- and an optimistic precision bound is precisely the
    failure this guarantee exists to prevent.

    Args:
        successes: Number of correct answers.
        trials: Number of answered items.
        delta: One-sided error probability.

    Returns:
        A lower bound ``p`` such that the true precision is at least ``p`` with
        probability ``1 - delta``. Returns 0.0 for zero trials.

    Example:
        >>> round(binomial_lower_bound(9, 10, 0.1), 3) < 0.9
        True
    """
    if trials <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    if successes >= trials:
        # Exact: solve p^n = delta.
        return float(delta ** (1.0 / trials))

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # P(X >= successes | p = mid) == I_mid(successes, trials - successes + 1)
        tail = _betainc_regularised(mid, successes, trials - successes + 1)
        if tail < delta:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
# Threshold tuning
# --------------------------------------------------------------------------- #
def tune_threshold(
    points: Sequence[CalibrationPoint],
    guarantee: Precision,
    *,
    signal: str = "combined",
    min_answered: int = 5,
) -> ThresholdPolicy:
    """Select the most permissive threshold that provably meets the target.

    Fixed-sequence testing: candidate thresholds are walked from permissive to
    strict and the first whose Clopper-Pearson lower bound clears the target is
    returned. Because the sequence is fixed in advance, no multiplicity
    correction is needed -- the family-wise error stays at ``delta``.

    Maximising coverage subject to the guarantee is the right objective: a
    threshold of 1.0 trivially satisfies any precision target by answering
    nothing, which is useless.

    Args:
        points: Calibration items, each a confidence and a correctness flag.
        guarantee: The declared target.
        signal: Name of the confidence signal, recorded in the policy.
        min_answered: Refuse to certify a threshold that answers fewer than
            this many calibration items; the bound would be vacuous.

    Returns:
        The tuned :class:`ThresholdPolicy`.

    Raises:
        GuaranteeUnmet: When no threshold meets the target on this split. The
            message reports the best lower bound actually achievable, so the
            user can lower the target or add calibration data.
    """
    if not points:
        raise GuaranteeUnmet(
            "Cannot tune a threshold with no calibration data",
            target=guarantee.target,
            best_achieved=0.0,
        )

    candidates = sorted({p.confidence for p in points})
    total = len(points)
    best_bound = 0.0
    best_policy: ThresholdPolicy | None = None

    for threshold in candidates:
        answered = [p for p in points if p.confidence >= threshold]
        if len(answered) < min_answered:
            continue
        correct = sum(1 for p in answered if p.correct)
        bound = binomial_lower_bound(correct, len(answered), guarantee.delta)
        best_bound = max(best_bound, bound)
        if bound >= guarantee.target:
            best_policy = ThresholdPolicy(
                threshold=threshold,
                target_precision=guarantee.target,
                delta=guarantee.delta,
                achieved_precision=correct / len(answered),
                achieved_coverage=len(answered) / total,
                precision_lower_bound=bound,
                n_calibration=total,
                n_answered=len(answered),
                signal=signal,
            )
            break  # First (most permissive) success maximises coverage.

    if best_policy is None:
        raise GuaranteeUnmet(
            f"No threshold on {total} calibration items meets precision {guarantee.target:.2f} "
            f"at delta={guarantee.delta:.2f}",
            target=guarantee.target,
            best_achieved=best_bound,
        )

    _LOG.info(
        "threshold tuned",
        extra={"event": "threshold_tuned", "slmflow": best_policy.as_row()},
    )
    return best_policy


def risk_coverage_curve(
    points: Sequence[CalibrationPoint], *, steps: int = 0
) -> list[tuple[float, float, float]]:
    """Build a risk-coverage curve.

    The Phase 10 headline plot. Each entry is
    ``(threshold, coverage, risk)`` where risk is the error rate on the
    answered subset. A useful confidence signal produces monotonically falling
    risk as coverage falls; a useless one produces a flat line, which is itself
    a reportable finding.

    Args:
        points: Calibration items.
        steps: Number of evenly spaced thresholds. ``0`` uses every observed
            confidence value, which is exact and needs no binning choice.

    Returns:
        Entries ordered by increasing threshold (decreasing coverage).
    """
    if not points:
        return []
    if steps > 0:
        lo = min(p.confidence for p in points)
        hi = max(p.confidence for p in points)
        span = hi - lo
        thresholds = [lo + span * i / max(1, steps - 1) for i in range(steps)]
    else:
        thresholds = sorted({p.confidence for p in points})

    total = len(points)
    curve: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        answered = [p for p in points if p.confidence >= threshold]
        if not answered:
            curve.append((threshold, 0.0, 0.0))
            continue
        errors = sum(1 for p in answered if not p.correct)
        curve.append((threshold, len(answered) / total, errors / len(answered)))
    return curve


def coverage_at_precision(
    points: Sequence[CalibrationPoint], target: float
) -> tuple[float, float] | None:
    """Empirical coverage at the first threshold reaching ``target`` precision.

    Unlike :func:`tune_threshold` this applies no statistical correction, so it
    reports what the split *did*, not what is guaranteed. Both numbers belong
    in the results table: the gap between them is the price of the guarantee.

    Returns:
        ``(threshold, coverage)``, or ``None`` when no threshold reaches the
        target even empirically.
    """
    for threshold in sorted({p.confidence for p in points}):
        answered = [p for p in points if p.confidence >= threshold]
        if not answered:
            continue
        precision = sum(1 for p in answered if p.correct) / len(answered)
        if precision >= target:
            return threshold, len(answered) / len(points)
    return None
