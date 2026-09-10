"""Cheap query features for the learned router.

Every feature must be computable in microseconds with stdlib only, because the
router itself has to run on T0 (a phone). That rules out embeddings and any
neural featuriser: the router's own cost must be negligible against the model
call it is deciding about, or routing loses money on every query.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from slmflow.types import SubTask, TaskType

__all__ = ["FEATURE_NAMES", "QueryFeatures", "extract_features", "feature_vector"]

#: Fixed order. The learned router's weight vector is indexed by this, so
#: appending is safe but reordering silently invalidates a trained model.
FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "log_chars",
    "log_words",
    "mean_word_len",
    "n_digits_ratio",
    "n_upper_ratio",
    "n_question_marks",
    "has_negation",
    "has_comparison",
    "has_enumeration",
    "n_clauses",
    "type_token_ratio",
    "long_word_ratio",
    "max_output_tokens_log",
    "has_schema",
    "n_context_chars_log",
    "is_classify",
    "is_extract",
    "is_summarise",
    "is_qa",
    "is_reason",
    "is_verify",
)

_NEGATION = re.compile(
    r"\b(not|no|never|without|except|unless|neither|nor|cannot|isn't|aren't|doesn't)\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(more|less|greater|fewer|better|worse|than|versus|vs\.?|compare|difference)\b",
    re.IGNORECASE,
)
_ENUMERATION = re.compile(r"(\blist\b|\ball\b|\bevery\b|\beach\b|\d\.\s|\n\s*[-*]\s)", re.IGNORECASE)
_CLAUSE = re.compile(r"[,;:]|\b(and|or|but|because|although|while|if|then)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    """Named features of one sub-task, plus the vector form the model consumes."""

    values: Mapping[str, float]

    def vector(self) -> list[float]:
        """Return features in :data:`FEATURE_NAMES` order."""
        return [float(self.values.get(name, 0.0)) for name in FEATURE_NAMES]

    def __getitem__(self, key: str) -> float:
        """Look up one feature."""
        return float(self.values[key])


def _safe_log1p(value: float) -> float:
    """``log1p`` clamped at zero so a negative can never leak in."""
    return math.log1p(max(0.0, value))


def extract_features(subtask: SubTask, context: Mapping[str, object] | None = None) -> QueryFeatures:
    """Compute features for one sub-task.

    Args:
        subtask: The sub-task being routed.
        context: Upstream pipeline context. Only its total text length is used,
            as a proxy for how much retrieved evidence the prompt will carry.

    Returns:
        A :class:`QueryFeatures`.

    Example:
        >>> from slmflow.types import SubTask, TaskSpec, TaskType
        >>> st = SubTask("s", TaskSpec("Is metformin used in cancer?", TaskType.CLASSIFY))
        >>> feats = extract_features(st)
        >>> feats["is_classify"]
        1.0
    """
    prompt = subtask.spec.prompt
    words = _WORD.findall(prompt)
    n_words = len(words)
    n_chars = len(prompt)
    lowered = [w.lower() for w in words]

    context_chars = 0.0
    if context:
        for value in context.values():
            if isinstance(value, str):
                context_chars += len(value)

    task_type = subtask.spec.task_type
    values: dict[str, float] = {
        "bias": 1.0,
        "log_chars": _safe_log1p(n_chars),
        "log_words": _safe_log1p(n_words),
        "mean_word_len": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "n_digits_ratio": (sum(c.isdigit() for c in prompt) / n_chars) if n_chars else 0.0,
        "n_upper_ratio": (sum(c.isupper() for c in prompt) / n_chars) if n_chars else 0.0,
        "n_question_marks": float(prompt.count("?")),
        "has_negation": 1.0 if _NEGATION.search(prompt) else 0.0,
        "has_comparison": 1.0 if _COMPARISON.search(prompt) else 0.0,
        "has_enumeration": 1.0 if _ENUMERATION.search(prompt) else 0.0,
        "n_clauses": float(len(_CLAUSE.findall(prompt))),
        "type_token_ratio": (len(set(lowered)) / n_words) if n_words else 0.0,
        "long_word_ratio": (sum(1 for w in words if len(w) >= 9) / n_words) if n_words else 0.0,
        "max_output_tokens_log": _safe_log1p(subtask.spec.max_output_tokens),
        "has_schema": 1.0 if subtask.spec.output_schema else 0.0,
        "n_context_chars_log": _safe_log1p(context_chars),
        "is_classify": 1.0 if task_type == TaskType.CLASSIFY else 0.0,
        "is_extract": 1.0 if task_type == TaskType.EXTRACT else 0.0,
        "is_summarise": 1.0 if task_type == TaskType.SUMMARISE else 0.0,
        "is_qa": 1.0 if task_type == TaskType.QA else 0.0,
        "is_reason": 1.0 if task_type == TaskType.REASON else 0.0,
        "is_verify": 1.0 if task_type == TaskType.VERIFY else 0.0,
    }
    return QueryFeatures(values=values)


def feature_vector(subtask: SubTask, context: Mapping[str, object] | None = None) -> list[float]:
    """Shorthand for ``extract_features(...).vector()``."""
    return extract_features(subtask, context).vector()


def standardise(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Compute per-feature mean and standard deviation.

    Logistic regression on raw features converges badly when scales differ by
    orders of magnitude, and these features mix ratios with log-counts. The
    ``bias`` column is left untouched (mean 0, std 1) so it stays a constant 1.

    Args:
        rows: Training rows in :data:`FEATURE_NAMES` order.

    Returns:
        ``(means, stds)``, with any zero std replaced by 1.0.
    """
    if not rows:
        return [0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)
    n = len(rows)
    width = len(FEATURE_NAMES)
    means = [0.0] * width
    for row in rows:
        for i in range(width):
            means[i] += row[i]
    means = [m / n for m in means]
    variances = [0.0] * width
    for row in rows:
        for i in range(width):
            variances[i] += (row[i] - means[i]) ** 2
    stds = [math.sqrt(v / n) or 1.0 for v in variances]
    bias_index = FEATURE_NAMES.index("bias")
    means[bias_index] = 0.0
    stds[bias_index] = 1.0
    return means, stds
