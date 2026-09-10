"""Confidence, verification and the C3 abstention contract."""

from slmflow.reliability.abstain import (
    AbstentionPolicy,
    FallbackChain,
    TinyVerifier,
)
from slmflow.reliability.calibration import (
    CalibrationPoint,
    Precision,
    ThresholdPolicy,
    risk_coverage_curve,
    tune_threshold,
)
from slmflow.reliability.confidence import (
    CombinedConfidence,
    CrossModelAgreement,
    LogProbMargin,
    RetrievalSupport,
    SelfConsistency,
    contradiction_rate,
    normalise_answer,
)
from slmflow.reliability.metrics import (
    accuracy,
    expected_calibration_error,
    mcnemar,
    mean_std,
    paired_bootstrap,
)

__all__ = [
    "AbstentionPolicy",
    "CalibrationPoint",
    "CombinedConfidence",
    "CrossModelAgreement",
    "FallbackChain",
    "LogProbMargin",
    "Precision",
    "RetrievalSupport",
    "SelfConsistency",
    "ThresholdPolicy",
    "TinyVerifier",
    "accuracy",
    "contradiction_rate",
    "expected_calibration_error",
    "mcnemar",
    "mean_std",
    "normalise_answer",
    "paired_bootstrap",
    "risk_coverage_curve",
    "tune_threshold",
]
