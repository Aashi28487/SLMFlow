"""Exception hierarchy for SLMFlow.

Every message must tell the caller what to change, not merely what went wrong
(``BUILD_BRIEF.md`` section 3). Exceptions carry structured attributes so the
telemetry layer can record them without re-parsing prose.
"""

from __future__ import annotations

__all__ = [
    "SLMFlowError",
    "ConfigError",
    "ResourceExhausted",
    "SchemaViolation",
    "AbstainedError",
    "BackendUnavailable",
    "ModelNotFound",
    "GuaranteeUnmet",
    "ScheduleInfeasible",
]


class SLMFlowError(Exception):
    """Root of every error raised by this library."""


class ConfigError(SLMFlowError):
    """A pipeline configuration is malformed or internally inconsistent."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        """Store the config location alongside the message.

        Args:
            message: What is wrong and what to change.
            path: Dotted path to the offending key, e.g. ``steps[2].model``.
        """
        self.path = path
        super().__init__(f"{message} (at {path})" if path else message)


class ResourceExhausted(SLMFlowError):
    """A memory tier could not satisfy a residency request.

    Names *which* tier was exceeded and by how much, per the two-tier budget
    rule in ``BUILD_BRIEF.md`` section 2.
    """

    def __init__(
        self,
        message: str,
        *,
        tier: str,
        requested_bytes: int,
        available_bytes: int,
        budget_bytes: int,
    ) -> None:
        """Record the tier and the arithmetic that failed."""
        self.tier = tier
        self.requested_bytes = requested_bytes
        self.available_bytes = available_bytes
        self.budget_bytes = budget_bytes
        over = requested_bytes - available_bytes
        super().__init__(
            f"{message}: {tier} needs {requested_bytes / 1e9:.2f} GB but only "
            f"{available_bytes / 1e9:.2f} GB of the {budget_bytes / 1e9:.2f} GB budget is free "
            f"(over by {over / 1e9:.2f} GB). Free a model, raise the {tier} budget, "
            f"or choose a smaller quantisation."
        )


class ScheduleInfeasible(SLMFlowError):
    """No schedule exists for this task graph within the budget."""


class SchemaViolation(SLMFlowError):
    """Model output did not satisfy the declared output schema."""

    def __init__(self, message: str, *, raw_output: str, schema_name: str) -> None:
        """Keep the raw text so a repair pass or a trace can use it."""
        self.raw_output = raw_output
        self.schema_name = schema_name
        preview = raw_output if len(raw_output) <= 200 else raw_output[:200] + "..."
        super().__init__(
            f"{message} (schema={schema_name}). Raw output was: {preview!r}. "
            f"Enable grammar-constrained decoding, or add a repair step."
        )


class AbstainedError(SLMFlowError):
    """The pipeline declined to answer and the caller asked for a hard failure."""

    def __init__(self, message: str, *, confidence: float, threshold: float, reason: str) -> None:
        """Record the confidence arithmetic behind the abstention."""
        self.confidence = confidence
        self.threshold = threshold
        self.reason = reason
        super().__init__(
            f"{message}: confidence {confidence:.3f} < threshold {threshold:.3f} "
            f"(reason={reason}). Lower the guarantee, allow escalation, "
            f"or read result.abstained instead of requiring an answer."
        )


class BackendUnavailable(SLMFlowError):
    """A backend plugin cannot be used on this machine."""

    def __init__(self, message: str, *, backend: str, install_hint: str = "") -> None:
        """Tell the user how to make the backend available."""
        self.backend = backend
        hint = f" Try: {install_hint}" if install_hint else ""
        super().__init__(f"{message} (backend={backend}).{hint}")


class ModelNotFound(SLMFlowError):
    """A model id was requested that the registry does not know."""

    def __init__(self, model_id: str, known: list[str]) -> None:
        """List what the registry does know, so the typo is obvious."""
        self.model_id = model_id
        self.known = known
        listing = ", ".join(sorted(known)) if known else "<registry is empty>"
        super().__init__(f"Unknown model {model_id!r}. Registered models: {listing}.")


class GuaranteeUnmet(SLMFlowError):
    """Threshold tuning could not reach the requested guarantee."""

    def __init__(self, message: str, *, target: float, best_achieved: float) -> None:
        """Report the best the calibration split could actually support."""
        self.target = target
        self.best_achieved = best_achieved
        super().__init__(
            f"{message}: target {target:.3f}, best achievable on the calibration split "
            f"{best_achieved:.3f}. Lower the target, add calibration data, "
            f"or add a stronger model to the pool."
        )
