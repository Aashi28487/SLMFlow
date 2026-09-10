"""LoRA adapter loading for a narrow task.

``BUILD_BRIEF.md`` Phase 8 keeps training off-machine (Colab-class) and has the
framework load the resulting adapter locally at inference. With ~5 GB usable
VRAM, QLoRA on a 0.5-1.5B model is *plausible* locally -- but per Ground Rule 1
that stays an open question until a run completes on the target machine, so
nothing here claims it works.

llama.cpp consumes LoRA as a GGUF adapter file. This module locates and
validates adapters and reports what it found; the actual attach happens through
the backend, since only the backend owns the model handle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from slmflow.errors import ConfigError
from slmflow.telemetry.trace import get_logger

__all__ = ["AdapterSpec", "attach_adapter", "list_adapters", "DEFAULT_ADAPTER_DIR"]

_LOG = get_logger("finetune")

DEFAULT_ADAPTER_DIR = Path("adapters")


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """A trained adapter and the provenance needed to reproduce it."""

    name: str
    path: Path
    base_model: str
    task: str
    scale: float = 1.0
    trained_at: str | None = None
    trained_on: str | None = None
    train_examples: int | None = None
    notes: str = ""

    @property
    def exists(self) -> bool:
        """True when the adapter file is on disk."""
        return self.path.exists()

    def as_row(self) -> dict[str, object]:
        """Flat dict for a results table."""
        return {
            "adapter": self.name,
            "base_model": self.base_model,
            "task": self.task,
            "scale": self.scale,
            "trained_at": self.trained_at or "not recorded",
            "trained_on": self.trained_on or "not recorded",
            "train_examples": self.train_examples,
            "present": self.exists,
        }


def list_adapters(directory: str | Path = DEFAULT_ADAPTER_DIR) -> list[AdapterSpec]:
    """Discover adapters described by ``*.json`` sidecars next to the weights.

    A sidecar is required rather than optional: an adapter without recorded
    provenance cannot appear in a results table, so loading one silently would
    produce an unreportable number.
    """
    d = Path(directory)
    if not d.exists():
        return []
    specs: list[AdapterSpec] = []
    for sidecar in sorted(d.glob("*.json")):
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"Adapter sidecar {sidecar} is not valid JSON: {exc}", path=str(sidecar)
            ) from exc
        for required in ("base_model", "task"):
            if not payload.get(required):
                raise ConfigError(
                    f"Adapter sidecar {sidecar} is missing {required!r}. An adapter without "
                    f"recorded provenance cannot be used in a measured result.",
                    path=str(sidecar),
                )
        weights = Path(payload.get("path") or sidecar.with_suffix(".gguf"))
        specs.append(
            AdapterSpec(
                name=payload.get("name") or sidecar.stem,
                path=weights,
                base_model=str(payload["base_model"]),
                task=str(payload["task"]),
                scale=float(payload.get("scale", 1.0)),
                trained_at=payload.get("trained_at"),
                trained_on=payload.get("trained_on"),
                train_examples=payload.get("train_examples"),
                notes=str(payload.get("notes", "")),
            )
        )
    return specs


def attach_adapter(backend: object, spec: AdapterSpec) -> bool:
    """Ask a backend to apply an adapter.

    Returns:
        True when the backend applied it, False when the backend has no adapter
        support. Never raises on an unsupported backend: an adapter is an
        optional enhancement, and a pipeline must still run without it.

    Raises:
        ConfigError: When the adapter file itself is missing.
    """
    if not spec.exists:
        raise ConfigError(
            f"Adapter weights not found at {spec.path}. Train off-machine and copy the "
            f"GGUF adapter here, or drop the adapter from the config.",
            path=str(spec.path),
        )
    hook = getattr(backend, "apply_adapter", None)
    if not callable(hook):
        _LOG.warning(
            "backend does not support adapters; running the base model",
            extra={
                "event": "adapter_unsupported",
                "slmflow": {
                    "backend": getattr(backend, "name", type(backend).__name__),
                    "adapter": spec.name,
                },
            },
        )
        return False
    hook(spec.path, spec.scale)
    _LOG.info(
        "adapter attached",
        extra={"event": "adapter_attached", "slmflow": spec.as_row()},
    )
    return True
