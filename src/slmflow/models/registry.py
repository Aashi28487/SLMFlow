"""Model registry and capability lookup."""

from __future__ import annotations

import json
from pathlib import Path

from typing import Iterable
from slmflow.types import Backend, ModelCard
from slmflow.errors import ModelNotFound, BackendUnavailable

class ModelRegistry:
    """Registry for available models and their backend implementations."""

    def __init__(self, models: Iterable[ModelCard] = ()):
        self._cards: dict[str, ModelCard] = {}
        self._backends: dict[str, Backend] = {}
        for card in models:
            self._cards[card.model_id] = card

    def register(self, card: ModelCard) -> None:
        """Register a model card."""
        self._cards[card.model_id] = card

    def register_backend(self, name: str, backend: Backend) -> None:
        """Register a backend implementation."""
        self._backends[name] = backend

    def get_card(self, model_id: str) -> ModelCard:
        """Get a model card by ID."""
        if model_id not in self._cards:
            raise ModelNotFound(model_id, known=list(self._cards.keys()))
        return self._cards[model_id]
        
    def __iter__(self):
        return iter(self._cards.values())

    def unprofiled_report(self) -> list[str]:
        return [c.model_id for c in self._cards.values() if not c.is_profiled]

    def get_backend(self, name: str) -> Backend:
        """Get a backend by name."""
        if name not in self._backends:
            raise BackendUnavailable(
                f"Backend {name!r} not registered.",
                backend=name,
                install_hint=f"Register backend via registry.register_backend('{name}', ...)",
            )
        return self._backends[name]

    def backend_for(self, card: ModelCard) -> Backend:
        return self.get_backend(card.backend)

    def within_budget(self, budget) -> list[ModelCard]:
        return [c for c in self._cards.values() if budget.fits(c.footprint)]

    def cascade_order(self, task_type: str) -> list[ModelCard]:
        def sort_key(c: ModelCard):
            acc = c.accuracy_for(task_type)
            if acc is None:
                return -1.0
            return acc
        return sorted(self._cards.values(), key=sort_key, reverse=True)

    def cheapest(self) -> ModelCard | None:
        """Return the card with the smallest VRAM footprint, or None if empty."""
        cards = list(self._cards.values())
        if not cards:
            return None
        return min(cards, key=lambda c: c.footprint.vram_bytes)

    def digests(self) -> dict[str, str | None]:
        """Return a mapping of model_id → digest for provenance tracking."""
        return {model_id: card.digest for model_id, card in self._cards.items()}

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> ModelRegistry:
        """Load model cards from JSON (default ``benchmarks/results/model_cards.json``)."""
        import json
        from pathlib import Path
        p = Path(path) if path is not None else DEFAULT_CARDS_PATH
        if not p.exists():
            from slmflow.errors import ConfigError
            raise ConfigError(
                f"Missing model cards file {p}. Create it with "
                f"`python -m benchmarks.profile_models` (needs a source checkout), "
                f"pass registry=ModelRegistry(...) explicitly, or use --mock on the CLI"
            )
        with p.open() as f:
            data = json.load(f)
        return cls([card_from_dict(d) for d in data])

from typing import Any
from slmflow.types import MemoryFootprint, Quantisation

DEFAULT_CARDS_PATH = Path("benchmarks") / "results" / "model_cards.json"


def load_cards(path: Path | str) -> list[ModelCard]:
    """Load a list of model cards from a JSON file."""
    p = Path(path)
    with p.open() as f:
        data = json.load(f)
    return [card_from_dict(d) for d in data]


def save_cards(cards: list[ModelCard], path: Path | str) -> None:
    """Save a list of model cards to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump([card_to_dict(c) for c in cards], f, indent=2)


def card_to_dict(card: ModelCard) -> dict[str, Any]:
    return {
        "model_id": card.model_id,
        "backend": card.backend,
        "params_b": card.params_b,
        "quantisation": card.quantisation,
        "digest": card.digest,
        "footprint": {
            "vram_bytes": card.footprint.vram_bytes,
            "ram_bytes": card.footprint.ram_bytes,
            "disk_bytes": card.footprint.disk_bytes,
        },
        "cold_load_s": card.cold_load_s,
        "warm_load_s": card.warm_load_s,
        "tokens_per_s": card.tokens_per_s,
        "prefill_tokens_per_s": card.prefill_tokens_per_s,
        "context_tokens": card.context_tokens,
        "supports_grammar": card.supports_grammar,
        "supports_logprobs": card.supports_logprobs,
        "accuracy_by_task": dict(card.accuracy_by_task),
        "measured_on_device": card.measured_on_device,
        "measured_at": card.measured_at,
        "notes": card.notes,
    }

def card_from_dict(data: dict[str, Any]) -> ModelCard:
    from slmflow.errors import ConfigError
    if "model_id" not in data:
        raise ConfigError("Missing model_id")
    if "backend" not in data:
        raise ConfigError("Missing backend")
    
    try:
        params_b = float(data.get("params_b", 0.0))
        cold_load_s = data.get("cold_load_s")
        if cold_load_s is not None:
            cold_load_s = float(cold_load_s)
    except (ValueError, TypeError):
        raise ConfigError("must be a number")

    fp_data = data.get("footprint", {})
    footprint = MemoryFootprint(
        vram_bytes=fp_data.get("vram_bytes", 0),
        ram_bytes=fp_data.get("ram_bytes", 0),
        disk_bytes=fp_data.get("disk_bytes", 0),
    )
    return ModelCard(
        model_id=data["model_id"],
        backend=data["backend"],
        params_b=params_b,
        quantisation=Quantisation(data.get("quantisation", "unknown")),
        digest=data.get("digest"),
        footprint=footprint,
        cold_load_s=cold_load_s,
        warm_load_s=data.get("warm_load_s"),
        tokens_per_s=data.get("tokens_per_s"),
        prefill_tokens_per_s=data.get("prefill_tokens_per_s"),
        context_tokens=data.get("context_tokens", 2048),
        supports_grammar=data.get("supports_grammar", False),
        supports_logprobs=data.get("supports_logprobs", False),
        accuracy_by_task=data.get("accuracy_by_task", {}),
        measured_on_device=data.get("measured_on_device"),
        measured_at=data.get("measured_at"),
        notes=data.get("notes", ""),
    )
