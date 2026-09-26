"""Mock backend for testing without weights."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from dataclasses import dataclass
from slmflow.types import Backend, Generation, MemoryFootprint, ModelCard

@dataclass
class ScriptedResponse:
    probabilities: dict[str, float]

def make_pool(specs: list[tuple[str, float, int]], accuracy: dict = None, load_s: dict = None, tps: dict = None) -> list[ModelCard]:
    cards = []
    accuracy = accuracy or {}
    load_s = load_s or {}
    tps = tps or {}
    for model_id, params, ram in specs:
        cards.append(ModelCard(
            model_id=model_id,
            backend="mock",
            params_b=params,
            footprint=MemoryFootprint(ram_bytes=ram),
            accuracy_by_task=accuracy.get(model_id, {}),
            cold_load_s=load_s.get(model_id, 0.5),
            warm_load_s=0.1,
            tokens_per_s=tps.get(model_id, 10.0),
        ))
    return cards

class MockBackend:
    name = "mock"

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] = None,
        default_answer: str = "unknown",
        load_seconds: dict[str, float] = None,
        tokens_per_s: dict[str, float] = None,
    ):
        self.responses = responses or {}
        self.default_answer = default_answer
        self.load_seconds = load_seconds or {}
        self.tokens_per_s = tokens_per_s or {}
        self.loaded: set[str] = set()

    def is_available(self) -> bool:
        return True

    def load(self, card: ModelCard) -> None:
        self.loaded.add(card.model_id)

    def unload(self, card: ModelCard) -> None:
        self.loaded.discard(card.model_id)

    def is_loaded(self, card: ModelCard) -> bool:
        return card.model_id in self.loaded

    def measure_footprint(self, card: ModelCard) -> MemoryFootprint:
        return card.footprint if self.is_loaded(card) else MemoryFootprint()

    def generate(
        self,
        card: ModelCard,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
        grammar: str | None = None,
        stop: Sequence[str] = (),
    ) -> Generation:
        if card.model_id not in self.loaded:
            raise RuntimeError(f"Model {card.model_id} generated before load")
        
        model_responses = self.responses.get(card.model_id, {})
        ans = self.default_answer
        for k, v in model_responses.items():
            if k in prompt or k == "*":
                if isinstance(v, ScriptedResponse):
                    ans = max(v.probabilities.items(), key=lambda x: x[1])[0]
                else:
                    ans = v
                break
                
        return Generation(
            text=ans,
            model_id=card.model_id,
            prompt_tokens=10,
            output_tokens=10,
            latency_s=1.0,
            load_s=0.0,
            was_cold=False,
            schema_valid=True
        )

class RecordingBackend:
    def __init__(self, inner: Backend):
        self.inner = inner
        self.name = inner.name
        self._loads: dict[str, int] = {}

    def is_available(self) -> bool:
        return self.inner.is_available()

    def load(self, card: ModelCard) -> None:
        self._loads[card.model_id] = self._loads.get(card.model_id, 0) + 1
        self.inner.load(card)

    def unload(self, card: ModelCard) -> None:
        self.inner.unload(card)

    def is_loaded(self, card: ModelCard) -> bool:
        return self.inner.is_loaded(card)

    def measure_footprint(self, card: ModelCard) -> MemoryFootprint:
        return self.inner.measure_footprint(card)

    def generate(self, card: ModelCard, prompt: str, **kwargs) -> Generation:
        return self.inner.generate(card, prompt, **kwargs)

    def loads_of(self, model_id: str) -> int:
        return self._loads.get(model_id, 0)
