"""Shared fixtures. The entire suite runs with no model weights on disk.

``BUILD_BRIEF.md`` section 3: "a ``MockBackend`` that returns scripted,
deterministic outputs so the entire test suite runs with no model weights
present". Any test needing real weights is marked ``real_model`` and is
deselected by default in CI.
"""

from __future__ import annotations

import pytest

from slmflow.models.backends.mock import MockBackend, ScriptedResponse, make_pool
from slmflow.models.registry import ModelRegistry
from slmflow.resource.budget import MemoryBudget
from slmflow.resource.residency import ResourceManager
from slmflow.types import MemoryFootprint, ModelCard, SubTask, TaskSpec, TaskType

GB = 1_000_000_000


@pytest.fixture
def small_card() -> ModelCard:
    """A 0.5B model that fits anywhere."""
    return ModelCard(
        model_id="small",
        backend="mock",
        params_b=0.5,
        footprint=MemoryFootprint(ram_bytes=400_000_000),
        cold_load_s=0.8,
        warm_load_s=0.1,
        tokens_per_s=34.0,
        accuracy_by_task={"classify": 0.60, "qa": 0.52},
    )


@pytest.fixture
def large_card() -> ModelCard:
    """A 3B model that competes for the same slot."""
    return ModelCard(
        model_id="large",
        backend="mock",
        params_b=3.0,
        footprint=MemoryFootprint(ram_bytes=2_200_000_000),
        cold_load_s=4.2,
        warm_load_s=0.6,
        tokens_per_s=9.5,
        accuracy_by_task={"classify": 0.81, "qa": 0.74},
    )


@pytest.fixture
def gpu_card() -> ModelCard:
    """A 7B model charged to VRAM, for two-tier budget tests."""
    return ModelCard(
        model_id="gpu-7b",
        backend="mock",
        params_b=7.0,
        footprint=MemoryFootprint(vram_bytes=4_400_000_000),
        cold_load_s=9.0,
        warm_load_s=1.2,
        tokens_per_s=18.0,
        accuracy_by_task={"classify": 0.88, "qa": 0.83},
    )


@pytest.fixture
def unprofiled_card() -> ModelCard:
    """A card with no measurements, to prove the library refuses to guess."""
    return ModelCard(model_id="unmeasured", backend="mock", params_b=1.5)


@pytest.fixture
def backend() -> MockBackend:
    """A deterministic backend whose two models disagree."""
    responses = {
        "small": {
            "metformin": ScriptedResponse({"investigational": 0.6, "approved": 0.4}),
            "*": ScriptedResponse({"hypothesis": 1.0}),
        },
        "large": {
            "metformin": ScriptedResponse({"investigational": 0.9, "hypothesis": 0.1}),
            "*": ScriptedResponse({"investigational": 1.0}),
        },
        "gpu-7b": {"*": ScriptedResponse({"investigational": 1.0})},
        "unmeasured": {"*": ScriptedResponse({"approved": 1.0})},
    }
    return MockBackend(
        responses,
        default_answer="unknown",
        load_seconds={"small": 0.8, "large": 4.2, "gpu-7b": 9.0, "unmeasured": 1.0},
        tokens_per_s={"small": 34.0, "large": 9.5, "gpu-7b": 18.0},
    )


@pytest.fixture
def registry(
    small_card: ModelCard, large_card: ModelCard, backend: MockBackend
) -> ModelRegistry:
    """A two-model registry on the mock backend."""
    reg = ModelRegistry([small_card, large_card])
    reg.register_backend("mock", backend)
    return reg


@pytest.fixture
def single_slot_budget() -> MemoryBudget:
    """A budget that fits the large model but not both models at once."""
    return MemoryBudget.single_slot(2_500_000_000)


@pytest.fixture
def two_tier_budget() -> MemoryBudget:
    """The shape of the real T1 machine: ~5 GB VRAM over ~8 GB RAM."""
    return MemoryBudget(vram_bytes=5 * GB, ram_bytes=8 * GB)


@pytest.fixture
def manager(
    single_slot_budget: MemoryBudget,
    small_card: ModelCard,
    large_card: ModelCard,
    backend: MockBackend,
) -> ResourceManager:
    """A resource manager with both models registered, measurement disabled."""
    rm = ResourceManager(single_slot_budget, measure_on_load=False)
    rm.register(small_card, backend)
    rm.register(large_card, backend)
    return rm


@pytest.fixture
def classify_task() -> SubTask:
    """A classification sub-task mentioning a scripted drug name."""
    return SubTask(
        name="label",
        spec=TaskSpec(
            prompt="Does metformin have evidence beyond its approved use in cancer?",
            task_type=TaskType.CLASSIFY,
            task_id="metformin|cancer",
            max_output_tokens=24,
        ),
    )


@pytest.fixture
def pool_of_three() -> list[ModelCard]:
    """Three synthetic cards for scheduler tests."""
    return make_pool(
        [("a", 0.5, 400_000_000), ("b", 1.5, 900_000_000), ("c", 3.0, 2_200_000_000)],
        accuracy={
            "a": {"classify": 0.55, "extract": 0.40, "qa": 0.50},
            "b": {"classify": 0.70, "extract": 0.65, "qa": 0.60},
            "c": {"classify": 0.85, "extract": 0.80, "qa": 0.78},
        },
        load_s={"a": 0.5, "b": 1.5, "c": 4.0},
        tps={"a": 40.0, "b": 20.0, "c": 8.0},
    )
