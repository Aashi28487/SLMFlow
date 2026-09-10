"""Value objects, enums and plugin protocols. Zero third-party imports.

This module is the contract every other module speaks. It is deliberately
dependency-free so that ``import slmflow`` works on a phone (T0) where numpy,
torch and pydantic-core are not installable.

Design notes (expanded in ``docs/ARCHITECTURE.md``):
    * ``Protocol`` for plugin seams: ``Backend``, ``Router``, ``Strategy``,
      ``ReliabilityCheck``. Structural typing means a plugin never imports us,
      so a third-party backend needs no dependency on ``slmflow``.
    * ``abc.ABC`` only where real shared implementation exists (see
      ``slmflow.strategies.base.BaseStrategy``).
    * ``@dataclass(frozen=True, slots=True)`` for every value object: a trace is
      a record of the past and must not be mutable after the fact.
"""

from __future__ import annotations

import enum
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AbstentionReason",
    "Backend",
    "DeviceCard",
    "DeviceTier",
    "ExecutionPattern",
    "Generation",
    "MemoryFootprint",
    "MemoryTier",
    "ModelCard",
    "Quantisation",
    "ReliabilityCheck",
    "Result",
    "Router",
    "RunTrace",
    "Schedule",
    "ScheduleEntry",
    "StepTrace",
    "StrEnum",
    "Strategy",
    "SubTask",
    "TaskGraph",
    "TaskSpec",
    "TaskType",
]


if sys.version_info >= (3, 11):  # pragma: no cover - version branch
    from enum import StrEnum
else:  # pragma: no cover - version branch

    class StrEnum(str, enum.Enum):  # noqa: D101 - stdlib backport for 3.10
        def __str__(self) -> str:  # noqa: D105
            return str(self.value)


# --------------------------------------------------------------------------- #
# Closed sets
# --------------------------------------------------------------------------- #
class TaskType(StrEnum):
    """What kind of work a sub-task is. Drives routing and metric choice."""

    CLASSIFY = "classify"
    EXTRACT = "extract"
    SUMMARISE = "summarise"
    QA = "qa"
    REASON = "reason"
    VERIFY = "verify"
    REWRITE = "rewrite"


class ExecutionPattern(StrEnum):
    """How the orchestrator composes steps."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ENSEMBLE = "ensemble"


class AbstentionReason(StrEnum):
    """Why a pipeline declined to answer."""

    LOW_CONFIDENCE = "low_confidence"
    CONTRADICTION = "contradiction"
    NO_RETRIEVAL_SUPPORT = "no_retrieval_support"
    SCHEMA_FAILURE = "schema_failure"
    VERIFIER_REJECTED = "verifier_rejected"
    RESOURCE_EXHAUSTED = "resource_exhausted"


class MemoryTier(StrEnum):
    """A level of the memory hierarchy the resource manager accounts for.

    The T1 machine has a discrete GPU, so the budget is a vector over these
    tiers rather than a scalar (``BUILD_BRIEF.md`` section 2).
    """

    VRAM = "vram"
    RAM = "ram"
    DISK = "disk"


class DeviceTier(StrEnum):
    """Hardware class, a first-class object per ``BUILD_BRIEF.md`` section 2."""

    T0_PHONE = "t0_phone"
    T1_LAPTOP = "t1_laptop"
    T2_WORKSTATION = "t2_workstation"


class Quantisation(StrEnum):
    """GGUF quantisation levels this project cares about."""

    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MemoryFootprint:
    """Bytes a model occupies, per tier. Measured, never estimated."""

    vram_bytes: int = 0
    ram_bytes: int = 0
    disk_bytes: int = 0

    def get(self, tier: MemoryTier) -> int:
        """Return the byte count for one tier."""
        if tier is MemoryTier.VRAM:
            return self.vram_bytes
        if tier is MemoryTier.RAM:
            return self.ram_bytes
        return self.disk_bytes

    def __add__(self, other: MemoryFootprint) -> MemoryFootprint:
        """Sum two footprints tier-wise."""
        return MemoryFootprint(
            vram_bytes=self.vram_bytes + other.vram_bytes,
            ram_bytes=self.ram_bytes + other.ram_bytes,
            disk_bytes=self.disk_bytes + other.disk_bytes,
        )

    @property
    def resident_bytes(self) -> int:
        """Total live memory, excluding disk (which is not residency)."""
        return self.vram_bytes + self.ram_bytes


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Measured capability and cost profile of one model on one device.

    Every numeric cost field defaults to ``None``, meaning *not yet measured*.
    Nothing in this library may substitute an estimate for a measurement
    (Ground Rule 1); consumers handle ``None`` rather than receive a guess.
    """

    model_id: str
    backend: str
    params_b: float
    quantisation: Quantisation = Quantisation.UNKNOWN
    digest: str | None = None
    footprint: MemoryFootprint = field(default_factory=MemoryFootprint)
    cold_load_s: float | None = None
    warm_load_s: float | None = None
    tokens_per_s: float | None = None
    prefill_tokens_per_s: float | None = None
    context_tokens: int = 2048
    supports_grammar: bool = False
    supports_logprobs: bool = False
    accuracy_by_task: Mapping[str, float] = field(default_factory=dict)
    measured_on_device: str | None = None
    measured_at: str | None = None
    notes: str = ""

    @property
    def is_profiled(self) -> bool:
        """True only when the fields the scheduler needs were actually measured."""
        return (
            self.cold_load_s is not None
            and self.tokens_per_s is not None
            and self.footprint.resident_bytes > 0
        )

    def accuracy_for(self, task_type: TaskType | str) -> float | None:
        """Measured accuracy on a task type, or ``None`` if never measured."""
        return self.accuracy_by_task.get(str(task_type))

    def expected_seconds(self, prompt_tokens: int, output_tokens: int) -> float | None:
        """Warm generation time predicted from measured throughput.

        Returns ``None`` when throughput was never measured, so a caller cannot
        silently propagate a fabricated cost.
        """
        if self.tokens_per_s is None or self.tokens_per_s <= 0:
            return None
        decode_s = output_tokens / self.tokens_per_s
        if self.prefill_tokens_per_s and self.prefill_tokens_per_s > 0:
            return decode_s + prompt_tokens / self.prefill_tokens_per_s
        return decode_s


@dataclass(frozen=True, slots=True)
class DeviceCard:
    """Measured properties of *this* machine. Produced by ``slmflow.device.probe``."""

    tier: DeviceTier
    cpu_model: str
    logical_cores: int
    total_ram_bytes: int
    available_ram_bytes: int
    total_vram_bytes: int = 0
    gpu_model: str = ""
    cuda_available: bool = False
    free_disk_bytes: int = 0
    platform: str = ""
    python_version: str = ""
    probed_at: str = ""

    @property
    def has_gpu(self) -> bool:
        """True when a GPU with non-zero VRAM was detected."""
        return self.total_vram_bytes > 0

    def default_budget_bytes(self, headroom: float = 0.75) -> dict[MemoryTier, int]:
        """Suggest a per-tier budget, leaving headroom for the OS.

        Args:
            headroom: Fraction of each tier the library may claim.

        Returns:
            Mapping of tier to byte budget. VRAM is 0 when no GPU is present.
        """
        return {
            MemoryTier.VRAM: int(self.total_vram_bytes * headroom),
            MemoryTier.RAM: int(self.available_ram_bytes * headroom),
            MemoryTier.DISK: self.free_disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One unit of user-visible work handed to a pipeline."""

    prompt: str
    task_type: TaskType = TaskType.QA
    task_id: str = ""
    inputs: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] | None = None
    max_output_tokens: int = 256
    gold: Any | None = None

    def __post_init__(self) -> None:
        """Assign a stable id when the caller did not supply one."""
        if not self.task_id:
            object.__setattr__(self, "task_id", uuid.uuid4().hex[:12])


@dataclass(frozen=True, slots=True)
class SubTask:
    """A node of the task DAG."""

    name: str
    spec: TaskSpec
    depends_on: tuple[str, ...] = ()
    candidate_models: tuple[str, ...] = ()
    pinned_model: str | None = None


@dataclass(frozen=True, slots=True)
class TaskGraph:
    """An immutable DAG of sub-tasks, lazily walkable in dependency order."""

    nodes: tuple[SubTask, ...]

    def __post_init__(self) -> None:
        """Reject duplicate names, dangling dependencies and cycles at construction."""
        seen: set[str] = set()
        for node in self.nodes:
            if node.name in seen:
                raise ValueError(f"Duplicate sub-task name {node.name!r} in task graph.")
            seen.add(node.name)
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in seen:
                    raise ValueError(
                        f"Sub-task {node.name!r} depends on {dep!r}, which is not in the "
                        f"graph. Known sub-tasks: {sorted(seen)}."
                    )
        # Kahn's algorithm. A short cycle report beats an execution-time hang.
        remaining = {n.name: set(n.depends_on) for n in self.nodes}
        while remaining:
            ready = [name for name, deps in remaining.items() if not deps]
            if not ready:
                raise ValueError(f"Task graph has a dependency cycle among: {sorted(remaining)}.")
            for name in ready:
                del remaining[name]
            for deps in remaining.values():
                deps.difference_update(ready)

    def by_name(self, name: str) -> SubTask:
        """Look up one sub-task by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"No sub-task named {name!r}.")

    def topological_layers(self) -> Iterator[tuple[SubTask, ...]]:
        """Yield successive layers of sub-tasks whose dependencies are satisfied.

        Each layer is internally independent, so a layer is the natural unit of
        parallel execution, bounded by the memory budget rather than CPU count.
        """
        done: set[str] = set()
        pending = list(self.nodes)
        while pending:
            layer = tuple(n for n in pending if set(n.depends_on) <= done)
            if not layer:  # pragma: no cover - __post_init__ rejects cycles already
                raise ValueError("Task graph is not a DAG.")
            yield layer
            done.update(n.name for n in layer)
            pending = [n for n in pending if n.name not in done]

    def linearise(self) -> tuple[SubTask, ...]:
        """Flatten to one valid execution order."""
        return tuple(n for layer in self.topological_layers() for n in layer)


@dataclass(frozen=True, slots=True)
class Generation:
    """One raw completion from one backend, with its measured cost."""

    text: str
    model_id: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    load_s: float = 0.0
    was_cold: bool = False
    logprob_sum: float | None = None
    top_token_margin: float | None = None
    finish_reason: str = "stop"
    schema_valid: bool | None = None

    @property
    def total_s(self) -> float:
        """Wall-clock cost including any load this generation paid for."""
        return self.latency_s + self.load_s


@dataclass(frozen=True, slots=True)
class StepTrace:
    """What happened at one sub-task. Written to JSONL by the telemetry layer."""

    step: str
    model_id: str
    strategy: str
    generations: tuple[Generation, ...] = ()
    confidence: float | None = None
    confidence_signals: Mapping[str, float] = field(default_factory=dict)
    swaps: int = 0
    swap_seconds: float = 0.0
    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0
    escalated_from: str | None = None
    abstained: bool = False
    abstention_reason: AbstentionReason | None = None
    error: str | None = None
    started_at: float = 0.0
    wall_s: float = 0.0

    @property
    def n_generations(self) -> int:
        """How many samples this step actually drew (the C4 cost number)."""
        return len(self.generations)

    @property
    def output_tokens(self) -> int:
        """Total tokens generated at this step."""
        return sum(g.output_tokens for g in self.generations)


@dataclass(frozen=True, slots=True)
class RunTrace:
    """The complete record of one pipeline run: quality and cost together."""

    run_id: str
    task_id: str
    pipeline: str
    device: str
    steps: tuple[StepTrace, ...] = ()
    wall_s: float = 0.0
    seed: int | None = None
    config_digest: str = ""
    started_at: float = field(default_factory=time.time)

    @property
    def total_swaps(self) -> int:
        """Model loads charged across the whole run."""
        return sum(s.swaps for s in self.steps)

    @property
    def swap_seconds(self) -> float:
        """Seconds spent loading models: the cost other frameworks ignore."""
        return sum(s.swap_seconds for s in self.steps)

    @property
    def total_generations(self) -> int:
        """Total samples drawn across every step."""
        return sum(s.n_generations for s in self.steps)

    @property
    def output_tokens(self) -> int:
        """Total tokens generated across every step."""
        return sum(s.output_tokens for s in self.steps)

    @property
    def peak_vram_bytes(self) -> int:
        """Highest VRAM residency observed during the run."""
        return max((s.peak_vram_bytes for s in self.steps), default=0)

    @property
    def peak_ram_bytes(self) -> int:
        """Highest system-RAM residency observed during the run."""
        return max((s.peak_ram_bytes for s in self.steps), default=0)

    @property
    def models_used(self) -> tuple[str, ...]:
        """Distinct models touched, in first-use order."""
        seen: list[str] = []
        for step in self.steps:
            if step.model_id and step.model_id not in seen:
                seen.append(step.model_id)
        return tuple(seen)

    def with_step(self, step: StepTrace) -> RunTrace:
        """Return a copy with one more step appended (traces are immutable)."""
        return replace(self, steps=(*self.steps, step))


@dataclass(frozen=True, slots=True)
class Result:
    """What a pipeline returns: the answer, its confidence, and the full trace."""

    answer: Any
    confidence: float | None
    trace: RunTrace
    abstained: bool = False
    abstention_reason: AbstentionReason | None = None
    schema_valid: bool | None = None

    def unwrap(self) -> Any:
        """Return the answer, raising ``AbstainedError`` if the pipeline declined.

        Use this when a missing answer should be a hard failure; read ``.answer``
        and ``.abstained`` directly when it should not.
        """
        from slmflow.errors import AbstainedError

        if self.abstained:
            raise AbstainedError(
                "Pipeline abstained",
                confidence=self.confidence if self.confidence is not None else 0.0,
                threshold=float("nan"),
                reason=str(self.abstention_reason or AbstentionReason.LOW_CONFIDENCE),
            )
        return self.answer


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    """One decision of a schedule: run this sub-task on this model."""

    step: str
    model_id: str
    load_before: bool
    evict_before: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Schedule:
    """A complete execution plan with its predicted cost and quality.

    Predictions come from measured ``ModelCard`` fields. When a card is
    unprofiled the planner leaves ``predicted_seconds`` as ``None`` rather than
    inventing a number.
    """

    entries: tuple[ScheduleEntry, ...]
    predicted_seconds: float | None = None
    predicted_quality: float | None = None
    predicted_loads: int = 0
    peak_footprint: MemoryFootprint = field(default_factory=MemoryFootprint)
    planner: str = ""

    def __len__(self) -> int:
        """Number of scheduled steps."""
        return len(self.entries)

    def model_for(self, step: str) -> str:
        """Which model this schedule assigns to a step."""
        for entry in self.entries:
            if entry.step == step:
                return entry.model_id
        raise KeyError(f"Step {step!r} is not in this schedule.")


# --------------------------------------------------------------------------- #
# Plugin seams: Protocols, so plugins never import slmflow
# --------------------------------------------------------------------------- #
@runtime_checkable
class Backend(Protocol):
    """A local inference engine.

    Implementations live in ``slmflow.models.backends`` and are lazily imported.
    ``load``/``unload`` must be idempotent; the resource manager owns residency
    and calls them in matched pairs.
    """

    name: str

    def is_available(self) -> bool:
        """True when this backend can run on this machine right now."""
        ...

    def load(self, card: ModelCard) -> None:
        """Make the model resident. Safe to call when already loaded."""
        ...

    def unload(self, card: ModelCard) -> None:
        """Release the model. Safe to call when not loaded."""
        ...

    def is_loaded(self, card: ModelCard) -> bool:
        """True when the model currently occupies memory."""
        ...

    def measure_footprint(self, card: ModelCard) -> MemoryFootprint:
        """Measure bytes actually resident for this model, right now."""
        ...

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
        """Produce one completion and its measured cost."""
        ...


@runtime_checkable
class Router(Protocol):
    """Chooses which model handles a sub-task, given live machine state."""

    name: str

    def select(
        self,
        subtask: SubTask,
        candidates: Sequence[ModelCard],
        *,
        resident: Sequence[str] = (),
    ) -> ModelCard:
        """Pick one model. ``resident`` lets a router prefer an already-warm model."""
        ...


@runtime_checkable
class Strategy(Protocol):
    """How a single sub-task is executed against one or more models."""

    name: str

    def run(self, subtask: SubTask, context: Mapping[str, Any]) -> StepTrace:
        """Execute the sub-task and return its trace."""
        ...


@runtime_checkable
class ReliabilityCheck(Protocol):
    """Scores a set of generations, producing a confidence in ``[0, 1]``."""

    name: str

    def score(self, generations: Sequence[Generation], context: Mapping[str, Any]) -> float:
        """Return a confidence in ``[0, 1]``. Higher means more trustworthy."""
        ...
