"""SLMFlow: memory-bounded orchestration of small language models.

Orchestrating small models on local hardware is a **scheduling problem under a
hard memory budget**. Existing frameworks (LangChain, LangGraph, AutoGen) treat
the model as an always-available remote endpoint, and routing work (RouteLLM,
FrugalGPT) optimises price per API call, where switching models is free. On a
laptop or a phone, switching models is the single most expensive operation in
the system -- and this library is the scheduler for it.

Public API. Everything below is stable; anything else is internal.

    >>> import slmflow
    >>> slmflow.__version__  # doctest: +ELLIPSIS
    '0....'

Quickstart, with no model weights on disk (the ``MockBackend`` path used by the
test suite):

    >>> from slmflow import MemoryBudget, ModelRegistry, Pipeline
    >>> from slmflow.models.backends.mock import MockBackend, make_pool
    >>> registry = ModelRegistry(make_pool(
    ...     [("small", 0.5, 400_000_000), ("large", 3.0, 2_200_000_000)],
    ...     accuracy={"small": {"classify": 0.62}, "large": {"classify": 0.81}},
    ... ))
    >>> registry.register_backend("mock", MockBackend(default_answer="investigational"))
    >>> pipeline = Pipeline.from_dict(
    ...     {
    ...         "name": "quickstart",
    ...         "steps": [{"name": "label", "task_type": "classify", "prompt": "Is it?"}],
    ...     },
    ...     registry=registry,
    ...     budget=MemoryBudget.single_slot(2_500_000_000),
    ... )
    >>> result = pipeline.run()
    >>> result.answer
    'investigational'
    >>> result.trace.total_swaps >= 1
    True

With real weights, declare a precision guarantee and let the framework tune its
own abstention threshold (contribution C3):

    >>> from slmflow import Pipeline, Precision                    # doctest: +SKIP
    >>> pipe = Pipeline.from_config("qa.yaml", guarantee=Precision(0.90))  # doctest: +SKIP
    >>> policy = pipe.calibrate(calibration_items)                 # doctest: +SKIP
    >>> policy.achieved_coverage                                   # doctest: +SKIP

See ``docs/ARCHITECTURE.md`` for the design and ``docs/RESULTS.md`` for
measured numbers. Every number in this repository comes from a run on real
hardware; anything not yet measured is written as ``not yet measured`` rather
than estimated.
"""

from __future__ import annotations

__version__ = "0.1.1"

from slmflow.config.loader import PipelineConfig, StepConfig, load_config, parse_config
from slmflow.device import DeviceCard, probe
from slmflow.errors import (
    AbstainedError,
    BackendUnavailable,
    ConfigError,
    GuaranteeUnmet,
    ModelNotFound,
    ResourceExhausted,
    ScheduleInfeasible,
    SchemaViolation,
    SLMFlowError,
)
from slmflow.models.registry import ModelRegistry
from slmflow.orchestration.pipeline import Pipeline, PipelineOptions, run
from slmflow.reliability.calibration import Precision, ThresholdPolicy
from slmflow.resource.budget import MemoryBudget
from slmflow.resource.residency import ResourceManager
from slmflow.types import (
    DeviceTier,
    MemoryFootprint,
    MemoryTier,
    ModelCard,
    Result,
    RunTrace,
    TaskGraph,
    TaskSpec,
    TaskType,
)

__all__ = [
    # entry points
    "Pipeline",
    "PipelineOptions",
    "run",
    "load_config",
    "parse_config",
    "probe",
    # value objects
    "DeviceCard",
    "DeviceTier",
    "MemoryBudget",
    "MemoryFootprint",
    "MemoryTier",
    "ModelCard",
    "ModelRegistry",
    "PipelineConfig",
    "Precision",
    "ResourceManager",
    "Result",
    "RunTrace",
    "StepConfig",
    "TaskGraph",
    "TaskSpec",
    "TaskType",
    "ThresholdPolicy",
    # errors
    "AbstainedError",
    "BackendUnavailable",
    "ConfigError",
    "GuaranteeUnmet",
    "ModelNotFound",
    "ResourceExhausted",
    "SLMFlowError",
    "ScheduleInfeasible",
    "SchemaViolation",
    "__version__",
]
