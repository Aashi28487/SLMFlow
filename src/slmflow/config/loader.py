"""Pipeline configuration: schema, validation, and good error messages.

A config error must say what to change, on which key, with the valid options
listed -- a research framework whose config errors are cryptic wastes the
researcher's time, which is the scarcest resource in the project.

Config shape::

    name: repurposing_qa
    budget:
      vram_gb: 5.0          # optional; probed defaults are used when absent
      ram_gb: 8.0
    router: heuristic        # heuristic | learned | random | static:<model_id>
    scheduler: greedy        # greedy | naive | oracle
    guarantee:
      precision: 0.90        # optional; enables the C3 contract
      delta: 0.1
    defaults:
      strategy: single
      max_output_tokens: 64
    steps:
      - name: classify
        task_type: classify
        prompt: |
          Is there evidence that {drug} has an effect beyond its approved use?
        strategy: adaptive        # single|slm_mux|agent_forest|cascade|adaptive
        k: 5
        temperature: 0.7
        candidate_models: [qwen2.5-0.5b, llama3.2-3b]
        output_schema:
          type: object
          properties:
            label: {type: string, enum: [approved, investigational, hypothesis]}
          required: [label]
        depends_on: []
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slmflow.config.minyaml import load_yaml_file, parse_yaml
from slmflow.errors import ConfigError
from slmflow.reliability.calibration import Precision
from slmflow.resource.budget import MemoryBudget
from slmflow.types import ExecutionPattern, SubTask, TaskGraph, TaskSpec, TaskType

__all__ = ["PipelineConfig", "StepConfig", "load_config", "parse_config"]

_STRATEGIES = ("single", "slm_mux", "agent_forest", "cascade", "adaptive")
_SCHEDULERS = ("greedy", "naive", "oracle")
_ROUTERS = ("heuristic", "learned", "random")


@dataclass(frozen=True, slots=True)
class StepConfig:
    """One configured step."""

    name: str
    prompt: str
    task_type: TaskType = TaskType.QA
    strategy: str = "single"
    k: int = 5
    temperature: float = 0.0
    model: str | None = None
    candidate_models: tuple[str, ...] = ()
    output_schema: Mapping[str, Any] | None = None
    depends_on: tuple[str, ...] = ()
    max_output_tokens: int = 256
    use_grammar: bool = True
    verifier: str | None = None
    threshold: float = 0.5

    def to_subtask(self, inputs: Mapping[str, Any] | None = None) -> SubTask:
        """Build a :class:`SubTask`, formatting ``{placeholders}`` from ``inputs``.

        Raises:
            ConfigError: When the prompt references a key ``inputs`` lacks. A
                silently-unformatted prompt would send ``{drug}`` to the model.
        """
        values = dict(inputs or {})
        try:
            prompt = self.prompt.format(**values) if values else self.prompt
        except KeyError as exc:
            raise ConfigError(
                f"Step {self.name!r} prompt references {exc.args[0]!r}, which was not "
                f"supplied. Available inputs: {sorted(values)}.",
                path=f"steps.{self.name}.prompt",
            ) from exc
        except (IndexError, ValueError) as exc:
            raise ConfigError(
                f"Step {self.name!r} prompt has malformed braces: {exc}. "
                f"Escape a literal brace as {{{{ or }}}}.",
                path=f"steps.{self.name}.prompt",
            ) from exc

        return SubTask(
            name=self.name,
            spec=TaskSpec(
                prompt=prompt,
                task_type=self.task_type,
                inputs=values,
                output_schema=self.output_schema,
                max_output_tokens=self.max_output_tokens,
            ),
            depends_on=self.depends_on,
            candidate_models=self.candidate_models,
            pinned_model=self.model,
        )


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """A validated pipeline configuration."""

    name: str
    steps: tuple[StepConfig, ...]
    router: str = "heuristic"
    scheduler: str = "greedy"
    pattern: ExecutionPattern = ExecutionPattern.SEQUENTIAL
    budget: MemoryBudget | None = None
    guarantee: Precision | None = None
    seed: int | None = None
    router_path: str | None = None
    source_path: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Stable hash of the raw config, recorded in every trace.

        Ground Rule 11: a results artifact must identify exactly which config
        produced it.
        """
        payload = json.dumps(self.raw, sort_keys=True, default=str)
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()

    def graph(self, inputs: Mapping[str, Any] | None = None) -> TaskGraph:
        """Build the task DAG, validating dependencies via ``TaskGraph``."""
        try:
            return TaskGraph(nodes=tuple(s.to_subtask(inputs) for s in self.steps))
        except ValueError as exc:
            raise ConfigError(f"Invalid step graph in pipeline {self.name!r}: {exc}") from exc

    def step(self, name: str) -> StepConfig:
        """Look up one step config."""
        for step in self.steps:
            if step.name == name:
                return step
        raise ConfigError(
            f"No step named {name!r} in pipeline {self.name!r}. "
            f"Steps: {[s.name for s in self.steps]}."
        )


def load_config(path: str | Path, **overrides: Any) -> PipelineConfig:
    """Load and validate a pipeline config from a YAML or JSON file."""
    p = Path(path)
    raw = load_yaml_file(p) if p.suffix in {".yaml", ".yml"} else json.loads(
        p.read_text(encoding="utf-8")
    )
    config = parse_config(raw, source_path=str(p))
    if overrides:
        from dataclasses import replace

        return replace(config, **overrides)
    return config


def parse_config(raw: Any, *, source_path: str | None = None) -> PipelineConfig:
    """Validate a parsed config mapping.

    Raises:
        ConfigError: With the offending key path and the valid options.
    """
    if isinstance(raw, str):
        raw = parse_yaml(raw)
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"A pipeline config must be a mapping at the top level, got "
            f"{type(raw).__name__}.",
            path=source_path,
        )

    name = str(raw.get("name") or "unnamed")

    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, str) or not steps_raw:
        raise ConfigError(
            "A pipeline needs a non-empty 'steps' list.", path="steps"
        )

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise ConfigError("'defaults' must be a mapping.", path="defaults")

    steps = tuple(
        _parse_step(entry, index, defaults) for index, entry in enumerate(steps_raw)
    )

    router = str(raw.get("router") or "heuristic")
    if not (router in _ROUTERS or router.startswith("static:")):
        raise ConfigError(
            f"Unknown router {router!r}. Valid: {', '.join(_ROUTERS)}, "
            f"or 'static:<model_id>'.",
            path="router",
        )

    scheduler = str(raw.get("scheduler") or "greedy")
    if scheduler not in _SCHEDULERS:
        raise ConfigError(
            f"Unknown scheduler {scheduler!r}. Valid: {', '.join(_SCHEDULERS)}. "
            f"Note 'oracle' is a measurement tool and is too slow for production use.",
            path="scheduler",
        )

    pattern_raw = str(raw.get("pattern") or "sequential")
    try:
        pattern = ExecutionPattern(pattern_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Unknown pattern {pattern_raw!r}. Valid: "
            f"{', '.join(p.value for p in ExecutionPattern)}.",
            path="pattern",
        ) from exc

    return PipelineConfig(
        name=name,
        steps=steps,
        router=router,
        scheduler=scheduler,
        pattern=pattern,
        budget=_parse_budget(raw.get("budget")),
        guarantee=_parse_guarantee(raw.get("guarantee")),
        seed=_optional_int(raw.get("seed"), "seed"),
        router_path=(str(raw["router_path"]) if raw.get("router_path") else None),
        source_path=source_path,
        raw=dict(raw),
    )


def _parse_step(entry: Any, index: int, defaults: Mapping[str, Any]) -> StepConfig:
    """Validate one step entry."""
    where = f"steps[{index}]"
    if not isinstance(entry, Mapping):
        raise ConfigError(f"Each step must be a mapping, got {type(entry).__name__}.", path=where)

    name = entry.get("name") or f"step_{index}"
    where = f"steps.{name}"

    prompt = entry.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigError(
            "Every step needs a non-empty 'prompt' string.", path=f"{where}.prompt"
        )

    task_type_raw = str(entry.get("task_type") or defaults.get("task_type") or "qa")
    try:
        task_type = TaskType(task_type_raw)
    except ValueError as exc:
        raise ConfigError(
            f"Unknown task_type {task_type_raw!r}. Valid: "
            f"{', '.join(t.value for t in TaskType)}.",
            path=f"{where}.task_type",
        ) from exc

    strategy = str(entry.get("strategy") or defaults.get("strategy") or "single")
    if strategy not in _STRATEGIES:
        raise ConfigError(
            f"Unknown strategy {strategy!r}. Valid: {', '.join(_STRATEGIES)}.",
            path=f"{where}.strategy",
        )

    temperature = float(entry.get("temperature", defaults.get("temperature", 0.0)))
    k = int(entry.get("k", defaults.get("k", 5)))
    if strategy in {"slm_mux", "agent_forest", "adaptive"} and temperature <= 0.0:
        raise ConfigError(
            f"Strategy {strategy!r} samples repeatedly, so it needs temperature > 0. "
            f"At temperature 0 every sample is identical and the confidence signal is "
            f"vacuously 1.0. Set temperature (0.7 is a reasonable start).",
            path=f"{where}.temperature",
        )
    if strategy in {"slm_mux", "agent_forest"} and k < 2:
        raise ConfigError(
            f"Strategy {strategy!r} needs k >= 2, got {k}.", path=f"{where}.k"
        )

    schema = entry.get("output_schema")
    if schema is not None and not isinstance(schema, Mapping):
        raise ConfigError(
            "'output_schema' must be a mapping (a JSON schema).",
            path=f"{where}.output_schema",
        )

    return StepConfig(
        name=str(name),
        prompt=prompt,
        task_type=task_type,
        strategy=strategy,
        k=k,
        temperature=temperature,
        model=(str(entry["model"]) if entry.get("model") else None),
        candidate_models=_string_tuple(entry.get("candidate_models"), f"{where}.candidate_models"),
        output_schema=dict(schema) if schema else None,
        depends_on=_string_tuple(entry.get("depends_on"), f"{where}.depends_on"),
        max_output_tokens=int(
            entry.get("max_output_tokens", defaults.get("max_output_tokens", 256))
        ),
        use_grammar=bool(entry.get("use_grammar", defaults.get("use_grammar", True))),
        verifier=(str(entry["verifier"]) if entry.get("verifier") else None),
        threshold=float(entry.get("threshold", defaults.get("threshold", 0.5))),
    )


def _string_tuple(value: Any, where: str) -> tuple[str, ...]:
    """Coerce to a tuple of strings, accepting a bare string."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value)
    raise ConfigError(f"Expected a list of strings, got {type(value).__name__}.", path=where)


def _parse_budget(value: Any) -> MemoryBudget | None:
    """Parse the optional memory budget.

    Accepts ``vram_gb``/``ram_gb`` (convenient) or ``vram_bytes``/``ram_bytes``
    (exact). Returning ``None`` means "probe the device", which is the right
    default because a hard-coded budget in a config is not portable across
    the hardware tiers C2 is about.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError("'budget' must be a mapping.", path="budget")

    def resolve(gb_key: str, byte_key: str) -> int:
        if byte_key in value:
            return int(value[byte_key])
        if gb_key in value:
            return int(float(value[gb_key]) * 1_000_000_000)
        return 0

    vram = resolve("vram_gb", "vram_bytes")
    ram = resolve("ram_gb", "ram_bytes")
    if vram == 0 and ram == 0:
        raise ConfigError(
            "'budget' was given but sets neither VRAM nor RAM. Use vram_gb/ram_gb, "
            "or omit 'budget' entirely to probe this machine.",
            path="budget",
        )
    return MemoryBudget(vram_bytes=vram, ram_bytes=ram)


def _parse_guarantee(value: Any) -> Precision | None:
    """Parse the optional C3 precision contract."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return Precision(float(value))
    if not isinstance(value, Mapping):
        raise ConfigError(
            "'guarantee' must be a number or a mapping with 'precision'.", path="guarantee"
        )
    if "precision" not in value:
        raise ConfigError(
            "'guarantee' needs a 'precision' target, e.g. precision: 0.90. "
            "Only precision guarantees are implemented (C3).",
            path="guarantee.precision",
        )
    try:
        return Precision(float(value["precision"]), delta=float(value.get("delta", 0.1)))
    except ValueError as exc:
        raise ConfigError(str(exc), path="guarantee") from exc


def _optional_int(value: Any, where: str) -> int | None:
    """Parse an optional integer."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Expected an integer, got {value!r}.", path=where) from exc
