"""Shared machinery for execution strategies.

``abc.ABC`` here rather than ``Protocol``, because every strategy shares real
implementation: prompt assembly, going through the resource manager for
residency, schema validation, confidence scoring and trace construction.
Subclasses override only :meth:`BaseStrategy.execute`, which decides *how many*
generations to draw and from *which* models -- the one axis the strategies
actually differ on.

The plugin seam remains :class:`slmflow.types.Strategy` (a Protocol), so a
third-party strategy need not inherit from this at all.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from slmflow.agents.schema import parse_and_validate, schema_to_gbnf
from slmflow.reliability.confidence import CombinedConfidence, modal_answer
from slmflow.resource.residency import ResourceManager
from slmflow.telemetry.trace import get_logger
from slmflow.types import (
    AbstentionReason,
    Generation,
    ModelCard,
    StepTrace,
    SubTask,
)

__all__ = ["BaseStrategy", "StrategyContext", "GenerationBatch"]

_LOG = get_logger("strategy")


@dataclass(slots=True)
class StrategyContext:
    """Everything a strategy needs that is not the sub-task itself."""

    manager: ResourceManager
    pool: Sequence[ModelCard]
    seed: int | None = None
    retrieved: tuple[str, ...] = ()
    upstream: Mapping[str, object] = field(default_factory=dict)
    grammar_cache: dict[str, str] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, object]:
        """The mapping form the reliability checks read."""
        return {"retrieved": list(self.retrieved), **dict(self.upstream)}


@dataclass(slots=True)
class GenerationBatch:
    """Generations drawn at one step, plus the residency cost they incurred."""

    generations: list[Generation] = field(default_factory=list)
    swaps: int = 0
    swap_seconds: float = 0.0
    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0

    def extend(self, other: GenerationBatch) -> None:
        """Merge another batch into this one."""
        self.generations.extend(other.generations)
        self.swaps += other.swaps
        self.swap_seconds += other.swap_seconds
        self.peak_vram_bytes = max(self.peak_vram_bytes, other.peak_vram_bytes)
        self.peak_ram_bytes = max(self.peak_ram_bytes, other.peak_ram_bytes)


class BaseStrategy(ABC):
    """Common behaviour for every execution strategy.

    Args:
        confidence: Reliability check used to score the step. Defaults to
            :meth:`CombinedConfidence.default`.
        use_grammar: Compile the sub-task's output schema to GBNF and constrain
            decoding. Turn off to measure the schema-validity rate *without*
            grammar, which is the Phase 10 before/after comparison.
        temperature: Sampling temperature. Self-consistency needs ``> 0``.
    """

    name: str = "abstract"

    def __init__(
        self,
        *,
        confidence: object | None = None,
        use_grammar: bool = True,
        temperature: float = 0.0,
    ) -> None:
        """Store strategy-wide settings."""
        self.confidence = confidence or CombinedConfidence.default()
        self.use_grammar = use_grammar
        self.temperature = temperature

    # -- the one method subclasses implement -------------------------------- #
    @abstractmethod
    def execute(self, subtask: SubTask, context: StrategyContext) -> GenerationBatch:
        """Draw generations for this sub-task. Subclasses decide how many."""

    # -- shared implementation ---------------------------------------------- #
    def run(self, subtask: SubTask, context: StrategyContext) -> StepTrace:
        """Execute the sub-task and build its trace.

        Wraps :meth:`execute` with timing, schema validation, confidence
        scoring and error capture, so every strategy produces a comparable
        trace and no strategy can forget to record its cost.
        """
        started = time.time()
        clock = time.perf_counter()
        try:
            batch = self.execute(subtask, context)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised by the executor
            return StepTrace(
                step=subtask.name,
                model_id="",
                strategy=self.name,
                error=repr(exc),
                started_at=started,
                wall_s=time.perf_counter() - clock,
                abstained=True,
                abstention_reason=_reason_for(exc),
            )

        ctx = context.as_mapping()
        signals = (
            self.confidence.components(batch.generations, ctx)  # type: ignore[attr-defined]
            if hasattr(self.confidence, "components")
            else {}
        )
        score = float(self.confidence.score(batch.generations, ctx))  # type: ignore[attr-defined]

        schema_ok: bool | None = None
        if subtask.spec.output_schema is not None:
            results = [
                parse_and_validate(g.text, subtask.spec.output_schema)
                for g in batch.generations
            ]
            schema_ok = all(r.valid for r in results) if results else False
            batch.generations = [
                Generation(**{**_as_kwargs(g), "schema_valid": r.valid})
                for g, r in zip(batch.generations, results, strict=True)
            ]

        # A step whose output does not satisfy its declared schema is an
        # abstention, not an answer: returning unparseable text as if it were
        # a result is the failure class Phase 10 exists to remove.
        schema_failed = schema_ok is False
        return StepTrace(
            step=subtask.name,
            model_id=batch.generations[0].model_id if batch.generations else "",
            strategy=self.name,
            generations=tuple(batch.generations),
            confidence=score,
            confidence_signals=signals,
            swaps=batch.swaps,
            swap_seconds=batch.swap_seconds,
            peak_vram_bytes=batch.peak_vram_bytes,
            peak_ram_bytes=batch.peak_ram_bytes,
            abstained=schema_failed,
            abstention_reason=AbstentionReason.SCHEMA_FAILURE if schema_failed else None,
            error=None,
            started_at=started,
            wall_s=time.perf_counter() - clock,
        )

    def sample_once(
        self,
        subtask: SubTask,
        card: ModelCard,
        context: StrategyContext,
        *,
        temperature: float | None = None,
        seed_offset: int = 0,
    ) -> GenerationBatch:
        """Draw exactly one generation, going through the resource manager.

        Every model access in the library funnels through here, which is what
        makes swap accounting complete rather than best-effort.
        """
        grammar = self._grammar_for(subtask, card, context)
        seed = None if context.seed is None else context.seed + seed_offset
        prompt = self.build_prompt(subtask, context)

        with context.manager.resident(card) as handle:
            generation = handle.backend.generate(
                card,
                prompt,
                max_tokens=subtask.spec.max_output_tokens,
                temperature=self.temperature if temperature is None else temperature,
                seed=seed,
                grammar=grammar,
            )
            footprint = context.manager.current_footprint()
            return GenerationBatch(
                generations=[
                    Generation(
                        **{
                            **_as_kwargs(generation),
                            "load_s": handle.load_seconds,
                            "was_cold": handle.was_cold,
                        }
                    )
                ],
                swaps=1 if handle.load_seconds > 0 or handle.evicted else 0,
                swap_seconds=handle.load_seconds,
                peak_vram_bytes=footprint.vram_bytes,
                peak_ram_bytes=footprint.ram_bytes,
            )

    def build_prompt(self, subtask: SubTask, context: StrategyContext) -> str:
        """Assemble the prompt: retrieved evidence, upstream results, question.

        Deliberately plain text with explicit section markers. Small models
        follow positional structure better than they follow instructions, so
        the schema hint goes last, closest to generation.
        """
        parts: list[str] = []
        if context.retrieved:
            parts.append("Evidence:")
            parts.extend(f"[{i + 1}] {p}" for i, p in enumerate(context.retrieved))
            parts.append("")
        for key, value in context.upstream.items():
            if isinstance(value, str) and value:
                parts.append(f"{key}: {value}")
        if parts and parts[-1] != "":
            parts.append("")
        parts.append(subtask.spec.prompt)
        if subtask.spec.output_schema is not None:
            parts.append("")
            parts.append("Reply with JSON only, matching this schema:")
            parts.append(_compact_schema(subtask.spec.output_schema))
        return "\n".join(parts)

    def _grammar_for(
        self, subtask: SubTask, card: ModelCard, context: StrategyContext
    ) -> str | None:
        """Compile and cache the GBNF grammar for this sub-task, if enabled."""
        if not self.use_grammar or subtask.spec.output_schema is None:
            return None
        if not card.supports_grammar:
            return None
        cache_key = subtask.name
        cached = context.grammar_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            grammar = schema_to_gbnf(subtask.spec.output_schema)
        except ValueError as exc:
            _LOG.warning(
                "schema could not be compiled to GBNF; falling back to validation",
                extra={
                    "event": "grammar_compile_failed",
                    "slmflow": {"step": subtask.name, "error": str(exc)},
                },
            )
            return None
        context.grammar_cache[cache_key] = grammar
        return grammar

    def candidates(self, subtask: SubTask, context: StrategyContext) -> list[ModelCard]:
        """Resolve which models this sub-task may use."""
        if subtask.pinned_model:
            return [c for c in context.pool if c.model_id == subtask.pinned_model]
        if subtask.candidate_models:
            allowed = set(subtask.candidate_models)
            return [c for c in context.pool if c.model_id in allowed]
        return list(context.pool)

    @staticmethod
    def final_answer(generations: Sequence[Generation]) -> str:
        """Modal answer across generations, or the first if all differ."""
        answer, count, _ = modal_answer(generations)
        if count > 1:
            for gen in generations:
                from slmflow.reliability.confidence import normalise_answer

                if normalise_answer(gen.text) == answer:
                    return gen.text
        return generations[0].text if generations else ""


def _as_kwargs(generation: Generation) -> dict[str, object]:
    """Explode a frozen ``Generation`` into constructor kwargs."""
    return {
        "text": generation.text,
        "model_id": generation.model_id,
        "prompt_tokens": generation.prompt_tokens,
        "output_tokens": generation.output_tokens,
        "latency_s": generation.latency_s,
        "load_s": generation.load_s,
        "was_cold": generation.was_cold,
        "logprob_sum": generation.logprob_sum,
        "top_token_margin": generation.top_token_margin,
        "finish_reason": generation.finish_reason,
        "schema_valid": generation.schema_valid,
    }


def _compact_schema(schema: Mapping[str, object]) -> str:
    """Render a schema compactly enough to fit a small model's context."""
    import json

    return json.dumps(schema, separators=(",", ":"))


def _reason_for(exc: Exception) -> AbstentionReason:
    """Map an exception to the abstention reason recorded in the trace."""
    from slmflow.errors import ResourceExhausted, SchemaViolation

    if isinstance(exc, ResourceExhausted):
        return AbstentionReason.RESOURCE_EXHAUSTED
    if isinstance(exc, SchemaViolation):
        return AbstentionReason.SCHEMA_FAILURE
    return AbstentionReason.LOW_CONFIDENCE
