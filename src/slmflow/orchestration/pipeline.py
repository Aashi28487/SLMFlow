"""The Pipeline: the one class most users touch.

Assembles device probe, budget, registry, scheduler, router, strategies,
reliability and telemetry into a single object with a sync API and an async
core (``BUILD_BRIEF.md`` section 3: "asyncio for parallel/ensemble execution,
with a sync facade so users are not forced into async").

Concurrency is bounded by the **memory budget**, not by CPU count. Two models
running concurrently must both be resident, and on a 6 GB VRAM machine that is
usually impossible -- so the semaphore is derived from how many candidate models
actually fit together, which is normally one.

The C3 contract lives here: :meth:`Pipeline.calibrate` tunes the abstention
threshold on a labelled split so the declared precision target is met, and
records the coverage achieved.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from slmflow.config.loader import PipelineConfig, StepConfig, load_config
from slmflow.device import ProbeReport, probe
from slmflow.errors import ConfigError, ScheduleInfeasible
from slmflow.models.registry import ModelRegistry
from slmflow.reliability.abstain import AbstentionPolicy, TinyVerifier
from slmflow.reliability.calibration import (
    CalibrationPoint,
    Precision,
    ThresholdPolicy,
    tune_threshold,
)
from slmflow.resource.budget import MemoryBudget
from slmflow.resource.residency import ResourceManager
from slmflow.resource.scheduler import (
    CostModel,
    GreedyScheduler,
    NaiveLoadOnDemandScheduler,
    OracleScheduler,
)
from slmflow.routing.heuristic import HeuristicRouter, RandomRouter, StaticRouter
from slmflow.strategies.adaptive import (
    AdaptiveSelfConsistency,
    CascadeStrategy,
    SingleStrategy,
)
from slmflow.strategies.base import BaseStrategy, StrategyContext
from slmflow.strategies.slm_mux import AgentForestStrategy, SLMMuxStrategy
from slmflow.telemetry.trace import TraceWriter, get_logger
from slmflow.types import (
    AbstentionReason,
    ModelCard,
    Result,
    Schedule,
    StepTrace,
    SubTask,
    TaskGraph,
)

__all__ = ["Pipeline", "PipelineOptions"]

_LOG = get_logger("pipeline")


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    """Knobs that are not part of the config file."""

    trace_path: str | Path | None = None
    seed: int | None = None
    headroom: float = 0.75
    measure_on_load: bool = True
    max_concurrency: int | None = None


class Pipeline:
    """Runs a configured task graph over local models under a memory budget.

    Example:
        >>> from slmflow.models.backends.mock import MockBackend, make_pool
        >>> from slmflow.models.registry import ModelRegistry
        >>> reg = ModelRegistry(make_pool([("tiny", 0.5, 400_000_000)],
        ...                               accuracy={"tiny": {"qa": 0.6}}))
        >>> reg.register_backend("mock", MockBackend(default_answer="ok"))
        >>> pipe = Pipeline.from_dict(
        ...     {"name": "demo", "steps": [{"name": "answer", "prompt": "hi"}]},
        ...     registry=reg,
        ...     budget=MemoryBudget.single_slot(1_000_000_000),
        ... )
        >>> pipe.run().answer
        'ok'
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: ModelRegistry,
        *,
        budget: MemoryBudget | None = None,
        device: ProbeReport | None = None,
        guarantee: Precision | None = None,
        options: PipelineOptions | None = None,
    ) -> None:
        """Assemble a pipeline.

        Args:
            config: Validated pipeline config.
            registry: Model registry with measured cards.
            budget: Explicit memory budget. Falls back to the config's, then to
                one derived from the probed device.
            device: A pre-computed probe, to avoid re-probing per pipeline.
            guarantee: C3 precision contract, overriding the config's.
            options: Trace path, seed, concurrency.
        """
        self.config = config
        self.registry = registry
        self.options = options or PipelineOptions()
        self.device = device or probe()
        self.budget = (
            budget
            or config.budget
            or MemoryBudget.from_device(self.device.card, self.options.headroom)
        )
        self.guarantee = guarantee or config.guarantee
        self.threshold_policy: ThresholdPolicy | None = None

        self.manager = ResourceManager(
            self.budget, measure_on_load=self.options.measure_on_load
        )
        self.pool = self._resolve_pool()
        for card in self.pool:
            self.manager.register(card, registry.backend_for(card))

        self.router = self._build_router()
        self.scheduler = self._build_scheduler()
        self.cost_model = CostModel(cards={c.model_id: c for c in self.pool})
        self._strategies: dict[str, BaseStrategy] = {}

    # -- construction helpers ---------------------------------------------- #
    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        registry: ModelRegistry | None = None,
        guarantee: Precision | None = None,
        options: PipelineOptions | None = None,
        budget: MemoryBudget | None = None,
    ) -> Pipeline:
        """Build from a YAML/JSON file.

        This is the C3 entry point from the brief::

            pipe = Pipeline.from_config("qa.yaml", guarantee=Precision(0.90))
        """
        config = load_config(path)
        return cls(
            config,
            registry or ModelRegistry.from_file(),
            guarantee=guarantee,
            options=options,
            budget=budget,
        )

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        registry: ModelRegistry,
        budget: MemoryBudget | None = None,
        guarantee: Precision | None = None,
        options: PipelineOptions | None = None,
    ) -> Pipeline:
        """Build from an in-memory mapping. Used by tests and examples."""
        from slmflow.config.loader import parse_config

        return cls(
            parse_config(raw),
            registry,
            budget=budget,
            guarantee=guarantee,
            options=options,
        )

    def _resolve_pool(self) -> list[ModelCard]:
        """Models admissible under the budget.

        Raises:
            ScheduleInfeasible: When nothing fits, with the arithmetic shown.
        """
        pool = self.registry.within_budget(self.budget)
        if not pool:
            sizes = {
                c.model_id: round(c.footprint.resident_bytes / 1e9, 2)
                for c in self.registry
            }
            raise ScheduleInfeasible(
                f"No registered model fits {self.budget.describe()}. "
                f"Model sizes (GB): {sizes}. Either raise the budget "
                f"(SLMFLOW_RAM_BUDGET / SLMFLOW_VRAM_BUDGET), or add a smaller "
                f"quantisation to the registry."
            )
        return pool

    def _build_router(self) -> object:
        """Instantiate the configured router."""
        spec = self.config.router
        if spec.startswith("static:"):
            return StaticRouter(spec.split(":", 1)[1])
        if spec == "random":
            return RandomRouter(seed=self.options.seed or self.config.seed)
        if spec == "learned":
            from slmflow.routing.learned import LearnedRouter

            path = self.config.router_path or "benchmarks/results/router.json"
            return LearnedRouter.load(path)
        return HeuristicRouter()

    def _build_scheduler(self) -> object:
        """Instantiate the configured scheduler."""
        if self.config.scheduler == "naive":
            return NaiveLoadOnDemandScheduler()
        if self.config.scheduler == "oracle":
            return OracleScheduler()
        return GreedyScheduler()

    def strategy_for(self, step: StepConfig) -> BaseStrategy:
        """Build (and cache) the strategy object for a step."""
        cached = self._strategies.get(step.name)
        if cached is not None:
            return cached

        common = {"use_grammar": step.use_grammar}
        if step.strategy == "slm_mux":
            strategy: BaseStrategy = SLMMuxStrategy(
                k=step.k, temperature=step.temperature, **common
            )
        elif step.strategy == "agent_forest":
            strategy = AgentForestStrategy(k=step.k, temperature=step.temperature, **common)
        elif step.strategy == "cascade":
            strategy = CascadeStrategy(
                threshold=step.threshold, k=step.k, temperature=step.temperature, **common
            )
        elif step.strategy == "adaptive":
            strategy = AdaptiveSelfConsistency(
                k=step.k, temperature=step.temperature, **common
            )
        else:
            strategy = SingleStrategy(model_id=step.model, **common)

        self._strategies[step.name] = strategy
        return strategy

    # -- planning ----------------------------------------------------------- #
    def plan(self, inputs: Mapping[str, Any] | None = None) -> Schedule:
        """Produce the execution schedule without running it.

        Useful on its own: ``slmflow plan`` prints the residency plan so a user
        can see the swaps a pipeline will pay for before paying for them.
        """
        graph = self.config.graph(inputs)
        return self.scheduler.plan(graph, self.pool, self.budget, self.cost_model)  # type: ignore[attr-defined]

    def concurrency_limit(self) -> int:
        """How many steps may run at once, derived from the memory budget.

        Never CPU count: two concurrent steps need two resident models. When
        the budget is single-slot the limit is 1, and the "parallel" pattern
        degenerates to sequential -- which is the honest behaviour on this
        hardware rather than a queue of stalled tasks.
        """
        if self.options.max_concurrency:
            return max(1, self.options.max_concurrency)
        footprints = sorted(
            (c.footprint for c in self.pool), key=lambda f: f.resident_bytes
        )
        limit = 0
        accumulated = footprints[0].__class__()
        for footprint in footprints:
            candidate = accumulated + footprint
            if not self.budget.fits(candidate):
                break
            accumulated = candidate
            limit += 1
        return max(1, limit)

    # -- execution ---------------------------------------------------------- #
    def run(
        self,
        inputs: Mapping[str, Any] | None = None,
        *,
        retrieved: Sequence[str] = (),
    ) -> Result:
        """Run the pipeline synchronously. The facade most users call."""
        return asyncio.run(self.arun(inputs, retrieved=retrieved))

    async def arun(
        self,
        inputs: Mapping[str, Any] | None = None,
        *,
        retrieved: Sequence[str] = (),
    ) -> Result:
        """Run the pipeline asynchronously.

        Steps within a topological layer run concurrently up to
        :meth:`concurrency_limit`; layers run in order.
        """
        graph = self.config.graph(inputs)
        schedule = self.scheduler.plan(graph, self.pool, self.budget, self.cost_model)  # type: ignore[attr-defined]

        writer = TraceWriter(
            self.options.trace_path,
            pipeline=self.config.name,
            device=str(self.device.card.tier),
            seed=self.options.seed or self.config.seed,
            config_digest=self.config.digest,
        )
        semaphore = asyncio.Semaphore(self.concurrency_limit())
        upstream: dict[str, Any] = {}
        last_trace: StepTrace | None = None

        for layer in graph.topological_layers():
            results = await asyncio.gather(
                *(
                    self._run_step_guarded(node, schedule, upstream, retrieved, semaphore)
                    for node in layer
                ),
                return_exceptions=False,
            )
            for node, step_trace in zip(layer, results, strict=True):
                writer.add_step(step_trace)
                upstream[node.name] = _answer_of(step_trace)
                last_trace = step_trace

        trace = writer.finish()
        return self._finalise(last_trace, trace)

    async def _run_step_guarded(
        self,
        node: SubTask,
        schedule: Schedule,
        upstream: Mapping[str, Any],
        retrieved: Sequence[str],
        semaphore: asyncio.Semaphore,
    ) -> StepTrace:
        """Run one step under the concurrency semaphore."""
        async with semaphore:
            return await asyncio.to_thread(
                self._run_step, node, schedule, upstream, retrieved
            )

    def _run_step(
        self,
        node: SubTask,
        schedule: Schedule,
        upstream: Mapping[str, Any],
        retrieved: Sequence[str],
    ) -> StepTrace:
        """Execute one step: route, generate, score, verify."""
        step_config = self.config.step(node.name)
        strategy = self.strategy_for(step_config)

        # The scheduler chose an assignment; the router refines it against live
        # residency. Both decisions are recorded so the ablation can separate
        # their contributions.
        try:
            scheduled_model = schedule.model_for(node.name)
        except KeyError:
            scheduled_model = ""

        candidates = [c for c in self.pool if c.model_id in (node.candidate_models or ())] or list(
            self.pool
        )
        if node.pinned_model:
            candidates = [c for c in self.pool if c.model_id == node.pinned_model] or candidates

        chosen = self.router.select(  # type: ignore[attr-defined]
            node, candidates, resident=self.manager.resident_models
        )
        effective = node if chosen.model_id == scheduled_model else replace(
            node, pinned_model=chosen.model_id
        )

        context = StrategyContext(
            manager=self.manager,
            pool=self.pool,
            seed=self.options.seed or self.config.seed,
            retrieved=tuple(retrieved),
            upstream=dict(upstream),
        )
        trace = strategy.run(effective, context)

        if step_config.verifier and not trace.abstained and trace.generations:
            verifier = TinyVerifier(step_config.verifier)
            verdict = verifier.verify(
                trace.generations[0].text,
                effective,
                context,
                confidence=trace.confidence or 0.0,
            )
            if not verdict.accepted:
                trace = replace(
                    trace,
                    abstained=True,
                    abstention_reason=AbstentionReason.VERIFIER_REJECTED,
                )

        policy = AbstentionPolicy(
            threshold=step_config.threshold, policy=self.threshold_policy
        )
        decision = policy.decide(
            trace.confidence or 0.0,
            generations=trace.generations,
            signals=trace.confidence_signals,
            schema_valid=None if trace.abstained else True,
        )
        if decision.abstained and not trace.abstained:
            trace = replace(
                trace, abstained=True, abstention_reason=decision.reason
            )
        return trace

    def _finalise(self, last: StepTrace | None, trace: Any) -> Result:
        """Build the ``Result`` from the terminal step."""
        if last is None:
            return Result(answer=None, confidence=None, trace=trace, abstained=True)
        return Result(
            answer=_answer_of(last),
            confidence=last.confidence,
            trace=trace,
            abstained=last.abstained,
            abstention_reason=last.abstention_reason,
            schema_valid=(
                all(g.schema_valid for g in last.generations if g.schema_valid is not None)
                if any(g.schema_valid is not None for g in last.generations)
                else None
            ),
        )

    # -- C3: the declarative contract -------------------------------------- #
    def calibrate(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        gold_key: str = "answer",
        guarantee: Precision | None = None,
    ) -> ThresholdPolicy:
        """Tune the abstention threshold so the precision target is met.

        This is what makes ``guarantee=Precision(0.90)`` more than a label: the
        threshold is fitted on held-out labelled data and the achieved coverage
        is reported, so the user learns the price of their guarantee.

        Args:
            items: Calibration items. Each needs the pipeline's inputs plus a
                gold answer under ``gold_key``.
            gold_key: Key holding the gold answer.
            guarantee: Target, defaulting to the pipeline's.

        Returns:
            The tuned :class:`ThresholdPolicy`, also stored on the pipeline.

        Raises:
            ConfigError: When no guarantee was declared anywhere.
            GuaranteeUnmet: When the target is unreachable on this split.
        """
        target = guarantee or self.guarantee
        if target is None:
            raise ConfigError(
                "calibrate() needs a precision target. Pass guarantee=Precision(0.9), "
                "or set 'guarantee: {precision: 0.9}' in the config.",
                path="guarantee",
            )

        from slmflow.reliability.confidence import normalise_answer

        points: list[CalibrationPoint] = []
        for item in items:
            gold = item.get(gold_key)
            result = self.run({k: v for k, v in item.items() if k != gold_key})
            predicted = "" if result.answer is None else str(result.answer)
            points.append(
                CalibrationPoint(
                    confidence=result.confidence or 0.0,
                    correct=normalise_answer(predicted) == normalise_answer(str(gold)),
                )
            )

        policy = tune_threshold(points, target, signal="pipeline_confidence")
        self.threshold_policy = policy
        _LOG.info(
            "pipeline calibrated",
            extra={"event": "calibrated", "slmflow": policy.as_row()},
        )
        return policy

    # -- introspection ------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        """Everything a results artifact needs to identify this configuration."""
        return {
            "pipeline": self.config.name,
            "config_digest": self.config.digest,
            "router": getattr(self.router, "name", self.config.router),
            "scheduler": getattr(self.scheduler, "name", self.config.scheduler),
            "budget": {
                "vram_bytes": self.budget.vram_bytes,
                "ram_bytes": self.budget.ram_bytes,
            },
            "device_tier": str(self.device.card.tier),
            "device_cpu": self.device.card.cpu_model,
            "device_gpu": self.device.card.gpu_model,
            "models": [c.model_id for c in self.pool],
            "model_digests": {c.model_id: c.digest for c in self.pool},
            "unprofiled_models": self.registry.unprofiled_report(),
            "concurrency_limit": self.concurrency_limit(),
            "guarantee": str(self.guarantee) if self.guarantee else None,
            "threshold_policy": (
                self.threshold_policy.as_row() if self.threshold_policy else None
            ),
            "seed": self.options.seed or self.config.seed,
        }

    def close(self) -> None:
        """Release every resident model."""
        self.manager.evict_all()

    def __enter__(self) -> Pipeline:
        """Enter a scope that frees models on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Free models."""
        self.close()

    def __repr__(self) -> str:
        """Config name, model count and budget."""
        return (
            f"Pipeline({self.config.name!r}, models={len(self.pool)}, "
            f"{self.budget.describe()})"
        )


def _answer_of(trace: StepTrace) -> str:
    """Extract the answer text from a step trace."""
    if not trace.generations:
        return ""
    from slmflow.reliability.confidence import modal_answer, normalise_answer

    answer, count, _ = modal_answer(trace.generations)
    if count > 1:
        for generation in trace.generations:
            if normalise_answer(generation.text) == answer:
                return generation.text
    return trace.generations[0].text


def run(
    config_path: str | Path,
    inputs: Mapping[str, Any] | None = None,
    *,
    registry: ModelRegistry | None = None,
) -> Result:
    """One-call convenience: load a config, run it, return the result.

    Example:
        >>> # run("examples/qa.yaml", {"drug": "metformin"})  # doctest: +SKIP
    """
    started = time.perf_counter()
    pipeline = Pipeline.from_config(config_path, registry=registry)
    try:
        result = pipeline.run(inputs)
    finally:
        pipeline.close()
    _LOG.info(
        "run complete",
        extra={
            "event": "run_complete",
            "slmflow": {"wall_s": round(time.perf_counter() - started, 4)},
        },
    )
    return result
