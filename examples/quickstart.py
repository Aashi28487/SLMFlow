"""Quickstart: the whole library in one file, with no model weights needed.

    python examples/quickstart.py

Runs on ``MockBackend``, so it works on any machine and demonstrates the parts
that matter: a two-tier memory budget, a forced model swap with its cost
charged, resource-aware routing, early-stopping self-consistency, and a
structured trace carrying quality and cost together.

Swap ``MockBackend`` for ``OllamaBackend`` (or profile real cards with
``python -m benchmarks.profile_models``) and the same code runs real models.
"""

from __future__ import annotations

import sys

from slmflow import MemoryBudget, ModelRegistry, Pipeline, PipelineOptions
from slmflow.models.backends.mock import MockBackend, ScriptedResponse, make_pool
from slmflow.resource.scheduler import (
    CostModel,
    GreedyScheduler,
    NaiveLoadOnDemandScheduler,
    OracleScheduler,
    regret,
)

GB = 1_000_000_000


def build_registry() -> ModelRegistry:
    """Two simulated models that disagree, so the ensemble path is exercised."""
    pool = make_pool(
        [("small-0.5b", 0.5, 400_000_000), ("mid-3b", 3.0, 2_200_000_000)],
        accuracy={
            "small-0.5b": {"classify": 0.58, "qa": 0.51},
            "mid-3b": {"classify": 0.79, "qa": 0.70},
        },
        load_s={"small-0.5b": 0.8, "mid-3b": 4.2},
        tps={"small-0.5b": 34.0, "mid-3b": 9.5},
    )
    registry = ModelRegistry(pool)
    registry.register_backend(
        "mock",
        MockBackend(
            {
                "small-0.5b": {
                    "metformin": ScriptedResponse({"investigational": 0.6, "approved": 0.4})
                },
                "mid-3b": {
                    "metformin": ScriptedResponse({"investigational": 0.9, "hypothesis": 0.1})
                },
            },
            default_answer="investigational",
        ),
    )
    return registry


def main() -> int:
    """Run the quickstart."""
    registry = build_registry()

    # A budget that fits either model, but not both at once. This is the T0/T1
    # reality the whole library is designed around: single-slot by default.
    budget = MemoryBudget.single_slot(2_500_000_000)

    pipeline = Pipeline.from_dict(
        {
            "name": "quickstart",
            "router": "heuristic",
            "scheduler": "greedy",
            "seed": 0,
            "steps": [
                {
                    "name": "label",
                    "task_type": "classify",
                    "strategy": "adaptive",
                    "k": 5,
                    "temperature": 0.7,
                    "max_output_tokens": 16,
                    "prompt": (
                        "Classify the repurposing evidence for {drug} in {indication}. "
                        "One of: approved, investigational, hypothesis."
                    ),
                }
            ],
        },
        registry=registry,
        budget=budget,
        options=PipelineOptions(trace_path="runs/quickstart.jsonl", seed=0),
    )

    print(f"budget            {budget.describe()}")
    print(f"models in pool    {[c.model_id for c in pipeline.pool]}")
    print(f"concurrency limit {pipeline.concurrency_limit()}  (bounded by memory, not CPU)")
    print()

    print("--- residency plan, before running anything ---")
    schedule = pipeline.plan({"drug": "metformin", "indication": "cancer"})
    for entry in schedule.entries:
        loads = "loads" if entry.load_before else "warm"
        evicts = f", evicts {list(entry.evict_before)}" if entry.evict_before else ""
        print(f"  {entry.step:<10} -> {entry.model_id:<12} ({loads}{evicts})")
    print(f"  predicted: {schedule.predicted_loads} loads, "
          f"{schedule.predicted_seconds:.2f} s")
    print()

    print("--- run ---")
    with pipeline:
        result = pipeline.run({"drug": "metformin", "indication": "cancer"})

    print(f"answer            {result.answer}")
    print(f"confidence        {result.confidence:.3f}")
    print(f"abstained         {result.abstained}")
    trace = result.trace
    print(f"samples drawn     {trace.total_generations}  (of k=5: early stopping is C4)")
    print(f"model loads       {trace.total_swaps}, {trace.swap_seconds:.2f} s")
    print(f"peak memory       vram {trace.peak_vram_bytes / GB:.2f} GB, "
          f"ram {trace.peak_ram_bytes / GB:.2f} GB")
    print(f"trace written     runs/quickstart.jsonl")
    print()

    print("--- C1: scheduler regret on a small instance ---")
    from slmflow.types import SubTask, TaskGraph, TaskSpec, TaskType

    graph = TaskGraph(
        nodes=tuple(
            SubTask(
                name=f"s{i}",
                spec=TaskSpec(
                    prompt=f"sub-task {i}",
                    task_type=[TaskType.CLASSIFY, TaskType.QA][i % 2],
                    task_id=f"s{i}",
                ),
            )
            for i in range(3)
        )
    )
    cost = CostModel(cards={c.model_id: c for c in pipeline.pool}, lambda_cost=0.01)
    greedy = GreedyScheduler().plan(graph, pipeline.pool, budget, cost)
    naive = NaiveLoadOnDemandScheduler().plan(graph, pipeline.pool, budget, cost)
    oracle = OracleScheduler().plan(graph, pipeline.pool, budget, cost)

    gap = regret(greedy, oracle)
    print(f"  greedy   {greedy.predicted_loads} loads, {greedy.predicted_seconds:.2f} s")
    print(f"  naive    {naive.predicted_loads} loads, {naive.predicted_seconds:.2f} s")
    print(f"  oracle   {oracle.predicted_loads} loads, {oracle.predicted_seconds:.2f} s")
    print(f"  scheduler regret: {gap.seconds_regret:+.3f} s, "
          f"quality {gap.quality_regret:+.4f}, loads {gap.load_regret:+d}")
    print()
    print("NOTE: every number above comes from MockBackend, not a real model.")
    print("It demonstrates the machinery. Real numbers require Phase 2 profiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
