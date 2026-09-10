"""The C1 headline: scheduler regret, and the swap-awareness ablation.

Three numbers this script produces, all on the same instances:

1. **Scheduler regret** -- greedy swap-aware versus the offline oracle.
2. **The swap-awareness ablation** -- greedy versus naive load-on-demand+LRU.
   ``BUILD_BRIEF.md`` C1 demands this be reported "even if the answer is
   'little, because model loads are amortised' -- that is a finding".
3. **Router regret** -- the heuristic router versus hindsight-optimal choice.

Instances are small by necessity: the oracle enumerates every topological order
crossed with every assignment, so it is capped and refuses rather than silently
returning a non-optimal plan.

Usage::

    python -m benchmarks.regret --cards benchmarks/results/model_cards.json
    python -m benchmarks.regret --mock --sizes 2,3,4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slmflow.device import probe
from slmflow.errors import ScheduleInfeasible
from slmflow.models.registry import ModelRegistry
from slmflow.resource.budget import MemoryBudget
from slmflow.resource.scheduler import (
    CostModel,
    GreedyScheduler,
    NaiveLoadOnDemandScheduler,
    OracleScheduler,
    regret,
)
from slmflow.types import SubTask, TaskGraph, TaskSpec, TaskType

__all__ = ["build_instance", "measure", "main"]


def build_instance(
    n_steps: int, task_types: Sequence[TaskType], *, chain: bool = False
) -> TaskGraph:
    """Build a synthetic DAG of ``n_steps`` sub-tasks.

    Args:
        n_steps: Number of sub-tasks.
        task_types: Cycled to assign a type per step, so different steps favour
            different models and the assignment problem is non-trivial.
        chain: When true, each step depends on the previous, which removes the
            scheduler's freedom to reorder. The difference between chained and
            unchained regret isolates how much of the gain comes from
            reordering versus from assignment.

    Returns:
        A :class:`TaskGraph`.
    """
    nodes: list[SubTask] = []
    for index in range(n_steps):
        task_type = task_types[index % len(task_types)]
        nodes.append(
            SubTask(
                name=f"s{index}",
                spec=TaskSpec(
                    prompt=f"synthetic sub-task {index} of type {task_type}",
                    task_type=task_type,
                    task_id=f"s{index}",
                    max_output_tokens=64,
                ),
                depends_on=(f"s{index - 1}",) if chain and index else (),
            )
        )
    return TaskGraph(nodes=tuple(nodes))


def measure(
    registry: ModelRegistry,
    budget: MemoryBudget,
    *,
    sizes: Sequence[int] = (2, 3, 4),
    lambda_cost: float = 0.01,
    max_instances: int = 200_000,
) -> dict[str, Any]:
    """Measure regret and the ablation across instance sizes."""
    pool = registry.within_budget(budget)
    cost = CostModel(
        cards={c.model_id: c for c in pool}, lambda_cost=lambda_cost
    )
    unprofiled = registry.unprofiled_report()

    greedy = GreedyScheduler()
    greedy_no_group = GreedyScheduler(group_by_model=False)
    naive = NaiveLoadOnDemandScheduler()
    oracle = OracleScheduler(max_instances=max_instances)

    task_types = (TaskType.CLASSIFY, TaskType.EXTRACT, TaskType.QA, TaskType.SUMMARISE)
    rows: list[dict[str, Any]] = []

    for size in sizes:
        for chain in (False, True):
            graph = build_instance(size, task_types, chain=chain)
            row: dict[str, Any] = {
                "n_steps": size,
                "shape": "chain" if chain else "independent",
                "n_models": len(pool),
            }
            try:
                plan_greedy = greedy.plan(graph, pool, budget, cost)
                plan_nogroup = greedy_no_group.plan(graph, pool, budget, cost)
                plan_naive = naive.plan(graph, pool, budget, cost)
            except ScheduleInfeasible as exc:
                row["error"] = str(exc)
                rows.append(row)
                continue

            row["greedy"] = _plan_row(plan_greedy)
            row["greedy_no_grouping"] = _plan_row(plan_nogroup)
            row["naive_lru"] = _plan_row(plan_naive)

            try:
                plan_oracle = oracle.plan(graph, pool, budget, cost)
            except ScheduleInfeasible as exc:
                row["oracle"] = None
                row["oracle_skipped"] = str(exc)
            else:
                row["oracle"] = _plan_row(plan_oracle)
                row["scheduler_regret"] = regret(plan_greedy, plan_oracle).as_row()
                row["naive_regret"] = regret(plan_naive, plan_oracle).as_row()

            # The ablation: what does swap-awareness buy over naive LRU?
            row["ablation_swap_awareness"] = {
                "greedy_loads": plan_greedy.predicted_loads,
                "naive_loads": plan_naive.predicted_loads,
                "loads_saved": plan_naive.predicted_loads - plan_greedy.predicted_loads,
                "greedy_seconds": plan_greedy.predicted_seconds,
                "naive_seconds": plan_naive.predicted_seconds,
                "seconds_saved": (
                    None
                    if plan_naive.predicted_seconds is None
                    or plan_greedy.predicted_seconds is None
                    else plan_naive.predicted_seconds - plan_greedy.predicted_seconds
                ),
                "grouping_loads_saved": (
                    plan_nogroup.predicted_loads - plan_greedy.predicted_loads
                ),
            }
            rows.append(row)

    return {
        "budget": {"vram_bytes": budget.vram_bytes, "ram_bytes": budget.ram_bytes},
        "models": [c.model_id for c in pool],
        "unprofiled_models": unprofiled,
        "lambda_cost": lambda_cost,
        "instances": rows,
        "caveat": (
            "Predicted seconds come from measured ModelCard throughput and load times, "
            "so they are a model-based prediction, not an end-to-end wall-clock "
            "measurement. Wall-clock regret requires running both schedules, which "
            "benchmarks/harness.py does via the ablate_scheduler configuration."
            if not unprofiled
            else "SOME MODELS ARE UNPROFILED: regret is reported as 'not measured' for "
            "any instance involving them. Run benchmarks/profile_models.py first."
        ),
    }


def _plan_row(plan: Any) -> dict[str, Any]:
    """Flatten a schedule for the report."""
    return {
        "planner": plan.planner,
        "loads": plan.predicted_loads,
        "seconds": plan.predicted_seconds,
        "quality": plan.predicted_quality,
        "peak_vram_bytes": plan.peak_footprint.vram_bytes,
        "peak_ram_bytes": plan.peak_footprint.ram_bytes,
        "order": [e.step for e in plan.entries],
        "assignment": {e.step: e.model_id for e in plan.entries},
    }


def render(report: dict[str, Any]) -> str:
    """Render the regret tables as markdown for ``docs/RESULTS.md``."""
    lines = ["## C1: scheduler regret and the swap-awareness ablation", ""]
    lines.append(f"Budget: VRAM {report['budget']['vram_bytes'] / 1e9:.2f} GB, "
                 f"RAM {report['budget']['ram_bytes'] / 1e9:.2f} GB. "
                 f"Models: {', '.join(report['models']) or 'none'}.")
    lines.append("")
    lines.append(f"> {report['caveat']}")
    lines.append("")
    lines.append("| steps | shape | greedy loads | naive loads | loads saved | "
                 "oracle loads | seconds regret | relative | quality regret |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in report["instances"]:
        if "error" in row:
            lines.append(
                f"| {row['n_steps']} | {row['shape']} | infeasible: {row['error'][:60]} | | | | | | |"
            )
            continue
        ablation = row["ablation_swap_awareness"]
        sched = row.get("scheduler_regret") or {}
        oracle = row.get("oracle") or {}
        lines.append(
            f"| {row['n_steps']} | {row['shape']} | {ablation['greedy_loads']} | "
            f"{ablation['naive_loads']} | {ablation['loads_saved']} | "
            f"{oracle.get('loads', 'skipped')} | "
            f"{sched.get('seconds_regret', 'not measured')} | "
            f"{sched.get('relative_seconds_regret', 'not measured')} | "
            f"{sched.get('quality_regret', 'not measured')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure regret and write the report."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.regret",
        description="Measure scheduler regret and the swap-awareness ablation (C1).",
    )
    parser.add_argument("--cards", default="benchmarks/results/model_cards.json")
    parser.add_argument("--sizes", default="2,3,4", help="comma-separated DAG sizes")
    parser.add_argument("--lambda-cost", type=float, default=0.01)
    parser.add_argument("--max-instances", type=int, default=200_000)
    parser.add_argument("--out", default="benchmarks/results/regret.json")
    parser.add_argument("--markdown", default=None, help="also write a markdown table here")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    if args.mock:
        from benchmarks.harness import build_registry

        registry = build_registry(mock=True, cards=args.cards)
    else:
        registry = ModelRegistry.from_file(args.cards)

    device = probe()
    budget = MemoryBudget.from_device(device.card)
    sizes = tuple(int(s) for s in args.sizes.split(",") if s.strip())

    report = measure(
        registry,
        budget,
        sizes=sizes,
        lambda_cost=args.lambda_cost,
        max_instances=args.max_instances,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    sys.stdout.write(render(report) + "\n")
    sys.stdout.write(f"wrote {out}\n")
    if args.markdown:
        Path(args.markdown).write_text(render(report), encoding="utf-8")
        sys.stdout.write(f"wrote {args.markdown}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
