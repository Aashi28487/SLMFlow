"""MBOP solvers: the greedy swap-aware scheduler and the offline oracle.

The Memory-Bounded Orchestration Problem (``BUILD_BRIEF.md`` section 0, C1).
Given a task DAG ``T``, a model pool ``M`` with measured profiles, and a hard
per-tier budget ``B``, produce an assignment of sub-tasks to models, an
ordering, and a residency plan that maximises expected quality subject to
``peak_resident_bytes <= B`` per tier.

Two solvers, as the brief requires:

* :class:`GreedyScheduler` -- ships by default, runs online, O(n * k).
* :class:`OracleScheduler` -- exhaustive over assignments and topological
  orders, used only for measurement on small instances.

The headline number is **scheduler regret**: the cost/quality gap between the
two on real runs. :func:`regret` computes it. When a model card is unprofiled
the cost model returns ``None`` and regret is reported as unavailable rather
than as zero (Ground Rule 1).

Formal statement, for ``docs/ARCHITECTURE.md`` and ``docs/RESULTS.md``:

    maximise   sum_i q(t_i, m(t_i))  -  lambda * cost(schedule)
    subject to peak_resident_bytes[tier] <= B[tier]  for every tier
    where     cost(schedule) = sum over steps of (load_s if cold else 0)
                               + generate_s(t_i, m(t_i))

This is a paging problem (variable page sizes, variable miss costs) fused with
an assignment problem. Both structures are classical; the fusion under a hard
ceiling is what has no existing solver.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from slmflow.errors import ScheduleInfeasible
from slmflow.resource.budget import RESIDENT_TIERS, MemoryBudget
from slmflow.types import (
    MemoryFootprint,
    ModelCard,
    Schedule,
    ScheduleEntry,
    SubTask,
    TaskGraph,
)

__all__ = [
    "CostModel",
    "GreedyScheduler",
    "OracleScheduler",
    "NaiveLoadOnDemandScheduler",
    "SchedulerRegret",
    "regret",
    "simulate",
]


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CostModel:
    """Turns measured model cards into schedule costs.

    ``lambda_cost`` trades quality against seconds. It is a declared
    preference, not a measured quantity, so it lives here rather than on a card.
    """

    cards: Mapping[str, ModelCard]
    lambda_cost: float = 0.0
    default_prompt_tokens: int = 256

    def generate_seconds(self, subtask: SubTask, model_id: str) -> float | None:
        """Predicted generation seconds, or ``None`` if the model is unprofiled."""
        card = self.cards.get(model_id)
        if card is None:
            return None
        return card.expected_seconds(
            self.default_prompt_tokens, subtask.spec.max_output_tokens
        )

    def load_seconds(self, model_id: str, *, cold: bool) -> float | None:
        """Measured load cost, or ``None`` if never measured."""
        card = self.cards.get(model_id)
        if card is None:
            return None
        if cold:
            return card.cold_load_s
        return card.warm_load_s if card.warm_load_s is not None else card.cold_load_s

    def quality(self, subtask: SubTask, model_id: str) -> float | None:
        """Measured accuracy of this model on this task type, or ``None``."""
        card = self.cards.get(model_id)
        if card is None:
            return None
        return card.accuracy_for(subtask.spec.task_type)

    def footprint(self, model_id: str) -> MemoryFootprint:
        """Declared footprint of a model, or an empty footprint if unknown."""
        card = self.cards.get(model_id)
        return card.footprint if card else MemoryFootprint()

    def objective(self, quality: float | None, seconds: float | None) -> float | None:
        """Combine quality and cost into the scalar the solvers maximise."""
        if quality is None:
            return None
        if self.lambda_cost == 0.0:
            return quality
        if seconds is None:
            return None
        return quality - self.lambda_cost * seconds


# --------------------------------------------------------------------------- #
# Simulation: shared by every solver so they are compared on identical rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SimResult:
    """Outcome of replaying a schedule against the cost model and the budget."""

    seconds: float | None
    quality: float | None
    loads: int
    peak: MemoryFootprint
    feasible: bool
    violated_tier: str | None = None


def simulate(
    order: Sequence[SubTask],
    assignment: Mapping[str, str],
    budget: MemoryBudget,
    cost: CostModel,
) -> SimResult:
    """Replay an ordering plus assignment, charging loads and checking the budget.

    Residency is single-slot-aware: a model is kept resident while it fits, and
    the least-recently-used resident model is dropped when it does not. Every
    solver is scored through this one function, so a comparison between them is
    a comparison of decisions rather than of accounting.

    Args:
        order: Sub-tasks in the execution order to evaluate.
        assignment: Sub-task name to model id.
        budget: Hard per-tier ceiling.
        cost: Measured cost model.

    Returns:
        A :class:`SimResult`. ``seconds``/``quality`` are ``None`` when any
        model involved is unprofiled.
    """
    resident: list[str] = []
    loads = 0
    seconds: float | None = 0.0
    quality: float | None = 0.0
    peak = MemoryFootprint()
    ever_loaded: set[str] = set()

    def used(models: Iterable[str]) -> MemoryFootprint:
        total = MemoryFootprint()
        for m in models:
            total = total + cost.footprint(m)
        return total

    for subtask in order:
        model_id = assignment[subtask.name]
        need = cost.footprint(model_id)

        if not budget.fits(need):
            worst = max(RESIDENT_TIERS, key=lambda t: need.get(t) - budget.get(t))
            return SimResult(None, None, loads, peak, False, str(worst))

        if model_id not in resident:
            # Evict LRU until it fits.
            while resident:
                trial = used([*resident, model_id])
                if all(trial.get(t) <= budget.get(t) for t in RESIDENT_TIERS):
                    break
                resident.pop(0)
            cold = model_id not in ever_loaded
            load_s = cost.load_seconds(model_id, cold=cold)
            loads += 1
            ever_loaded.add(model_id)
            resident.append(model_id)
            seconds = None if (seconds is None or load_s is None) else seconds + load_s
        else:
            resident.remove(model_id)
            resident.append(model_id)

        snapshot = used(resident)
        peak = MemoryFootprint(
            vram_bytes=max(peak.vram_bytes, snapshot.vram_bytes),
            ram_bytes=max(peak.ram_bytes, snapshot.ram_bytes),
        )
        for tier in RESIDENT_TIERS:
            if snapshot.get(tier) > budget.get(tier):  # pragma: no cover - eviction prevents this
                return SimResult(None, None, loads, peak, False, str(tier))

        gen_s = cost.generate_seconds(subtask, model_id)
        seconds = None if (seconds is None or gen_s is None) else seconds + gen_s
        q = cost.quality(subtask, model_id)
        quality = None if (quality is None or q is None) else quality + q

    return SimResult(seconds, quality, loads, peak, True, None)


def _schedule_from(
    order: Sequence[SubTask],
    assignment: Mapping[str, str],
    budget: MemoryBudget,
    cost: CostModel,
    planner: str,
) -> Schedule:
    """Build a :class:`Schedule` value object from an order plus assignment."""
    sim = simulate(order, assignment, budget, cost)
    resident: list[str] = []
    entries: list[ScheduleEntry] = []

    def used(models: Iterable[str]) -> MemoryFootprint:
        total = MemoryFootprint()
        for m in models:
            total = total + cost.footprint(m)
        return total

    for subtask in order:
        model_id = assignment[subtask.name]
        evicted: list[str] = []
        load_before = model_id not in resident
        if load_before:
            while resident:
                trial = used([*resident, model_id])
                if all(trial.get(t) <= budget.get(t) for t in RESIDENT_TIERS):
                    break
                evicted.append(resident.pop(0))
            resident.append(model_id)
        else:
            resident.remove(model_id)
            resident.append(model_id)
        entries.append(
            ScheduleEntry(
                step=subtask.name,
                model_id=model_id,
                load_before=load_before,
                evict_before=tuple(evicted),
            )
        )
    n = len(order)
    return Schedule(
        entries=tuple(entries),
        predicted_seconds=sim.seconds,
        predicted_quality=(sim.quality / n if sim.quality is not None and n else None),
        predicted_loads=sim.loads,
        peak_footprint=sim.peak,
        planner=planner,
    )


def _candidates_for(subtask: SubTask, pool: Sequence[ModelCard]) -> list[ModelCard]:
    """Resolve which models may run a sub-task."""
    if subtask.pinned_model:
        return [c for c in pool if c.model_id == subtask.pinned_model]
    if subtask.candidate_models:
        allowed = set(subtask.candidate_models)
        return [c for c in pool if c.model_id in allowed]
    return list(pool)


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
class NaiveLoadOnDemandScheduler:
    """The ablation baseline: best model per task, load on demand, evict LRU.

    This is what a framework without a scheduler does. The C1 ablation asks
    what swap-awareness buys over exactly this.
    """

    name = "naive_lru"

    def plan(
        self, graph: TaskGraph, pool: Sequence[ModelCard], budget: MemoryBudget, cost: CostModel
    ) -> Schedule:
        """Assign the highest-quality feasible model to each task, ignoring swaps."""
        order = graph.linearise()
        assignment: dict[str, str] = {}
        for subtask in order:
            feasible = [
                c for c in _candidates_for(subtask, pool) if budget.fits(c.footprint)
            ]
            if not feasible:
                raise ScheduleInfeasible(
                    f"No model fits the budget for sub-task {subtask.name!r}. "
                    f"{budget.describe()}; smallest candidate needs "
                    f"{min((c.footprint.resident_bytes for c in _candidates_for(subtask, pool)), default=0) / 1e9:.2f} GB."
                )
            best = max(
                feasible,
                key=lambda c: (
                    cost.quality(subtask, c.model_id) or 0.0,
                    -c.footprint.resident_bytes,
                ),
            )
            assignment[subtask.name] = best.model_id
        return _schedule_from(order, assignment, budget, cost, self.name)


class GreedyScheduler:
    """Swap-aware greedy scheduler. The default that ships.

    Two mechanisms distinguish it from :class:`NaiveLoadOnDemandScheduler`:

    1. **Load amortisation.** When choosing a model for a sub-task it credits
       models already resident with the load cost they avoid, so a marginally
       weaker warm model can beat a marginally better cold one.
    2. **Swap-aware reordering.** Within each topological layer it groups
       sub-tasks by assigned model, so each model loads once per layer instead
       of once per sub-task. This is the Phase 11 batching optimisation
       expressed as a scheduling decision.
    """

    name = "greedy_swap_aware"

    def __init__(self, *, group_by_model: bool = True) -> None:
        """Args: group_by_model: enable swap-aware reordering within a layer."""
        self.group_by_model = group_by_model

    def plan(
        self, graph: TaskGraph, pool: Sequence[ModelCard], budget: MemoryBudget, cost: CostModel
    ) -> Schedule:
        """Produce a schedule, respecting dependencies and the budget."""
        assignment: dict[str, str] = {}
        final_order: list[SubTask] = []
        resident: list[str] = []
        ever_loaded: set[str] = set()

        for layer in graph.topological_layers():
            layer_assign: dict[str, str] = {}
            for subtask in layer:
                feasible = [
                    c for c in _candidates_for(subtask, pool) if budget.fits(c.footprint)
                ]
                if not feasible:
                    raise ScheduleInfeasible(
                        f"No model fits the budget for sub-task {subtask.name!r} "
                        f"({budget.describe()})."
                    )
                best = max(
                    feasible,
                    key=lambda c: self._score(subtask, c, resident, ever_loaded, cost),
                )
                layer_assign[subtask.name] = best.model_id

            ordered = list(layer)
            if self.group_by_model:
                # Warm models first, then group the rest by model so each loads once.
                ordered.sort(
                    key=lambda st: (
                        0 if layer_assign[st.name] in resident else 1,
                        layer_assign[st.name],
                        st.name,
                    )
                )
            for subtask in ordered:
                model_id = layer_assign[subtask.name]
                assignment[subtask.name] = model_id
                final_order.append(subtask)
                if model_id not in resident:
                    while resident:
                        total = MemoryFootprint()
                        for m in [*resident, model_id]:
                            total = total + cost.footprint(m)
                        if all(total.get(t) <= budget.get(t) for t in RESIDENT_TIERS):
                            break
                        resident.pop(0)
                    resident.append(model_id)
                    ever_loaded.add(model_id)
                else:
                    resident.remove(model_id)
                    resident.append(model_id)

        return _schedule_from(final_order, assignment, budget, cost, self.name)

    @staticmethod
    def _score(
        subtask: SubTask,
        card: ModelCard,
        resident: Sequence[str],
        ever_loaded: set[str],
        cost: CostModel,
    ) -> tuple[float, float]:
        """Quality minus the load cost this choice would actually incur.

        Unprofiled models score 0 quality, so a profiled model always wins.
        The tie-break prefers the smaller model, which keeps more headroom.
        """
        quality = cost.quality(subtask, card.model_id) or 0.0
        if card.model_id in resident:
            penalty = 0.0
        else:
            load_s = cost.load_seconds(card.model_id, cold=card.model_id not in ever_loaded)
            penalty = cost.lambda_cost * (load_s or 0.0)
        return (quality - penalty, -float(card.footprint.resident_bytes))


class OracleScheduler:
    """Offline optimal scheduler. For measurement only, never for production.

    Exhaustively searches every valid topological order crossed with every
    model assignment, scoring each through :func:`simulate`. Complexity is
    ``O(orders * k^n)``, so it is capped by ``max_instances``; above that it
    raises rather than silently returning a non-optimal plan.

    The gap between this and :class:`GreedyScheduler` is **scheduler regret**,
    the headline C1 number.
    """

    name = "oracle_exhaustive"

    def __init__(self, *, max_instances: int = 200_000) -> None:
        """Args: max_instances: refuse instances whose search space exceeds this."""
        self.max_instances = max_instances

    def plan(
        self, graph: TaskGraph, pool: Sequence[ModelCard], budget: MemoryBudget, cost: CostModel
    ) -> Schedule:
        """Return the schedule maximising the objective, or raise if too large."""
        orders = list(_all_topological_orders(graph))
        per_task = [
            [c.model_id for c in _candidates_for(st, pool) if budget.fits(c.footprint)]
            for st in graph.nodes
        ]
        for subtask, options in zip(graph.nodes, per_task, strict=True):
            if not options:
                raise ScheduleInfeasible(
                    f"No model fits the budget for sub-task {subtask.name!r} "
                    f"({budget.describe()})."
                )
        space = len(orders)
        for options in per_task:
            space *= len(options)
        if space > self.max_instances:
            raise ScheduleInfeasible(
                f"Oracle search space is {space} (> max_instances={self.max_instances}). "
                f"Reduce the DAG size or the candidate pool; the oracle is a measurement "
                f"tool for small instances only."
            )

        names = [st.name for st in graph.nodes]
        best: tuple[float, Schedule] | None = None
        for combo in itertools.product(*per_task):
            assignment = dict(zip(names, combo, strict=True))
            for order in orders:
                sim = simulate(order, assignment, budget, cost)
                if not sim.feasible or sim.quality is None:
                    continue
                score = cost.objective(sim.quality, sim.seconds)
                if score is None:
                    continue
                if best is None or score > best[0]:
                    best = (
                        score,
                        _schedule_from(order, assignment, budget, cost, self.name),
                    )
        if best is None:
            raise ScheduleInfeasible(
                "Oracle found no feasible, fully-profiled schedule. Every candidate model "
                "must have measured accuracy and cost for the objective to be defined; run "
                "Phase 2a profiling first."
            )
        return best[1]


def _all_topological_orders(graph: TaskGraph) -> Iterable[tuple[SubTask, ...]]:
    """Enumerate every valid topological order of the DAG.

    Exponential by nature. Only the oracle calls this, and only on the small
    instances the brief scopes it to.
    """
    nodes = {n.name: n for n in graph.nodes}
    deps = {n.name: set(n.depends_on) for n in graph.nodes}

    def walk(done: list[str], remaining: set[str]) -> Iterable[tuple[SubTask, ...]]:
        if not remaining:
            yield tuple(nodes[name] for name in done)
            return
        for name in sorted(remaining):
            if deps[name] <= set(done):
                yield from walk([*done, name], remaining - {name})

    return walk([], set(nodes))


# --------------------------------------------------------------------------- #
# Regret
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SchedulerRegret:
    """The C1 headline: how far the online scheduler is from offline optimal.

    ``None`` fields mean the underlying quantity was never measured. They are
    reported as unavailable, never as zero.
    """

    online_planner: str
    oracle_planner: str
    online_seconds: float | None
    oracle_seconds: float | None
    online_quality: float | None
    oracle_quality: float | None
    online_loads: int
    oracle_loads: int

    @property
    def seconds_regret(self) -> float | None:
        """Extra seconds the online scheduler spends. Lower is better."""
        if self.online_seconds is None or self.oracle_seconds is None:
            return None
        return self.online_seconds - self.oracle_seconds

    @property
    def quality_regret(self) -> float | None:
        """Quality the online scheduler gives up. Lower is better."""
        if self.online_quality is None or self.oracle_quality is None:
            return None
        return self.oracle_quality - self.online_quality

    @property
    def relative_seconds_regret(self) -> float | None:
        """Seconds regret as a fraction of the oracle's cost."""
        r = self.seconds_regret
        if r is None or not self.oracle_seconds:
            return None
        return r / self.oracle_seconds

    @property
    def load_regret(self) -> int:
        """Extra model loads the online scheduler pays for."""
        return self.online_loads - self.oracle_loads

    def as_row(self) -> dict[str, object]:
        """Flat dict for a results table. ``None`` renders as 'not measured'."""

        def fmt(v: float | None, digits: int = 3) -> str:
            return "not measured" if v is None else f"{v:.{digits}f}"

        return {
            "online": self.online_planner,
            "oracle": self.oracle_planner,
            "online_s": fmt(self.online_seconds),
            "oracle_s": fmt(self.oracle_seconds),
            "seconds_regret": fmt(self.seconds_regret),
            "relative_seconds_regret": fmt(self.relative_seconds_regret),
            "quality_regret": fmt(self.quality_regret),
            "online_loads": self.online_loads,
            "oracle_loads": self.oracle_loads,
            "load_regret": self.load_regret,
        }


def regret(online: Schedule, oracle: Schedule) -> SchedulerRegret:
    """Compare two schedules of the same instance."""
    return SchedulerRegret(
        online_planner=online.planner,
        oracle_planner=oracle.planner,
        online_seconds=online.predicted_seconds,
        oracle_seconds=oracle.predicted_seconds,
        online_quality=online.predicted_quality,
        oracle_quality=oracle.predicted_quality,
        online_loads=online.predicted_loads,
        oracle_loads=oracle.predicted_loads,
    )
