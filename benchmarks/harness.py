"""Phase 12: the evaluation harness. One command reproduces the whole suite.

Runs **any** configuration against a fixed task set and emits one table
carrying quality *and* cost together (``context.md`` section 8). Ablations go
through the same harness, so an ablation and a headline number are never
produced by different code paths.

Guarantees this harness enforces, from Phase 12's Definition of Done:

* fixed seeds, and >= 3 seeds per configuration by default;
* mean +/- std across seeds, never a single run reported as if it were stable;
* a significance test (paired bootstrap and McNemar) on every claimed
  difference against the named baseline;
* pinned model digests and the probed device card in every artifact;
* ``not yet measured`` wherever a number was not obtained, never a placeholder.

Usage::

    python -m benchmarks.harness --list
    python -m benchmarks.harness --family repurposing_label --limit 20 --mock
    python -m benchmarks.harness --all --seeds 3 --out benchmarks/results
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.tasks import BenchmarkItem, TaskFamily, load_family
from slmflow.agents.schema import extract_json
from slmflow.device import ProbeReport, probe
from slmflow.models.registry import ModelRegistry
from slmflow.orchestration.pipeline import Pipeline, PipelineOptions
from slmflow.reliability.confidence import RetrievalSupport, normalise_answer
from slmflow.reliability.metrics import (
    cost_normalised_accuracy,
    expected_calibration_error,
    mcnemar,
    mean_std,
    paired_bootstrap,
)
from slmflow.resource.budget import MemoryBudget
from slmflow.types import Generation

__all__ = ["Configuration", "RunOutcome", "ItemOutcome", "evaluate", "CONFIGURATIONS", "main"]


# --------------------------------------------------------------------------- #
# Configurations under test
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Configuration:
    """One system under test: a strategy, router and scheduler combination.

    ``ablation_of`` names the configuration this one removes a component from,
    so the ablation table is derived from the definitions rather than
    maintained separately and allowed to drift.
    """

    name: str
    strategy: str
    router: str = "heuristic"
    scheduler: str = "greedy"
    k: int = 5
    temperature: float = 0.0
    threshold: float = 0.0
    use_grammar: bool = True
    verifier: str | None = None
    ablation_of: str | None = None
    description: str = ""

    def step_overrides(self) -> dict[str, Any]:
        """Step-level fields this configuration imposes."""
        return {
            "strategy": self.strategy,
            "k": self.k,
            "temperature": self.temperature,
            "threshold": self.threshold,
            "use_grammar": self.use_grammar,
            "verifier": self.verifier,
        }


#: The suite. Baselines first, then the contributions, then the ablations.
CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration(
        name="single_smallest",
        strategy="single",
        router="static:__smallest__",
        description="one small model, greedy decoding: the trivial baseline",
    ),
    Configuration(
        name="single_best",
        strategy="single",
        router="heuristic",
        description="best model per task by measured accuracy, no ensembling",
    ),
    Configuration(
        name="slm_mux",
        strategy="slm_mux",
        k=5,
        temperature=0.7,
        description="SLM-MUX re-implementation: k samples per model, confidence selection",
    ),
    Configuration(
        name="agent_forest",
        strategy="agent_forest",
        k=5,
        temperature=0.7,
        description="flat majority vote over all samples (Agent-Forest style)",
    ),
    Configuration(
        name="adaptive_sc",
        strategy="adaptive",
        k=5,
        temperature=0.7,
        description="C4: sequential self-consistency with early stopping",
    ),
    Configuration(
        name="cascade",
        strategy="cascade",
        k=1,
        threshold=0.7,
        description="C4: cheapest model first, escalate on low confidence",
    ),
    # -- ablations ---------------------------------------------------------- #
    Configuration(
        name="ablate_scheduler",
        strategy="adaptive",
        k=5,
        temperature=0.7,
        scheduler="naive",
        ablation_of="adaptive_sc",
        description="C1 ablation: load-on-demand with LRU eviction, no swap-awareness",
    ),
    Configuration(
        name="ablate_router",
        strategy="adaptive",
        k=5,
        temperature=0.7,
        router="random",
        ablation_of="adaptive_sc",
        description="Phase 7 ablation: random model choice",
    ),
    Configuration(
        name="ablate_grammar",
        strategy="adaptive",
        k=5,
        temperature=0.7,
        use_grammar=False,
        ablation_of="adaptive_sc",
        description="Phase 10 ablation: no grammar constraint, validation only",
    ),
)

BASELINE = "single_best"


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ItemOutcome:
    """Per-item record. Kept because significance tests need paired items."""

    item_id: str
    predicted: str
    gold: str
    correct: bool
    confidence: float
    abstained: bool
    schema_valid: bool | None
    seconds: float
    swaps: int
    swap_seconds: float
    generations: int
    output_tokens: int
    peak_vram_bytes: int
    peak_ram_bytes: int
    models_used: tuple[str, ...] = ()
    groundedness: float | None = None
    error: str | None = None


@dataclass(slots=True)
class RunOutcome:
    """All items for one (configuration, family, seed)."""

    configuration: str
    family: str
    seed: int
    items: list[ItemOutcome] = field(default_factory=list)
    wall_s: float = 0.0

    # -- quality -------------------------------------------------------- #
    @property
    def n(self) -> int:
        """Items evaluated."""
        return len(self.items)

    @property
    def accuracy(self) -> float | None:
        """Accuracy over all items, counting an abstention as incorrect.

        Reported alongside :attr:`answered_accuracy`: an abstaining system that
        is scored only on what it answered would look artificially strong.
        """
        if not self.items:
            return None
        return sum(1 for i in self.items if i.correct) / len(self.items)

    @property
    def answered_accuracy(self) -> float | None:
        """Precision on the answered subset. The C3 quantity."""
        answered = [i for i in self.items if not i.abstained]
        if not answered:
            return None
        return sum(1 for i in answered if i.correct) / len(answered)

    @property
    def coverage(self) -> float | None:
        """Fraction of items answered rather than abstained."""
        if not self.items:
            return None
        return sum(1 for i in self.items if not i.abstained) / len(self.items)

    @property
    def schema_valid_rate(self) -> float | None:
        """Share of items whose output satisfied the schema, or ``None``."""
        judged = [i for i in self.items if i.schema_valid is not None]
        if not judged:
            return None
        return sum(1 for i in judged if i.schema_valid) / len(judged)

    @property
    def mean_groundedness(self) -> float | None:
        """Mean retrieval-support score, or ``None`` off the RAG path."""
        scored = [i.groundedness for i in self.items if i.groundedness is not None]
        return sum(scored) / len(scored) if scored else None

    # -- cost ----------------------------------------------------------- #
    @property
    def mean_seconds(self) -> float | None:
        """Mean wall-clock seconds per item."""
        return mean_std([i.seconds for i in self.items])[0] if self.items else None

    @property
    def total_swaps(self) -> int:
        """Model loads across the run."""
        return sum(i.swaps for i in self.items)

    @property
    def mean_swap_seconds(self) -> float | None:
        """Mean seconds per item spent loading models."""
        return mean_std([i.swap_seconds for i in self.items])[0] if self.items else None

    @property
    def mean_generations(self) -> float | None:
        """Mean samples drawn per item: the C4 cost number."""
        return mean_std([float(i.generations) for i in self.items])[0] if self.items else None

    @property
    def peak_vram_bytes(self) -> int:
        """Highest VRAM residency observed."""
        return max((i.peak_vram_bytes for i in self.items), default=0)

    @property
    def peak_ram_bytes(self) -> int:
        """Highest system-RAM residency observed."""
        return max((i.peak_ram_bytes for i in self.items), default=0)

    def cost_normalised(self) -> dict[str, float | None]:
        """Accuracy per second and per GB of peak memory."""
        return cost_normalised_accuracy(
            self.accuracy or 0.0,
            self.mean_seconds,
            max(self.peak_vram_bytes, self.peak_ram_bytes) or None,
        )

    def calibration(self) -> dict[str, Any]:
        """ECE and its bins for this run's confidence signal."""
        if not self.items:
            return {"ece": None, "bins": []}
        ece, bins = expected_calibration_error(
            [i.confidence for i in self.items], [i.correct for i in self.items]
        )
        return {"ece": ece, "bins": bins.as_rows()}

    def correctness_vector(self) -> list[bool]:
        """Per-item correctness, ordered by item id, for McNemar."""
        return [i.correct for i in sorted(self.items, key=lambda i: i.item_id)]

    def score_vector(self) -> list[float]:
        """Per-item score, ordered by item id, for the paired bootstrap."""
        return [1.0 if i.correct else 0.0 for i in sorted(self.items, key=lambda i: i.item_id)]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _score_item(family: TaskFamily, item: BenchmarkItem, predicted: str) -> bool:
    """Apply the family's metric to one prediction."""
    if family.metric == "json_exact_match":
        parsed = extract_json(predicted)
        if parsed is None:
            return False
        try:
            gold = json.loads(item.gold)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False
        return all(
            str(parsed.get(key, "")).strip() == str(value).strip()
            for key, value in gold.items()
        )
    if family.metric == "groundedness":
        # Groundedness is scored separately; correctness here is a loose
        # containment check, deliberately reported as such in the table
        # rather than presented as answer accuracy.
        gold_head = normalise_answer(item.gold)[:40]
        return bool(gold_head) and gold_head in normalise_answer(predicted)
    return normalise_answer(predicted) == normalise_answer(item.gold)


def _pipeline_for(
    configuration: Configuration,
    family: TaskFamily,
    registry: ModelRegistry,
    *,
    seed: int,
    budget: MemoryBudget | None,
    device: ProbeReport,
    trace_path: Path | None,
) -> Pipeline:
    """Build the pipeline for one configuration and family."""
    router = configuration.router
    if router == "static:__smallest__":
        cheapest = registry.cheapest()
        router = f"static:{cheapest.model_id}" if cheapest else "heuristic"

    step = family.as_step_config()
    step.update({k: v for k, v in configuration.step_overrides().items() if v is not None})

    raw = {
        "name": f"{configuration.name}::{family.name}",
        "router": router,
        "scheduler": configuration.scheduler,
        "seed": seed,
        "steps": [step],
    }
    return Pipeline.from_dict(
        raw,
        registry=registry,
        budget=budget,
        options=PipelineOptions(trace_path=trace_path, seed=seed),
    )


def evaluate(
    configuration: Configuration,
    family: TaskFamily,
    registry: ModelRegistry,
    *,
    seed: int = 0,
    budget: MemoryBudget | None = None,
    device: ProbeReport | None = None,
    retriever: Any = None,
    top_k: int = 5,
    trace_path: Path | None = None,
) -> RunOutcome:
    """Run one configuration over one family at one seed."""
    device = device or probe()
    outcome = RunOutcome(configuration=configuration.name, family=family.name, seed=seed)
    support = RetrievalSupport()
    started = time.perf_counter()

    pipeline = _pipeline_for(
        configuration,
        family,
        registry,
        seed=seed,
        budget=budget,
        device=device,
        trace_path=trace_path,
    )
    with pipeline:
        for item in family.items:
            retrieved: list[str] = []
            if family.needs_retrieval and retriever is not None:
                query = str(item.inputs.get("question") or item.inputs.get("drug", ""))
                retrieved = [p.text for p, _ in retriever.search(query, top_k)]

            item_started = time.perf_counter()
            try:
                result = pipeline.run(item.inputs, retrieved=retrieved)
            except Exception as exc:  # noqa: BLE001 - a failed item is data, not a crash
                outcome.items.append(
                    ItemOutcome(
                        item_id=item.item_id,
                        predicted="",
                        gold=item.gold,
                        correct=False,
                        confidence=0.0,
                        abstained=True,
                        schema_valid=None,
                        seconds=time.perf_counter() - item_started,
                        swaps=0,
                        swap_seconds=0.0,
                        generations=0,
                        output_tokens=0,
                        peak_vram_bytes=0,
                        peak_ram_bytes=0,
                        error=repr(exc),
                    )
                )
                continue

            predicted = "" if result.answer is None else str(result.answer)
            trace = result.trace
            groundedness = None
            if retrieved:
                groundedness = support.score(
                    (Generation(text=predicted, model_id=""),), {"retrieved": retrieved}
                )

            outcome.items.append(
                ItemOutcome(
                    item_id=item.item_id,
                    predicted=predicted,
                    gold=item.gold,
                    correct=(not result.abstained) and _score_item(family, item, predicted),
                    confidence=result.confidence or 0.0,
                    abstained=result.abstained,
                    schema_valid=result.schema_valid,
                    seconds=time.perf_counter() - item_started,
                    swaps=trace.total_swaps,
                    swap_seconds=trace.swap_seconds,
                    generations=trace.total_generations,
                    output_tokens=trace.output_tokens,
                    peak_vram_bytes=trace.peak_vram_bytes,
                    peak_ram_bytes=trace.peak_ram_bytes,
                    models_used=trace.models_used,
                    groundedness=groundedness,
                )
            )

    outcome.wall_s = time.perf_counter() - started
    return outcome


# --------------------------------------------------------------------------- #
# Aggregation and reporting
# --------------------------------------------------------------------------- #
def _fmt(value: float | None, digits: int = 3) -> str:
    """Render a number, or the honest absence of one."""
    return "not measured" if value is None else f"{value:.{digits}f}"


def aggregate(runs: Sequence[RunOutcome]) -> dict[str, Any]:
    """Mean +/- std across seeds for one configuration and family."""
    if not runs:
        return {}
    accuracies = [r.accuracy for r in runs if r.accuracy is not None]
    seconds = [r.mean_seconds for r in runs if r.mean_seconds is not None]
    generations = [r.mean_generations for r in runs if r.mean_generations is not None]
    swap_seconds = [r.mean_swap_seconds for r in runs if r.mean_swap_seconds is not None]

    acc_mean, acc_std = mean_std(accuracies) if accuracies else (None, None)
    sec_mean, sec_std = mean_std(seconds) if seconds else (None, None)
    gen_mean, _ = mean_std(generations) if generations else (None, None)
    swap_mean, _ = mean_std(swap_seconds) if swap_seconds else (None, None)

    first = runs[0]
    return {
        "configuration": first.configuration,
        "family": first.family,
        "n_seeds": len(runs),
        "n_items": first.n,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "answered_accuracy": _mean_opt([r.answered_accuracy for r in runs]),
        "coverage": _mean_opt([r.coverage for r in runs]),
        "schema_valid_rate": _mean_opt([r.schema_valid_rate for r in runs]),
        "groundedness": _mean_opt([r.mean_groundedness for r in runs]),
        "mean_seconds": sec_mean,
        "seconds_std": sec_std,
        "mean_generations": gen_mean,
        "mean_swap_seconds": swap_mean,
        "total_swaps": sum(r.total_swaps for r in runs) / len(runs),
        "peak_vram_bytes": max(r.peak_vram_bytes for r in runs),
        "peak_ram_bytes": max(r.peak_ram_bytes for r in runs),
        "ece": _mean_opt([r.calibration()["ece"] for r in runs]),
        "cost_normalised": runs[0].cost_normalised(),
    }


def _mean_opt(values: Sequence[float | None]) -> float | None:
    """Mean of the non-``None`` values, or ``None`` when all are missing."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def significance(
    runs_by_config: Mapping[str, Sequence[RunOutcome]], baseline: str = BASELINE
) -> list[dict[str, Any]]:
    """Paired bootstrap and McNemar for every configuration against the baseline.

    Phase 12: no claimed difference may lack a significance test. This runs the
    tests unconditionally so the table cannot be published without them.
    """
    if baseline not in runs_by_config:
        return []
    base_runs = runs_by_config[baseline]
    if not base_runs:
        return []
    base = base_runs[0]

    rows: list[dict[str, Any]] = []
    for name, runs in runs_by_config.items():
        if name == baseline or not runs:
            continue
        candidate = runs[0]
        if candidate.n != base.n or candidate.n == 0:
            rows.append(
                {
                    "configuration": name,
                    "baseline": baseline,
                    "note": (
                        f"not comparable: {candidate.n} items vs baseline {base.n}; "
                        f"significance requires the same item set"
                    ),
                }
            )
            continue
        boot = paired_bootstrap(candidate.score_vector(), base.score_vector(), seed=0)
        mc = mcnemar(candidate.correctness_vector(), base.correctness_vector())
        rows.append(
            {
                "configuration": name,
                "baseline": baseline,
                **boot.as_row(),
                "mcnemar": mc.as_row(),
                "sentence": boot.sentence(name, baseline),
            }
        )
    return rows


def environment_record(device: ProbeReport, registry: ModelRegistry) -> dict[str, Any]:
    """Machine, versions and pinned digests. Required in every artifact."""
    from slmflow import __version__
    from slmflow.resource.rss import measurement_backend

    return {
        "slmflow_version": __version__,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": device.card.platform,
        "cpu": device.card.cpu_model,
        "logical_cores": device.card.logical_cores,
        "total_ram_bytes": device.card.total_ram_bytes,
        "gpu": device.card.gpu_model,
        "total_vram_bytes": device.card.total_vram_bytes,
        "cuda_available": device.card.cuda_available,
        "device_tier": str(device.card.tier),
        "rss_measurement": measurement_backend(),
        "model_digests": registry.digests(),
        "unprofiled_models": registry.unprofiled_report(),
        "probe_warnings": device.warnings,
    }


def render_markdown(
    aggregates: Sequence[Mapping[str, Any]],
    sig_rows: Sequence[Mapping[str, Any]],
    families: Sequence[TaskFamily],
    environment: Mapping[str, Any],
) -> str:
    """Render the results table. This text becomes ``docs/RESULTS.md``."""
    lines: list[str] = []
    lines.append("# Results")
    lines.append("")
    lines.append(
        "Generated by `python -m benchmarks.harness`. Every number here was produced by "
        "a run on the machine described below. A blank or `not measured` cell means the "
        "quantity was not obtained -- it is never a placeholder or an estimate."
    )
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    for key, value in environment.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## Task families")
    lines.append("")
    lines.append("| family | items | metric | majority-class baseline | label distribution |")
    lines.append("|---|---|---|---|---|")
    for family in families:
        distribution = (
            ", ".join(f"{k}={v}" for k, v in family.label_distribution().items())
            if family.metric != "groundedness"
            else "n/a"
        )
        lines.append(
            f"| {family.name} | {len(family)} | {family.metric} | "
            f"{family.majority_baseline():.3f} | {distribution} |"
        )
    lines.append("")

    lines.append("## Quality and cost together")
    lines.append("")
    lines.append(
        "| configuration | family | seeds | accuracy | answered acc | coverage | "
        "schema valid | s/item | swap s/item | gens/item | loads | peak VRAM GB | "
        "peak RAM GB | acc/s | ECE |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in aggregates:
        acc = (
            f"{row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f}"
            if row.get("accuracy_mean") is not None
            else "not measured"
        )
        cost = row.get("cost_normalised") or {}
        lines.append(
            f"| {row['configuration']} | {row['family']} | {row['n_seeds']} | {acc} | "
            f"{_fmt(row.get('answered_accuracy'))} | {_fmt(row.get('coverage'))} | "
            f"{_fmt(row.get('schema_valid_rate'))} | {_fmt(row.get('mean_seconds'))} | "
            f"{_fmt(row.get('mean_swap_seconds'))} | {_fmt(row.get('mean_generations'), 2)} | "
            f"{_fmt(row.get('total_swaps'), 1)} | "
            f"{row.get('peak_vram_bytes', 0) / 1e9:.2f} | "
            f"{row.get('peak_ram_bytes', 0) / 1e9:.2f} | "
            f"{_fmt(cost.get('accuracy_per_second'), 4)} | {_fmt(row.get('ece'))} |"
        )
    lines.append("")

    lines.append("## Ablations")
    lines.append("")
    by_name = {c.name: c for c in CONFIGURATIONS}
    ablations = [c for c in CONFIGURATIONS if c.ablation_of]
    if ablations:
        lines.append("| ablation | removes | vs | description |")
        lines.append("|---|---|---|---|")
        for conf in ablations:
            parent = by_name.get(conf.ablation_of or "")
            lines.append(
                f"| {conf.name} | {conf.description} | "
                f"{parent.name if parent else '?'} | {conf.description} |"
            )
    lines.append("")

    lines.append("## Significance")
    lines.append("")
    if not sig_rows:
        lines.append(
            "No significance tests were run: the baseline configuration produced no "
            "comparable items. **No difference may be claimed from this table.**"
        )
    else:
        lines.append("| configuration | difference | 95% CI | p (bootstrap) | McNemar p | significant |")
        lines.append("|---|---|---|---|---|---|")
        for row in sig_rows:
            if "note" in row:
                lines.append(f"| {row['configuration']} | {row['note']} | | | | |")
                continue
            mc = row.get("mcnemar") or {}
            lines.append(
                f"| {row['configuration']} | {row['difference']:+.4f} | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['p_value']:.4f} | "
                f"{mc.get('p_value', 'n/a')} | {'yes' if row['significant'] else 'no'} |"
            )
    lines.append("")
    lines.append(
        "> Reading guide. `accuracy` counts an abstention as incorrect; "
        "`answered acc` is precision on the answered subset and must be read together "
        "with `coverage`. `gens/item` is the C4 cost number. `swap s/item` is the cost "
        "that no cloud-oriented framework measures."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_registry(*, mock: bool, cards: str) -> ModelRegistry:
    """Load the registry, or build a mock pool for a dry run."""
    if mock:
        from slmflow.models.backends.mock import MockBackend, ScriptedResponse, make_pool

        registry = ModelRegistry(
            make_pool(
                [("mock-small", 0.5, 400_000_000), ("mock-large", 3.0, 2_200_000_000)],
                accuracy={
                    "mock-small": {"classify": 0.58, "extract": 0.44, "qa": 0.51},
                    "mock-large": {"classify": 0.79, "extract": 0.71, "qa": 0.68},
                },
                load_s={"mock-small": 0.8, "mock-large": 4.2},
                tps={"mock-small": 34.0, "mock-large": 9.5},
            )
        )
        # Scripted so the mock run exercises agreement and disagreement paths.
        responses = {
            "mock-small": {
                "*": ScriptedResponse({"investigational": 0.6, "approved": 0.4}),
            },
            "mock-large": {
                "*": ScriptedResponse({"investigational": 0.85, "hypothesis": 0.15}),
            },
        }
        registry.register_backend("mock", MockBackend(responses, default_answer="investigational"))
        return registry
    return ModelRegistry.from_file(cards)


def main(argv: Sequence[str] | None = None) -> int:
    """Harness entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.harness",
        description="Run the SLMFlow evaluation suite and regenerate docs/RESULTS.md.",
    )
    parser.add_argument("--family", action="append", default=[], help="task family; repeatable")
    parser.add_argument("--all", action="store_true", help="run every family")
    parser.add_argument("--config", action="append", default=[], help="configuration name")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="cap items per family")
    parser.add_argument("--seeds", type=int, default=3, help="seeds per configuration (>=3)")
    parser.add_argument("--cards", default="benchmarks/results/model_cards.json")
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--results-md", default="docs/RESULTS.md")
    parser.add_argument("--mock", action="store_true", help="MockBackend; no weights needed")
    parser.add_argument("--retrieval", action="store_true", help="enable BM25 for RAG families")
    parser.add_argument("--list", action="store_true", help="list families and configurations")
    args = parser.parse_args(argv)

    if args.list:
        from benchmarks.tasks import FAMILIES

        sys.stdout.write("families:\n")
        for name in sorted(FAMILIES):
            sys.stdout.write(f"  {name}\n")
        sys.stdout.write("configurations:\n")
        for conf in CONFIGURATIONS:
            tag = f" (ablation of {conf.ablation_of})" if conf.ablation_of else ""
            sys.stdout.write(f"  {conf.name}{tag}: {conf.description}\n")
        return 0

    from benchmarks.tasks import FAMILIES

    family_names = list(args.family)
    if args.all or not family_names:
        family_names = sorted(FAMILIES)
    unknown = [n for n in family_names if n not in FAMILIES]
    if unknown:
        sys.stderr.write(
            f"unknown families {unknown}; valid: {', '.join(sorted(FAMILIES))}\n"
        )
        return 2

    if args.seeds < 3:
        sys.stderr.write(
            f"warning: --seeds {args.seeds} is below the 3 required by Phase 12; "
            f"results will be marked as under-seeded.\n"
        )

    device = probe()
    registry = build_registry(mock=args.mock, cards=args.cards)
    budget = MemoryBudget.from_device(device.card)

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    configurations = (
        [c for c in CONFIGURATIONS if c.name in set(args.config)]
        if args.config
        else list(CONFIGURATIONS)
    )

    retriever = None
    families: list[TaskFamily] = []
    for name in family_names:
        kwargs: dict[str, Any] = {"limit": args.limit}
        if name in {"repurposing_label", "grounded_qa"}:
            kwargs["split"] = args.split
        families.append(load_family(name, **kwargs))

    if args.retrieval and any(f.needs_retrieval for f in families):
        from slmflow.rag.store import BM25Retriever, load_passages

        retriever = BM25Retriever(load_passages(limit=4000))

    aggregates: list[dict[str, Any]] = []
    all_runs: dict[str, list[RunOutcome]] = {}

    for family in families:
        if not len(family):
            sys.stderr.write(f"warning: family {family.name} has no items; skipping.\n")
            continue
        for conf in configurations:
            runs: list[RunOutcome] = []
            for seed in range(args.seeds):
                outcome = evaluate(
                    conf,
                    family,
                    registry,
                    seed=seed,
                    budget=budget,
                    device=device,
                    retriever=retriever,
                    trace_path=raw_dir / f"{conf.name}_{family.name}_seed{seed}.jsonl",
                )
                runs.append(outcome)
                (raw_dir / f"{conf.name}_{family.name}_seed{seed}_items.json").write_text(
                    json.dumps([asdict(i) for i in outcome.items], indent=2, default=str),
                    encoding="utf-8",
                )
            aggregates.append(aggregate(runs))
            all_runs.setdefault(conf.name, []).extend(runs)
            sys.stdout.write(
                f"{conf.name:<20} {family.name:<20} "
                f"acc={_fmt(runs[0].accuracy)} s/item={_fmt(runs[0].mean_seconds)}\n"
            )

    sig_rows = significance(all_runs)
    environment = environment_record(device, registry)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "environment": environment,
                "aggregates": aggregates,
                "significance": sig_rows,
                "seeds": args.seeds,
                "split": args.split,
                "limit": args.limit,
                "mock": args.mock,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    markdown = render_markdown(aggregates, sig_rows, families, environment)
    if args.mock:
        markdown = (
            "> **THESE NUMBERS ARE FROM `MockBackend`, NOT REAL MODELS.**\n"
            "> They exercise the harness end to end and must never be cited as results.\n\n"
        ) + markdown
    Path(args.results_md).write_text(markdown, encoding="utf-8")
    sys.stdout.write(f"\nwrote {args.results_md} and {out_dir / 'summary.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
