"""The ``slmflow`` command line. Stdlib ``argparse`` only.

The CLI uses ``argparse`` so the base install stays dependency-free
(``BUILD_BRIEF.md`` section 3).

Commands:

    slmflow probe                    print the measured device card
    slmflow doctor                   check backends, weights and config health
    slmflow plan   <config.yaml>     show the residency plan and predicted cost
    slmflow run    <config.yaml>     run a pipeline once
    slmflow inspect <trace.jsonl>    summarise a trace file
    slmflow models                   list registered model cards
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slmflow import __version__

__all__ = ["main", "build_parser"]

_GB = 1_000_000_000


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="slmflow",
        description="Memory-bounded orchestration of small language models.",
    )
    parser.add_argument("--version", action="version", version=f"slmflow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="measure this machine and print the device card")

    doctor = sub.add_parser("doctor", help="check that this machine can run a pipeline")
    doctor.add_argument("--cards", default="benchmarks/results/model_cards.json")

    for name, help_text in (
        ("plan", "show the residency plan without running"),
        ("run", "run a pipeline once"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("config", help="path to a pipeline YAML/JSON config")
        cmd.add_argument("--cards", default="benchmarks/results/model_cards.json")
        cmd.add_argument(
            "--input",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="prompt input; repeatable",
        )
        cmd.add_argument("--trace", default=None, help="write a JSONL trace here")
        cmd.add_argument("--seed", type=int, default=None)
        cmd.add_argument("--mock", action="store_true", help="use MockBackend, no weights")
        cmd.add_argument("--json", action="store_true", help="machine-readable output")

    inspect = sub.add_parser("inspect", help="summarise a JSONL trace file")
    inspect.add_argument("trace")
    inspect.add_argument("--json", action="store_true")

    models = sub.add_parser("models", help="list registered model cards")
    models.add_argument("--cards", default="benchmarks/results/model_cards.json")
    models.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Errors print a single actionable line rather than a traceback, because a
    stack trace is the wrong output for "your model file is missing".
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            return _cmd_probe()
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
        if args.command == "models":
            return _cmd_models(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _err("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, do not traceback
        from slmflow.errors import SLMFlowError

        if isinstance(exc, SLMFlowError):
            _err(str(exc))
            return 2
        raise
    return 0  # pragma: no cover - argparse requires a subcommand


# --------------------------------------------------------------------------- #
def _out(text: str = "") -> None:
    """Write a line to stdout. The CLI is the only place output is allowed."""
    sys.stdout.write(text + "\n")


def _err(text: str) -> None:
    """Write an error line to stderr."""
    sys.stderr.write(f"slmflow: {text}\n")


def _cmd_probe() -> int:
    """Print the measured device card."""
    from slmflow.device import probe

    report = probe()
    _out(report.as_markdown())
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Check backends, model cards and budgets, and say what to fix."""
    from slmflow.device import probe
    from slmflow.models.backends import available_backends
    from slmflow.resource.budget import MemoryBudget
    from slmflow.resource.rss import measurement_backend

    report = probe()
    card = report.card
    _out("Device")
    _out(f"  tier             {card.tier}")
    _out(f"  cpu              {card.cpu_model or 'not detected'}")
    _out(f"  cores            {card.logical_cores}")
    _out(f"  ram              {card.total_ram_bytes / _GB:.2f} GB total, "
         f"{card.available_ram_bytes / _GB:.2f} GB available")
    _out(f"  gpu              {card.gpu_model or 'none detected'}")
    _out(f"  vram             {card.total_vram_bytes / _GB:.2f} GB")
    _out(f"  free disk        {card.free_disk_bytes / _GB:.2f} GB")
    _out(f"  rss measured by  {measurement_backend()}")

    budget = MemoryBudget.from_device(card)
    _out("")
    _out(f"Derived budget     {budget.describe()}")

    _out("")
    _out("Backends")
    problems = 0
    for name, ok in available_backends().items():
        _out(f"  {name:<10} {'available' if ok else 'NOT available'}")
        if name != "mock" and not ok:
            problems += 1

    _out("")
    _out("Model cards")
    cards_path = Path(args.cards)
    if not cards_path.exists():
        _out(f"  none at {cards_path} -- run Phase 2 profiling to create it")
        problems += 1
    else:
        from slmflow.models.registry import ModelRegistry

        registry = ModelRegistry.from_file(cards_path)
        _out(f"  {len(registry)} registered, {len(registry.profiled())} fully profiled")
        unprofiled = registry.unprofiled_report()
        if unprofiled:
            _out(f"  not yet measured: {', '.join(unprofiled)}")
        fits = [c.model_id for c in registry.within_budget(budget)]
        _out(f"  fit the budget: {', '.join(fits) if fits else 'NONE'}")
        if not fits:
            problems += 1

    for warning in report.warnings:
        _out("")
        _out(f"WARNING: {warning}")

    _out("")
    _out("OK" if not problems else f"{problems} issue(s) need attention")
    return 0 if not problems else 1


def _build_registry(args: argparse.Namespace) -> Any:
    """Load a registry, or build a mock one when ``--mock`` is set."""
    from slmflow.models.registry import ModelRegistry

    if getattr(args, "mock", False):
        from slmflow.models.backends.mock import MockBackend, make_pool

        _err("MOCK BACKEND: scripted answers, no model was run. Numbers are not measurements.")

        registry = ModelRegistry(
            make_pool(
                [("mock-small", 0.5, 400_000_000), ("mock-large", 3.0, 2_200_000_000)],
                accuracy={
                    "mock-small": {t: 0.60 for t in ("qa", "classify", "extract")},
                    "mock-large": {t: 0.78 for t in ("qa", "classify", "extract")},
                },
            )
        )
        registry.register_backend("mock", MockBackend(default_answer="mock answer"))
        return registry
    return ModelRegistry.from_file(args.cards)


def _parse_inputs(pairs: Sequence[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` CLI inputs."""
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"slmflow: --input needs KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = value
    return out


def _cmd_plan(args: argparse.Namespace) -> int:
    """Print the residency plan and its predicted cost."""
    from slmflow.orchestration.pipeline import Pipeline, PipelineOptions

    pipeline = Pipeline.from_config(
        args.config,
        registry=_build_registry(args),
        options=PipelineOptions(seed=args.seed),
    )
    schedule = pipeline.plan(_parse_inputs(args.input))

    if args.json:
        _out(json.dumps(
            {
                "pipeline": pipeline.describe(),
                "entries": [
                    {
                        "step": e.step,
                        "model": e.model_id,
                        "load_before": e.load_before,
                        "evict_before": list(e.evict_before),
                    }
                    for e in schedule.entries
                ],
                "predicted_seconds": schedule.predicted_seconds,
                "predicted_quality": schedule.predicted_quality,
                "predicted_loads": schedule.predicted_loads,
                "peak_vram_bytes": schedule.peak_footprint.vram_bytes,
                "peak_ram_bytes": schedule.peak_footprint.ram_bytes,
            },
            indent=2,
        ))
        return 0

    _out(f"Pipeline   {pipeline.config.name}  (planner: {schedule.planner})")
    _out(f"Budget     {pipeline.budget.describe()}")
    _out(f"Models     {', '.join(c.model_id for c in pipeline.pool)}")
    _out("")
    _out(f"{'step':<20} {'model':<24} {'load':<6} evicts")
    _out("-" * 72)
    for entry in schedule.entries:
        _out(
            f"{entry.step:<20} {entry.model_id:<24} "
            f"{'yes' if entry.load_before else 'no':<6} "
            f"{', '.join(entry.evict_before) or '-'}"
        )
    _out("")
    _out(f"predicted loads    {schedule.predicted_loads}")
    _out(f"predicted seconds  {_fmt(schedule.predicted_seconds)}")
    _out(f"predicted quality  {_fmt(schedule.predicted_quality)}")
    _out(
        f"peak memory        vram {schedule.peak_footprint.vram_bytes / _GB:.2f} GB, "
        f"ram {schedule.peak_footprint.ram_bytes / _GB:.2f} GB"
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a pipeline once and print the answer plus its measured cost."""
    from slmflow.orchestration.pipeline import Pipeline, PipelineOptions

    pipeline = Pipeline.from_config(
        args.config,
        registry=_build_registry(args),
        options=PipelineOptions(trace_path=args.trace, seed=args.seed),
    )
    with pipeline:
        result = pipeline.run(_parse_inputs(args.input))

    trace = result.trace
    if args.json:
        from slmflow.telemetry.trace import trace_to_dict

        _out(json.dumps(
            {
                "answer": result.answer,
                "confidence": result.confidence,
                "abstained": result.abstained,
                "abstention_reason": (
                    str(result.abstention_reason) if result.abstention_reason else None
                ),
                "trace": trace_to_dict(trace),
            },
            indent=2,
        ))
        return 0

    if result.abstained:
        _out(f"ABSTAINED ({result.abstention_reason})")
    else:
        _out(str(result.answer))
    _out("")
    _out(f"confidence      {_fmt(result.confidence)}")
    _out(f"models used     {', '.join(trace.models_used) or '-'}")
    _out(f"model loads     {trace.total_swaps} ({trace.swap_seconds:.2f} s)")
    _out(f"generations     {trace.total_generations}")
    _out(f"output tokens   {trace.output_tokens}")
    _out(
        f"peak memory     vram {trace.peak_vram_bytes / _GB:.2f} GB, "
        f"ram {trace.peak_ram_bytes / _GB:.2f} GB"
    )
    _out(f"wall clock      {trace.wall_s:.2f} s")
    if args.trace:
        _out(f"trace written   {args.trace}")
    return 3 if result.abstained else 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Summarise a JSONL trace file."""
    from slmflow.telemetry.trace import read_traces

    runs = [r for r in read_traces(args.trace) if "run_id" in r]
    if not runs:
        _err(f"no run records in {args.trace}")
        return 1

    if args.json:
        _out(json.dumps(_summarise(runs), indent=2))
        return 0

    summary = _summarise(runs)
    _out(f"Trace          {args.trace}")
    _out(f"runs           {summary['n_runs']}")
    _out(f"pipelines      {', '.join(summary['pipelines'])}")
    _out(f"mean wall s    {_fmt(summary['mean_wall_s'])}")
    _out(f"mean loads     {_fmt(summary['mean_loads'])}")
    _out(f"mean swap s    {_fmt(summary['mean_swap_s'])}")
    _out(f"abstentions    {summary['abstentions']}")
    _out(f"peak vram GB   {_fmt(summary['peak_vram_gb'])}")
    _out(f"peak ram GB    {_fmt(summary['peak_ram_gb'])}")
    return 0


def _summarise(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate run records."""
    from slmflow.reliability.metrics import mean_std

    walls = [float(r.get("derived", {}).get("wall_s", 0.0)) for r in runs]
    loads = [float(r.get("derived", {}).get("total_swaps", 0)) for r in runs]
    swaps = [float(r.get("derived", {}).get("swap_seconds", 0.0)) for r in runs]
    abstentions = sum(
        1
        for r in runs
        for s in r.get("steps", [])
        if isinstance(s, dict) and s.get("abstained")
    )
    return {
        "n_runs": len(runs),
        "pipelines": sorted({str(r.get("pipeline", "?")) for r in runs}),
        "mean_wall_s": mean_std(walls)[0],
        "mean_loads": mean_std(loads)[0],
        "mean_swap_s": mean_std(swaps)[0],
        "abstentions": abstentions,
        "peak_vram_gb": max(
            (float(r.get("derived", {}).get("peak_vram_bytes", 0)) / _GB for r in runs),
            default=0.0,
        ),
        "peak_ram_gb": max(
            (float(r.get("derived", {}).get("peak_ram_bytes", 0)) / _GB for r in runs),
            default=0.0,
        ),
    }


def _cmd_models(args: argparse.Namespace) -> int:
    """List registered model cards and flag unmeasured fields."""
    from slmflow.models.registry import ModelRegistry, card_to_dict

    registry = ModelRegistry.from_file(args.cards)
    if args.json:
        _out(json.dumps([card_to_dict(c) for c in registry], indent=2))
        return 0

    _out(f"{'model':<26} {'params':>7} {'quant':<9} {'GB':>6} {'load s':>7} {'tok/s':>7}")
    _out("-" * 72)
    for card in registry:
        _out(
            f"{card.model_id:<26} {card.params_b:>6.1f}B {str(card.quantisation):<9} "
            f"{card.footprint.resident_bytes / _GB:>6.2f} "
            f"{_fmt(card.cold_load_s, 2):>7} {_fmt(card.tokens_per_s, 1):>7}"
        )
    unprofiled = registry.unprofiled_report()
    if unprofiled:
        _out("")
        _out(f"not yet measured: {', '.join(unprofiled)}")
    return 0


def _fmt(value: float | None, digits: int = 3) -> str:
    """Render a possibly-unmeasured number honestly."""
    return "n/a" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
