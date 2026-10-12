"""Phase 10 / C3: risk-coverage, ECE, and the precision contract.

Produces the Phase 10 evidence:

* a risk-coverage curve, the headline plot;
* ECE and Brier **per confidence signal**, so signals are compared rather than
  assumed -- this is what makes "we address SLM-MUX's stated weakness" a
  measurement instead of a claim;
* the tuned threshold for a declared precision target, plus the coverage it
  costs, evaluated on a **held-out** half so the guarantee is not tested on the
  data that fitted it;
* schema-validity rate with and without grammar constraints.

Usage::

    python -m benchmarks.calibrate --family repurposing_label --limit 80 --target 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.tasks import load_family
from slmflow.device import probe
from slmflow.errors import GuaranteeUnmet
from slmflow.models.registry import ModelRegistry
from slmflow.orchestration.pipeline import Pipeline, PipelineOptions
from slmflow.reliability.calibration import (
    CalibrationPoint,
    Precision,
    coverage_at_precision,
    risk_coverage_curve,
    tune_threshold,
)
from slmflow.reliability.confidence import normalise_answer
from slmflow.reliability.metrics import brier_score, expected_calibration_error
from slmflow.resource.budget import MemoryBudget

__all__ = ["main"]


def _collect(
    registry: ModelRegistry,
    budget: MemoryBudget,
    family_name: str,
    *,
    limit: int,
    split: str,
    use_grammar: bool,
    k: int,
) -> list[dict[str, Any]]:
    """Run the pipeline over a slice, recording confidence and correctness."""
    kwargs: dict[str, Any] = {"limit": limit}
    if family_name in {"repurposing_label", "grounded_qa"}:
        kwargs["split"] = split
    family = load_family(family_name, **kwargs)

    step = family.as_step_config()
    step.update(
        {"strategy": "adaptive", "k": k, "temperature": 0.7, "use_grammar": use_grammar}
    )
    pipeline = Pipeline.from_dict(
        {"name": f"calibrate::{family.name}", "steps": [step], "seed": 0},
        registry=registry,
        budget=budget,
        options=PipelineOptions(seed=0),
    )

    records: list[dict[str, Any]] = []
    with pipeline:
        for item in family.items:
            result = pipeline.run(item.inputs)
            predicted = "" if result.answer is None else str(result.answer)
            signals = (
                dict(result.trace.steps[-1].confidence_signals)
                if result.trace.steps
                else {}
            )
            records.append(
                {
                    "item_id": item.item_id,
                    "confidence": result.confidence or 0.0,
                    "correct": normalise_answer(predicted) == normalise_answer(item.gold),
                    "schema_valid": result.schema_valid,
                    "signals": signals,
                }
            )
    return records


def _schema_rate(records: Sequence[dict[str, Any]]) -> float | None:
    """Share of records whose output satisfied the schema, or ``None``."""
    judged = [r for r in records if r["schema_valid"] is not None]
    if not judged:
        return None
    return sum(1 for r in judged if r["schema_valid"]) / len(judged)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure calibration and tune the C3 threshold."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.calibrate",
        description="Risk-coverage, ECE and the C3 precision contract (Phase 10).",
    )
    parser.add_argument("--family", default="repurposing_label")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--target", type=float, default=0.9, help="precision target")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cards", default="benchmarks/results/model_cards.json")
    parser.add_argument("--out", default="benchmarks/results/calibration.json")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    if args.mock:
        from benchmarks.harness import build_registry

        registry = build_registry(mock=True, cards=args.cards)
    else:
        from slmflow.models.backends import get_backend

        registry = ModelRegistry.from_file(args.cards)
        # Register backends for every unique backend name in the loaded cards
        for card in registry:
            if card.backend not in registry._backends:
                registry.register_backend(card.backend, get_backend(card.backend))
    budget = MemoryBudget.from_device(probe().card)

    with_grammar = _collect(
        registry,
        budget,
        args.family,
        limit=args.limit,
        split=args.split,
        use_grammar=True,
        k=args.k,
    )
    without_grammar = _collect(
        registry,
        budget,
        args.family,
        limit=args.limit,
        split=args.split,
        use_grammar=False,
        k=args.k,
    )
    if not with_grammar:
        sys.stderr.write("no items collected; nothing to calibrate\n")
        return 1

    # Half fits the threshold, half tests the guarantee. Splitting matters: a
    # threshold evaluated on its own fitting data always looks calibrated.
    midpoint = max(1, len(with_grammar) // 2)
    points_fit = [CalibrationPoint(r["confidence"], r["correct"]) for r in with_grammar[:midpoint]]
    points_held = [CalibrationPoint(r["confidence"], r["correct"]) for r in with_grammar[midpoint:]]
    all_points = [CalibrationPoint(r["confidence"], r["correct"]) for r in with_grammar]

    guarantee = Precision(args.target, delta=args.delta)
    tuned: dict[str, Any] | None = None
    held_out_precision: float | None = None
    held_out_coverage: float | None = None
    guarantee_error: str | None = None
    try:
        policy = tune_threshold(points_fit, guarantee, signal="pipeline_confidence")
    except GuaranteeUnmet as exc:
        guarantee_error = str(exc)
    else:
        tuned = policy.as_row()
        answered = [p for p in points_held if p.confidence >= policy.threshold]
        if answered:
            held_out_precision = sum(1 for p in answered if p.correct) / len(answered)
            held_out_coverage = len(answered) / len(points_held)

    ece, bins = expected_calibration_error(
        [r["confidence"] for r in with_grammar], [r["correct"] for r in with_grammar]
    )

    per_signal: dict[str, Any] = {}
    for name in sorted({key for r in with_grammar for key in r["signals"]}):
        rows = [r for r in with_grammar if name in r["signals"]]
        if not rows:
            continue
        confidences = [float(r["signals"][name]) for r in rows]
        correct = [bool(r["correct"]) for r in rows]
        signal_ece, _ = expected_calibration_error(confidences, correct)
        per_signal[name] = {
            "ece": round(signal_ece, 4),
            "brier": round(brier_score(confidences, correct), 4),
            "n": len(rows),
        }

    report = {
        "family": args.family,
        "split": args.split,
        "n_items": len(with_grammar),
        "guarantee": str(guarantee),
        "tuned_policy": tuned,
        "guarantee_error": guarantee_error,
        "held_out_precision": held_out_precision,
        "held_out_coverage": held_out_coverage,
        "guarantee_held": (
            None if held_out_precision is None else held_out_precision >= args.target
        ),
        "ece": round(ece, 4),
        "reliability_bins": bins.as_rows(),
        "per_signal": per_signal,
        "empirical_coverage_at_target": coverage_at_precision(all_points, args.target),
        "risk_coverage_curve": [
            {"threshold": round(t, 4), "coverage": round(c, 4), "risk": round(risk, 4)}
            for t, c, risk in risk_coverage_curve(all_points)
        ],
        "schema_valid_rate_with_grammar": _schema_rate(with_grammar),
        "schema_valid_rate_without_grammar": _schema_rate(without_grammar),
        "mock": args.mock,
        "caveat": (
            "MOCK RUN: numbers describe MockBackend, not real models."
            if args.mock
            else "Measured on real local models; see benchmarks/results/model_cards.json."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    sys.stdout.write(f"items                 {report['n_items']}\n")
    sys.stdout.write(f"ECE                   {report['ece']}\n")
    sys.stdout.write(f"guarantee             {report['guarantee']}\n")
    if guarantee_error:
        sys.stdout.write(f"  UNMET               {guarantee_error}\n")
    else:
        sys.stdout.write(
            f"tuned threshold       {tuned['threshold'] if tuned else 'n/a'}\n"
        )
        sys.stdout.write(f"held-out precision    {held_out_precision}\n")
        sys.stdout.write(f"held-out coverage     {held_out_coverage}\n")
        sys.stdout.write(f"guarantee held        {report['guarantee_held']}\n")
    sys.stdout.write(
        f"schema valid          with grammar="
        f"{report['schema_valid_rate_with_grammar']} "
        f"without={report['schema_valid_rate_without_grammar']}\n"
    )
    if per_signal:
        sys.stdout.write("\nper-signal calibration (lower ECE is better)\n")
        for name, stats in per_signal.items():
            sys.stdout.write(
                f"  {name:<26} ece={stats['ece']:.4f} brier={stats['brier']:.4f}\n"
            )
    sys.stdout.write(f"\nwrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
