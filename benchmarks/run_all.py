"""One-command end-to-end run for a fresh machine.

Runs, in order: environment probe, model pull and profiling, regret, router
training, calibration, the full harness (validation split, then held-out
test split). Each stage is a separate subprocess, so one failure is reported
and the remaining stages still run unless ``--stop-on-error`` is given.

    python -m benchmarks.run_all --dry-run          # print the plan only
    python -m benchmarks.run_all --seeds 3          # the real run

Model specs are ``id:params_b[:backend]``, for example
``llama3.2:3b-instruct-q4_K_M:3.0``. The id may itself contain colons, so the
last one or two colon-separated fields are parsed from the right.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

__all__ = ["DEFAULT_MODELS", "main", "parse_model_spec"]

DEFAULT_MODELS = (
    "qwen2.5:0.5b-instruct-q4_K_M:0.5",
    "qwen2.5:1.5b-instruct-q4_K_M:1.5",
    "llama3.2:3b-instruct-q4_K_M:3.0",
)

_BACKENDS = ("ollama", "llama_cpp", "mock")


def parse_model_spec(spec: str) -> tuple[str, float, str]:
    """Split ``id:params_b[:backend]`` into ``(id, params_b, backend)``.

    Example:
        >>> parse_model_spec("llama3.2:3b:3.0")
        ('llama3.2:3b', 3.0, 'ollama')
        >>> parse_model_spec("m.gguf:7.0:llama_cpp")
        ('m.gguf', 7.0, 'llama_cpp')
    """
    parts = spec.split(":")
    backend = "ollama"
    if parts[-1] in _BACKENDS:
        backend = parts.pop()
    try:
        params_b = float(parts.pop())
    except (ValueError, IndexError) as exc:
        raise ValueError(f"model spec {spec!r} must end in :<params_b>[:<backend>]") from exc
    model_id = ":".join(parts)
    if not model_id:
        raise ValueError(f"model spec {spec!r} has no model id")
    return model_id, params_b, backend


def _run(label: str, cmd: list[str], *, dry_run: bool, results: list[tuple[str, int, float]]) -> int:
    """Run one stage, stream its output, and record the exit code."""
    sys.stdout.write(f"\n=== {label}\n$ {' '.join(cmd)}\n")
    sys.stdout.flush()
    if dry_run:
        results.append((label, 0, 0.0))
        return 0
    started = time.perf_counter()
    code = subprocess.call(cmd)
    results.append((label, code, time.perf_counter() - started))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.run_all", description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), metavar="SPEC")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="cap items per family")
    parser.add_argument("--slice", type=int, default=20, dest="slice_size")
    parser.add_argument("--skip-pull", action="store_true", help="models are already pulled")
    parser.add_argument("--skip-profile", action="store_true", help="reuse model_cards.json")
    parser.add_argument("--skip-test-split", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="delete old model cards first")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = parser.parse_args(argv)

    specs = [parse_model_spec(s) for s in args.models]
    py = sys.executable
    cards = Path("benchmarks/results/model_cards.json")
    results: list[tuple[str, int, float]] = []
    limit = ["--limit", str(args.limit)] if args.limit else []

    def stage(label: str, cmd: list[str]) -> bool:
        code = _run(label, cmd, dry_run=args.dry_run, results=results)
        return code == 0 or not args.stop_on_error

    if args.fresh and cards.exists() and not args.dry_run:
        cards.unlink()

    if not stage("probe environment", [py, "-m", "slmflow.cli", "probe"]):
        return 1

    if not args.skip_profile:
        for model_id, params_b, backend in specs:
            if backend == "ollama" and not args.skip_pull and shutil.which("ollama"):
                if not stage(f"pull {model_id}", ["ollama", "pull", model_id]):
                    return 1
            ok = stage(
                f"profile {model_id}",
                [py, "-m", "benchmarks.profile_models", "--model", model_id,
                 "--backend", backend, "--params-b", str(params_b),
                 "--slice", str(args.slice_size), "--cards", str(cards)],
            )
            if not ok:
                return 1

    seeds = ["--seeds", str(args.seeds)]
    pipeline = [
        ("regret", [py, "-m", "benchmarks.regret", "--sizes", "2,3,4"]),
        ("train router", [py, "-m", "benchmarks.train_router",
                          "--family", "repurposing_label", "--limit", "60"]),
        ("calibrate", [py, "-m", "benchmarks.calibrate",
                       "--family", "repurposing_label", "--limit", "80", "--target", "0.9"]),
        ("harness (validation split)",
         [py, "-m", "benchmarks.harness", "--all", "--retrieval", *seeds, *limit]),
    ]
    if not args.skip_test_split:
        pipeline.append(
            ("harness (held-out test split)",
             [py, "-m", "benchmarks.harness", "--all", "--retrieval", "--split", "test",
              "--out", "benchmarks/results/test_split",
              "--results-md", "docs/RESULTS_TEST.md", *seeds, *limit])
        )
    for label, cmd in pipeline:
        if not stage(label, cmd):
            return 1

    sys.stdout.write("\n=== summary\n")
    for label, code, seconds in results:
        status = "ok" if code == 0 else f"FAILED ({code})"
        sys.stdout.write(f"  {status:<12} {seconds:8.1f}s  {label}\n")
    return 0 if all(code == 0 for _, code, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
