"""Phase 2: measure one model on this machine, and write its card.

Deliberately **one model per invocation**. The §2.1 acquisition ramp forbids
downloading a candidate set and benchmarking everything at once, and this
script's shape enforces that: profile a model, commit the card, then decide
whether the next download is worth it.

What is measured, all of it on the real machine:

* ``cold_load_s`` -- first load in a fresh process. Only the first load of a
  process is genuinely cold, so this script measures it once and then reloads
  for the warm figure. To get a *truly* cold number, run with ``--cold-only``
  in a fresh process, which is what the RUN.md recipe does.
* ``warm_load_s`` -- reload after an unload, with the OS page cache warm.
* ``tokens_per_s`` / ``prefill_tokens_per_s`` -- decode and prefill throughput.
* ``footprint`` -- measured VRAM and RAM, per tier, not derived from file size.
* ``accuracy_by_task`` -- accuracy on a small slice of each task family.

Nothing is estimated. A quantity that cannot be measured is written as ``null``
and the card reports itself unprofiled (Ground Rule 1).

Usage::

    python -m benchmarks.profile_models --model qwen2.5:0.5b-instruct-q4_K_M \\
        --backend ollama --params-b 0.5 --slice 20
"""

from __future__ import annotations

import argparse
import datetime as _dt
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmarks.tasks import FAMILIES, load_family
from slmflow.device import probe
from slmflow.models.backends import get_backend
from slmflow.models.registry import DEFAULT_CARDS_PATH, load_cards, save_cards
from slmflow.reliability.confidence import normalise_answer
from slmflow.resource.budget import MemoryBudget
from slmflow.resource.residency import ResourceManager
from slmflow.types import MemoryFootprint, ModelCard, Quantisation, SubTask, TaskSpec, TaskType

__all__ = ["profile_model", "main"]

_PROBE_PROMPT = (
    "Answer in one word. What is the capital city of France?"
)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile, or ``None`` for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def profile_model(
    model_id: str,
    *,
    backend_name: str,
    params_b: float,
    quantisation: str = "Q4_K_M",
    repeats: int = 5,
    slice_size: int = 20,
    context_tokens: int = 2048,
    n_gpu_layers: int = -1,
    model_dir: str = "models",
    families: Sequence[str] = ("repurposing_label",),
    supports_grammar: bool | None = None,
    supports_logprobs: bool | None = None,
) -> ModelCard:
    """Measure one model and return its card.

    Args:
        model_id: Backend-specific model identifier.
        backend_name: ``llama_cpp``, ``ollama`` or ``mock``.
        params_b: Parameter count in billions, for the record.
        quantisation: Quantisation label, for the record.
        repeats: Generation repeats for the throughput median.
        slice_size: Items per task family for the accuracy slice.
        context_tokens: Context window to load with. Costs VRAM, so it is part
            of the measured configuration rather than a free choice.
        n_gpu_layers: llama.cpp offload; ``-1`` offloads everything.
        model_dir: Where GGUF files live.
        families: Task families to measure accuracy on.
        supports_grammar: Override capability detection.
        supports_logprobs: Override capability detection.

    Returns:
        A :class:`ModelCard` with measured values, and ``None`` for anything
        that could not be measured.
    """
    device = probe()
    kwargs: dict[str, Any] = {}
    if backend_name == "llama_cpp":
        kwargs = {
            "model_dir": model_dir,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": context_tokens,
        }
    backend = get_backend(backend_name, **kwargs)

    if not backend.is_available():
        sys.stderr.write(
            f"backend {backend_name!r} is not available on this machine. "
            f"Run `slmflow doctor` for the install hint.\n"
        )
        raise SystemExit(2)

    try:
        quant = Quantisation(quantisation)
    except ValueError:
        quant = Quantisation.UNKNOWN

    digest = None
    getter = getattr(backend, "digest_for", None)
    if callable(getter):
        digest = getter(model_id)

    provisional = ModelCard(
        model_id=model_id,
        backend=backend_name,
        params_b=params_b,
        quantisation=quant,
        digest=digest,
        context_tokens=context_tokens,
        supports_grammar=(
            supports_grammar if supports_grammar is not None else backend_name == "llama_cpp"
        ),
        supports_logprobs=(
            supports_logprobs if supports_logprobs is not None else backend_name == "llama_cpp"
        ),
    )

    # -- cold load ------------------------------------------------------- #
    sys.stdout.write(f"[1/4] cold load of {model_id} ...\n")
    cold_start = time.perf_counter()
    backend.load(provisional)
    cold_load_s = time.perf_counter() - cold_start
    footprint = backend.measure_footprint(provisional)
    sys.stdout.write(
        f"      cold_load_s={cold_load_s:.2f}  "
        f"vram={footprint.vram_bytes / 1e9:.2f} GB  ram={footprint.ram_bytes / 1e9:.2f} GB\n"
    )

    # -- throughput ------------------------------------------------------ #
    sys.stdout.write(f"[2/4] throughput over {repeats} generations ...\n")
    decode_rates: list[float] = []
    prefill_rates: list[float] = []
    for _ in range(repeats):
        generation = backend.generate(
            provisional, _PROBE_PROMPT, max_tokens=64, temperature=0.0, seed=0
        )
        if generation.latency_s > 0 and generation.output_tokens > 0:
            decode_rates.append(generation.output_tokens / generation.latency_s)
        if generation.prompt_tokens > 0 and generation.latency_s > 0:
            prefill_rates.append(generation.prompt_tokens / max(generation.latency_s, 1e-6))
    tokens_per_s = statistics.median(decode_rates) if decode_rates else None
    prefill_tokens_per_s = statistics.median(prefill_rates) if prefill_rates else None
    sys.stdout.write(
        f"      tokens_per_s={_show(tokens_per_s)}  "
        f"p10={_show(_percentile(decode_rates, 0.1))} "
        f"p90={_show(_percentile(decode_rates, 0.9))}\n"
    )

    # -- warm reload ------------------------------------------------------ #
    sys.stdout.write("[3/4] warm reload ...\n")
    backend.unload(provisional)
    warm_start = time.perf_counter()
    backend.load(provisional)
    warm_load_s = time.perf_counter() - warm_start
    sys.stdout.write(f"      warm_load_s={warm_load_s:.2f}\n")

    # -- accuracy slices --------------------------------------------------- #
    sys.stdout.write(f"[4/4] accuracy on {slice_size} items per family ...\n")
    accuracy: dict[str, float] = {}
    card_for_run = replace(
        provisional,
        footprint=footprint,
        cold_load_s=cold_load_s,
        warm_load_s=warm_load_s,
        tokens_per_s=tokens_per_s,
        prefill_tokens_per_s=prefill_tokens_per_s,
    )
    budget = MemoryBudget.from_device(device.card)
    if not budget.fits(footprint):
        sys.stderr.write(
            f"warning: measured footprint exceeds the derived budget "
            f"({budget.describe()}); accuracy slices are skipped.\n"
        )
    else:
        manager = ResourceManager(budget, measure_on_load=False)
        manager.register(card_for_run, backend)
        for family_name in families:
            if family_name not in FAMILIES:
                sys.stderr.write(f"warning: unknown family {family_name!r}, skipped\n")
                continue
            family = load_family(family_name, limit=slice_size)
            if not len(family):
                continue
            hits = 0
            for item in family.items:
                prompt = family.prompt_template.format(**item.inputs)
                if family.output_schema:
                    prompt += "\n\nReply with JSON only."
                subtask = SubTask(
                    name=family.name,
                    spec=TaskSpec(
                        prompt=prompt,
                        task_type=family.task_type,
                        output_schema=family.output_schema,
                        max_output_tokens=family.max_output_tokens,
                    ),
                )
                with manager.resident(card_for_run) as handle:
                    generation = handle.backend.generate(
                        card_for_run,
                        subtask.spec.prompt,
                        max_tokens=family.max_output_tokens,
                        temperature=0.0,
                        seed=0,
                    )
                if normalise_answer(generation.text) == normalise_answer(item.gold):
                    hits += 1
                elif normalise_answer(item.gold) in normalise_answer(generation.text):
                    hits += 1
            accuracy[str(family.task_type)] = hits / len(family)
            sys.stdout.write(
                f"      {family.name:<20} accuracy={hits / len(family):.3f} "
                f"(n={len(family)}, majority baseline={family.majority_baseline():.3f})\n"
            )
        manager.evict_all()

    backend.unload(card_for_run)

    return replace(
        card_for_run,
        accuracy_by_task=accuracy,
        measured_on_device=(
            f"{device.card.cpu_model} / {device.card.gpu_model or 'no gpu'} / "
            f"{device.card.total_ram_bytes / 1e9:.0f}GB RAM"
        ),
        measured_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        notes=(
            f"profiled by benchmarks/profile_models.py; repeats={repeats}, "
            f"slice={slice_size}, n_ctx={context_tokens}, n_gpu_layers={n_gpu_layers}"
        ),
    )


def _show(value: float | None) -> str:
    """Render a measured value or its honest absence."""
    return "not measured" if value is None else f"{value:.2f}"


def main(argv: Sequence[str] | None = None) -> int:
    """Profile one model and merge its card into the registry file."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.profile_models",
        description="Measure one model on this machine and write its card.",
    )
    parser.add_argument("--model", required=True, help="backend model id")
    parser.add_argument(
        "--backend", default="ollama", choices=["ollama", "llama_cpp", "mock"]
    )
    parser.add_argument("--params-b", type=float, required=True)
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--slice", type=int, default=20, dest="slice_size")
    parser.add_argument("--n-ctx", type=int, default=2048, dest="context_tokens")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument(
        "--family", action="append", default=[], help="task family; repeatable"
    )
    parser.add_argument("--cards", default=str(DEFAULT_CARDS_PATH))
    args = parser.parse_args(argv)

    card = profile_model(
        args.model,
        backend_name=args.backend,
        params_b=args.params_b,
        quantisation=args.quant,
        repeats=args.repeats,
        slice_size=args.slice_size,
        context_tokens=args.context_tokens,
        n_gpu_layers=args.n_gpu_layers,
        model_dir=args.model_dir,
        families=args.family or ("repurposing_label",),
    )

    path = Path(args.cards)
    existing = load_cards(path) if path.exists() else []
    merged = [c for c in existing if c.model_id != card.model_id] + [card]
    save_cards(merged, path)

    sys.stdout.write("\n")
    sys.stdout.write(f"card written to {path}\n")
    sys.stdout.write(f"  is_profiled       {card.is_profiled}\n")
    sys.stdout.write(f"  digest            {card.digest or 'not recorded'}\n")
    sys.stdout.write(f"  resident bytes    {card.footprint.resident_bytes / 1e9:.2f} GB\n")
    sys.stdout.write(f"  cold / warm load  {_show(card.cold_load_s)} / {_show(card.warm_load_s)} s\n")
    sys.stdout.write(f"  tokens per second {_show(card.tokens_per_s)}\n")
    if not card.is_profiled:
        sys.stdout.write(
            "\nWARNING: this card is not fully profiled. The scheduler will refuse to "
            "cost it and will report regret as unavailable.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
