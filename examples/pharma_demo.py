"""Phase 19: the pharma demonstration, entirely local, with citations.

    python examples/pharma_demo.py --drug metformin --indication cancer
    python examples/pharma_demo.py --drug metformin --indication cancer --mock

Retrieves from the real DRE-Corpus in ``data/``, runs the four-step pipeline in
``examples/pharma_rag.yaml``, and prints the report with the citations its
claims trace to. Nothing leaves the machine: no hosted API is reachable from
this code path.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from slmflow.models.registry import ModelRegistry
from slmflow.orchestration.pipeline import Pipeline, PipelineOptions
from slmflow.rag.store import BM25Retriever, load_passages
from slmflow.reliability.confidence import RetrievalSupport
from slmflow.types import Generation

GB = 1_000_000_000


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demonstration."""
    parser = argparse.ArgumentParser(description="Local drug-repurposing evidence report.")
    parser.add_argument("--drug", default="metformin")
    parser.add_argument("--indication", default="cancer")
    parser.add_argument("--config", default="examples/pharma_rag.yaml")
    parser.add_argument("--cards", default="benchmarks/results/model_cards.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--corpus-limit", type=int, default=6000)
    parser.add_argument("--trace", default="runs/pharma_demo.jsonl")
    parser.add_argument("--mock", action="store_true", help="no weights needed")
    args = parser.parse_args(argv)

    print(f"loading corpus (first {args.corpus_limit} passages) ...")
    passages = load_passages(limit=args.corpus_limit)
    retriever = BM25Retriever(passages)
    print(f"  {len(passages)} passages indexed\n")

    query = f"{args.drug} {args.indication} clinical trial evidence off-label"
    hits = retriever.search(query, args.top_k)
    if not hits:
        print(f"No passages matched {query!r}. Try a different drug or indication.")
        return 1

    print(f"--- retrieved evidence for {args.drug} / {args.indication} ---")
    for index, (passage, score) in enumerate(hits, start=1):
        print(f"[{index}] ({score:.2f}) {passage.title[:88]}")
        print(f"     {passage.cite()[:110]}")
    print()

    if args.mock:
        from benchmarks.harness import build_registry

        registry = build_registry(mock=True, cards=args.cards)
    else:
        registry = ModelRegistry.from_file(args.cards)

    pipeline = Pipeline.from_config(
        args.config,
        registry=registry,
        options=PipelineOptions(trace_path=args.trace, seed=0),
    )

    print(f"budget            {pipeline.budget.describe()}")
    print(f"models            {[c.model_id for c in pipeline.pool]}")
    print()

    with pipeline:
        result = pipeline.run(
            {"drug": args.drug, "indication": args.indication},
            retrieved=[p.text for p, _ in hits],
        )

    print("--- report ---")
    if result.abstained:
        print(f"ABSTAINED: {result.abstention_reason}")
        print("The pipeline declined rather than guess. That is the C3 contract working.")
    else:
        print(result.answer)
    print()

    groundedness = RetrievalSupport().score(
        (Generation(text=str(result.answer or ""), model_id=""),),
        {"retrieved": [p.text for p, _ in hits]},
    )

    print("--- cost and reliability, measured ---")
    trace = result.trace
    print(f"confidence        {result.confidence if result.confidence is None else round(result.confidence, 3)}")
    print(f"groundedness      {groundedness:.3f}  (token overlap with retrieved evidence)")
    print(f"steps             {[s.step for s in trace.steps]}")
    print(f"models used       {', '.join(trace.models_used) or '-'}")
    print(f"model loads       {trace.total_swaps}, {trace.swap_seconds:.2f} s")
    print(f"samples drawn     {trace.total_generations}")
    print(f"output tokens     {trace.output_tokens}")
    print(f"peak memory       vram {trace.peak_vram_bytes / GB:.2f} GB, "
          f"ram {trace.peak_ram_bytes / GB:.2f} GB")
    print(f"wall clock        {trace.wall_s:.2f} s")
    print(f"trace             {args.trace}")
    print()
    print("--- citations ---")
    for index, (passage, _) in enumerate(hits, start=1):
        print(f"[{index}] {passage.cite()}")

    if args.mock:
        print()
        print("NOTE: --mock was used. The report text is scripted, not generated.")
    return 3 if result.abstained else 0


if __name__ == "__main__":
    sys.exit(main())
