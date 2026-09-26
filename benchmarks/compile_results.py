"""Standalone script to compile raw trace outputs into summary.json and docs/RESULTS.md."""

import json
from pathlib import Path
from dataclasses import asdict
from typing import Any

from benchmarks.harness import (
    CONFIGURATIONS,
    BASELINE,
    ItemOutcome,
    RunOutcome,
    aggregate,
    significance,
    environment_record,
    render_markdown,
    build_registry,
    probe,
    load_family,
)

def compile_results():
    out_dir = Path("benchmarks/results")
    raw_dir = out_dir / "raw"
    
    device = probe()
    registry = build_registry(mock=False, cards="benchmarks/results/model_cards.json")
    
    family_names = ["grounded_qa", "repurposing_label", "trial_lookup"]
    families = []
    for name in family_names:
        kwargs = {}
        if name in {"repurposing_label", "grounded_qa"}:
            kwargs["split"] = "train"
        families.append(load_family(name, **kwargs))
    
    aggregates: list[dict[str, Any]] = []
    all_runs: dict[str, list[RunOutcome]] = {}
    
    for family in families:
        for conf in CONFIGURATIONS:
            runs: list[RunOutcome] = []
            for seed in range(3):
                items_file = raw_dir / f"{conf.name}_{family.name}_seed{seed}_items.json"
                if not items_file.exists():
                    print(f"Skipping missing items file: {items_file}")
                    continue
                
                raw_items = json.loads(items_file.read_text(encoding="utf-8"))
                items = []
                for item in raw_items:
                    items.append(ItemOutcome(
                        item_id=item["item_id"],
                        predicted=item["predicted"],
                        gold=item["gold"],
                        correct=item["correct"],
                        confidence=item["confidence"],
                        abstained=item["abstained"],
                        schema_valid=item.get("schema_valid"),
                        seconds=item["seconds"],
                        swaps=item["swaps"],
                        swap_seconds=item["swap_seconds"],
                        generations=item["generations"],
                        output_tokens=item["output_tokens"],
                        peak_vram_bytes=item["peak_vram_bytes"],
                        peak_ram_bytes=item["peak_ram_bytes"],
                        models_used=tuple(item.get("models_used", [])),
                        groundedness=item.get("groundedness"),
                        error=item.get("error")
                    ))
                
                run = RunOutcome(configuration=conf.name, family=family.name, seed=seed, items=items)
                runs.append(run)
            
            if runs:
                aggregates.append(aggregate(runs))
                all_runs.setdefault(conf.name, []).extend(runs)
                print(f"Aggregated {conf.name:<20} {family.name:<20} acc={runs[0].accuracy:.3f}")

    sig_rows = significance(all_runs) if all_runs else []
    environment = environment_record(device, registry)
    
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "environment": environment,
                "aggregates": aggregates,
                "significance": sig_rows,
                "seeds": 3,
                "split": "train",
                "limit": None,
                "mock": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    
    markdown = render_markdown(aggregates, sig_rows, families, environment)
    results_md = Path("docs/RESULTS.md")
    results_md.write_text(markdown, encoding="utf-8")
    print(f"\nWrote {results_md} and {summary_path}")

if __name__ == "__main__":
    compile_results()
