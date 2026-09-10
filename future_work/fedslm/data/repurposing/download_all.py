"""One command to rebuild the whole drug-repurposing dataset from scratch.

    cd implementation/src/fedslm/data/repurposing
    python download_all.py

Set NCBI_API_KEY first if you have one - it triples the PubMed fetch rate.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
STEPS = [
    ("PubMed abstracts", ["fetch_pubmed.py", "--per-drug", "60", "--per-pair", "25"]),
    ("ClinicalTrials.gov studies", ["fetch_ctgov.py", "--per-drug", "60",
                                    "--per-pair", "20", "--general", "300"]),
    ("openFDA labels", ["fetch_openfda.py"]),
    ("Build corpus + partitions", ["build_dataset.py"]),
    ("Dataset card", ["make_card.py"]),
]


def main() -> int:
    for name, cmd in STEPS:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, *cmd], cwd=HERE)
        if rc != 0:
            print(f"FAILED: {name} (exit {rc})", file=sys.stderr)
            return rc
        print(f"-- {name} done in {time.time() - t0:.0f}s", flush=True)
    print("\nAll steps complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
