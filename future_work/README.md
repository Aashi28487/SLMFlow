# future_work/

Demoted 2026-09-10 per `BUILD_BRIEF.md` Ground Rule 4: the thesis is **SLM orchestration**
(`src/slmflow/`), not federated learning. Nothing here is deleted, imported, or built against.

- `fedslm/` — the federated-learning direction (LoRA adapters, simulated hospitals, Flower).
  Was `src/fedslm/`. Contained only the `data/repurposing/` download pipeline directories.
- `docs_fedslm/` — `01_PRD.md`, `02_TECH_STACK.md`, `03_UI_LAYOUT.md`, all written for that
  direction. Were `docs/`.

`data/` deliberately stays at the repo root: the DRE-Corpus is shared and is reused by the
orchestration work (RAG index in Phase 9, benchmark task families in Phase 12, pharma demo
in Phase 19).
