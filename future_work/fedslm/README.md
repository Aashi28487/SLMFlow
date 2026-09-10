# fedslm — demoted 2026-09-10

Federated-learning direction, demoted to future work per `BUILD_BRIEF.md` Ground Rule 4.
The thesis is SLM orchestration under a memory budget; carrying both directions would
weaken the evaluation of each. Not deleted, not built against, not imported by `slmflow`.

The data-download pipeline that once lived here produced `data/raw/repurposing/` and
`data/processed/repurposing/`, which the orchestration work **does** reuse.
