# evals/ — the frozen golden set

**Empty by design.** Phase 12's Definition of Done includes a frozen golden set
in `evals/` that acts as a regression guard: a small fixed set of items with
their expected outputs, so a later refactor cannot silently change behaviour.

It is not here yet, and it cannot be, because **the expected outputs must come
from a real run**. Writing plausible expected outputs by hand would breach
Ground Rule 1 as thoroughly as writing plausible benchmark numbers — the golden
set would then encode what someone imagined the models would say, and every
future regression check would validate against fiction.

## How to generate it, after `RUN.md` step 6

Once two models are profiled and the vertical slice is green:

```bash
# 1. Run a small fixed slice with a fixed seed
python -m benchmarks.harness \
    --family repurposing_label \
    --limit 20 --seeds 1 \
    --out evals/golden_run

# 2. Inspect the per-item outputs and confirm they are sane
cat evals/golden_run/raw/single_best_repurposing_label_seed0_items.json

# 3. Freeze them, recording the exact conditions
cp evals/golden_run/raw/single_best_repurposing_label_seed0_items.json \
   evals/golden_repurposing_label.json
```

## What the frozen file must record alongside the outputs

A golden set without its conditions is not reproducible:

- the **model digests** (from `slmflow models`), because a re-pulled model with
  the same tag is not the same weights;
- the **seed**, and the config digest from the trace;
- the **device card**, since throughput and therefore any timing assertion is
  machine-specific;
- the **library version**.

`benchmarks/harness.py` already writes all of these into
`benchmarks/results/summary.json` under `environment`; copy that block in beside
the items.

## What the regression test should assert

- exact-match on the **answers**, which are deterministic at temperature 0 with
  a fixed seed;
- the **schema-validity** flag per item;
- that **abstention decisions** are unchanged;
- **not** wall-clock timings, and **not** memory figures. Those vary with
  machine load and would make the guard flap, which trains people to ignore it.

Sampling strategies (`slm_mux`, `agent_forest`, `adaptive`) are stable only for
a fixed seed *and* a fixed backend build. Prefer `single` at temperature 0 for
the golden set, and cover the sampling strategies with the property tests in
`tests/` instead.
