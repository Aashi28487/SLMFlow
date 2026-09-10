# Model selection (Phase 3)

> **NOT YET MEASURED.** Phase 3 consumes the Phase 2b sweep, which has not run.
> No model has been downloaded. This file records the *method* and the decision
> criteria so the analysis is specified before the data exists — which is the
> right order, since choosing criteria after seeing results is how conclusions
> get fitted to noise.
>
> Populate it with:
>
> ```
> python -m benchmarks.profile_models --model <id> --backend ollama --params-b <n>
> python -m benchmarks.train_router --family repurposing_label
> ```

## Candidate set (planned, not yet acquired)

Follows the §2.1 acquisition ramp: one model at a time, measured and committed
before the next download. Total GGUF capped at ~40 GB against the 512 GB SSD
shared with `data/`.

| Stage | Model class | Q4_K_M size | Role | Status |
|---|---|---|---|---|
| A | 0.5B instruct | ~0.4 GB | first slice; T0 candidate; tiny verifier | not acquired |
| B | 3B instruct | ~2.0 GB | workhorse; second router arm | not acquired |
| C | 1.5B instruct | ~1.1 GB | middle cascade rung | not acquired |
| C | 4B instruct | ~2.7 GB | quality arm | not acquired |
| C | alternate 3B family | ~2.0 GB | **complementarity**, not capability | not acquired |
| D | 7-8B instruct | ~4.4-4.9 GB | **Phase 16 big-model baseline** | not acquired |

The Stage C alternate-family entry matters more than another size step:
SLM-MUX selection rewards **complementarity**, so two models that fail on
*different* items are worth more than one slightly stronger model.

## Selection criteria, fixed in advance

1. **Cost-normalised accuracy**, not accuracy. `accuracy/second` and
   `accuracy/GB-peak`. A model three points better and four times slower loses.
2. **Complementarity** via `select_model_set`, maximising
   `UnionAcc - lambda * Contradiction` (`strategies/slm_mux.py`).
3. **Co-residency**, which is this project's addition to the paper's criterion.
   A set whose members cannot be resident together forces a swap *per query*.
   `peak_bytes` is recorded per candidate set so the cost is visible rather than
   hidden behind a better objective score.
4. **Capability floor.** Grammar support (llama.cpp) and log-prob support decide
   whether the cheap `LogProbMargin` confidence signal is available at all —
   which in turn decides whether reliability costs one sample or `k`.
5. **Majority-class sanity.** `repurposing_label` is 75.7 % `investigational`.
   A model below that is worse than a constant predictor and is rejected
   regardless of any other property.

## Complementarity matrix (to be filled)

| | model A | model B | model C |
|---|---|---|---|
| solo accuracy | not measured | not measured | not measured |
| union with A | — | not measured | not measured |
| contradiction with A | — | not measured | not measured |
| co-resident within budget | — | not measured | not measured |

## What would falsify the multi-model premise

Stated in advance so it cannot be argued away later. If the sweep shows

- one model dominating every task family on cost-normalised accuracy, **and**
- union accuracy barely exceeding the best solo model,

then orchestration buys nothing on these tasks, and that is the finding to
report (Ground Rule 9). The framework would remain a contribution as a
**scheduler** — C1 does not depend on ensembling — but the ensemble claim would
have to be withdrawn rather than defended.
