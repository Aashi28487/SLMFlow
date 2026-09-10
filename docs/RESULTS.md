# Results

> **Status: NOTHING HAS BEEN MEASURED YET.**
>
> This file is a template with the structure the Phase 12 harness will fill in.
> Every cell reads `not yet measured` because no benchmark has been run on real
> hardware. Per Ground Rule 1, no number appears here until it comes from code
> actually executed on the evaluation machine.
>
> Regenerate this file with:
>
> ```
> python -m benchmarks.harness --all --seeds 3 --retrieval
> ```
>
> The harness overwrites this file wholesale, including the environment record
> and the significance tests. A run with `--mock` stamps a warning at the top;
> mock numbers describe `MockBackend` and must never be cited.

---

## Environment

Filled in by `benchmarks.harness.environment_record`. Required in every
artifact per Ground Rule 11.

| Property | Value |
|---|---|
| slmflow_version | 0.1.0 |
| cpu | not yet measured |
| logical_cores | not yet measured |
| total_ram_bytes | not yet measured |
| gpu | not yet measured |
| total_vram_bytes | not yet measured |
| cuda_available | not yet measured |
| device_tier | not yet measured |
| rss_measurement | not yet measured |
| model_digests | none — no model downloaded |

---

## Task families

| family | items | metric | majority-class baseline | label distribution |
|---|---|---|---|---|
| repurposing_label | 140 (train) / 64 (test) | exact_match | **0.757** (computed from the corpus) | investigational=106, approved=24, hypothesis=10 |
| trial_lookup | not yet measured | json_exact_match | n/a | n/a |
| grounded_qa | 333 (train) / 64 (test) | groundedness | n/a | n/a |

> The majority-class baseline is the one number here not requiring a model run:
> it comes from `stats.json`. **Any accuracy below 0.757 on `repurposing_label`
> is worse than a constant predictor.** The harness prints this row first for
> exactly that reason.

---

## Quality and cost together

`context.md` §8 requires quality and cost in one table.

| configuration | accuracy | answered acc | coverage | schema valid | s/item | swap s/item | gens/item | peak VRAM GB | peak RAM GB | acc/s | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single_smallest | not yet measured | | | | | | | | | | |
| single_best | not yet measured | | | | | | | | | | |
| slm_mux | not yet measured | | | | | | | | | | |
| agent_forest | not yet measured | | | | | | | | | | |
| adaptive_sc | not yet measured | | | | | | | | | | |
| cascade | not yet measured | | | | | | | | | | |

Reading guide, which travels with the table:

- `accuracy` counts an abstention as **incorrect**.
- `answered acc` is precision on the answered subset and is meaningless without
  `coverage` beside it.
- `gens/item` is the C4 cost number.
- `swap s/item` is the cost no cloud-oriented framework measures.

---

## C1 — scheduler regret and the swap-awareness ablation

From `python -m benchmarks.regret`.

| steps | shape | greedy loads | naive loads | loads saved | oracle loads | seconds regret | relative | quality regret |
|---|---|---|---|---|---|---|---|---|
| 2 | independent | not yet measured | | | | | | |
| 3 | independent | not yet measured | | | | | | |
| 4 | independent | not yet measured | | | | | | |
| 2–4 | chain | not yet measured | | | | | | |

**The honest ablation the brief demands.** What does swap-aware scheduling buy
over naive load-on-demand with LRU eviction? Report it *even if the answer is
"little, because model loads are amortised"* — that is a finding, not a failure
(Ground Rule 9).

**Caveat that must stay attached to these numbers.** `benchmarks/regret.py`
computes cost from measured `ModelCard` values, so it is a **model-based
prediction**, not end-to-end wall clock. Wall-clock regret comes from the
`ablate_scheduler` configuration in the harness, which runs both schedules for
real.

---

## Phase 7 — router regret

From `python -m benchmarks.train_router`.

| router | accuracy | oracle accuracy | regret | relative regret |
|---|---|---|---|---|
| learned | not yet measured | | | |
| heuristic | not yet measured | | | |
| random | not yet measured | | | |
| static:<each model> | not yet measured | | | |

Cost-normalised accuracy (accuracy/second and accuracy/GB-peak) is the metric
that makes the "small beats big on a laptop" argument concrete. Reported in the
main table above.

---

## C3 — abstention as a declarative contract

From `python -m benchmarks.calibrate`.

| quantity | value |
|---|---|
| target precision | not yet measured |
| tuned threshold | not yet measured |
| Clopper-Pearson lower bound | not yet measured |
| achieved precision (fit split) | not yet measured |
| **held-out precision** | not yet measured |
| **held-out coverage** | not yet measured |
| guarantee held on held-out data | not yet measured |
| ECE | not yet measured |

Per-signal calibration comparison — this is what makes "we address SLM-MUX's
stated weakness" a measurement rather than a claim:

| signal | ECE | Brier | cost per item |
|---|---|---|---|
| self_consistency | not yet measured | | k samples |
| logprob_margin | not yet measured | | **1** sample |
| cross_model_agreement | not yet measured | | \|M\| samples |
| retrieval_support | not yet measured | | free |

Schema validity, before and after grammar constraints:

| condition | schema-valid rate |
|---|---|
| with GBNF grammar (llama.cpp) | not yet measured |
| validation only, no grammar | not yet measured |

Risk-coverage curve: written to `benchmarks/results/calibration.json`.

---

## Phase 9 — retrieval quality, reported separately

| retriever | k | recall@k | mean relevant retrieved | n queries |
|---|---|---|---|---|
| bm25 | 5 | not yet measured | | |
| dense (MiniLM + FAISS) | 5 | not yet measured | | |

Reported separately from answer quality so a failure is attributable to
retrieval or to generation, not to "the pipeline".

---

## Phase 16 — the headline validation

The question the thesis turns on: **does orchestrating small models beat one
bigger model on this hardware?**

| system | accuracy | s/item | peak VRAM GB | acc/s | acc/GB |
|---|---|---|---|---|---|
| best orchestrated configuration | not yet measured | | | | |
| single 7–8B Q4_K_M baseline | not yet measured | | | | |

The corrected hardware makes this answerable **on the same machine**: a 7–8B
Q4_K_M is ~4.4–4.9 GB and fits in 6 GB VRAM at modest context. No borrowed GPU
box is needed. Verify the fit by loading and measuring, never by arithmetic on
the file size.

---

## Significance

| configuration | difference | 95% CI | p (bootstrap) | McNemar p | significant |
|---|---|---|---|---|---|
| — | not yet measured | | | | |

**No difference may be claimed from an empty table.** The harness runs the
paired bootstrap and McNemar unconditionally against `single_best`, so any
published row carries its test.

---

## Negative results

Ground Rule 9: this section is expected to have content, and a brief with only
wins is a brief that lied. Candidates to check honestly:

- Orchestration overhead may exceed its benefit on CPU for some task family.
- Swap-aware scheduling may buy little because loads amortise over a run.
- Self-consistency may not correlate with correctness on 0.5B models, which
  would make the whole confidence apparatus weaker than hoped.
- BM25 may beat dense retrieval on this corpus, making the `[rag]` extra
  unnecessary.

None of these is yet measured either way.
