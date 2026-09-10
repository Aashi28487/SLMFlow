# Decisions log

Append-only. Ground Rule 8: every judgment call gets a dated entry saying what
was ambiguous, what was chosen, why, and what was given up. This is the audit
trail and the raw material for the thesis's related-work and limitations
sections.

---

## 2026-09-10 — Scope conflict resolved in favour of orchestration

**Ambiguous.** `context.md` describes an SLM orchestration framework;
`docs/01_PRD.md` and `src/fedslm/` describe a federated-learning direction
(FedSLM). Both are large. Ground Rule 4 requires resolving this without asking.

**Chosen.** Orchestration is the thesis. Moved `src/fedslm/` →
`future_work/fedslm/` and the three `docs/0*.md` → `future_work/docs_fedslm/`,
each with a dated README explaining the demotion. `data/` stays at the repo
root because the DRE-Corpus is shared and is reused by Phase 9, Phase 12 and
Phase 19.

**Given up.** The federated direction, including the existing simulated-hospital
partitions in `data/partitions/`. Nothing is deleted, so it remains available as
future work.

---

## 2026-09-10 — Hardware corrected: the machine has a GPU

**Ambiguous.** `BUILD_BRIEF.md` §2 and `context.md` §6 both described the
evaluation machine as "Intel i5, Iris iGPU, **no CUDA, no usable VRAM**". The
actual machine has a discrete GPU with ~6 GB VRAM and working CUDA.

**Chosen.** Corrected both source documents with dated correction notes, and
made the memory budget a **vector** `(B_vram, B_ram)` rather than a scalar
throughout: `MemoryBudget`, `MemoryFootprint`, `ResourceExhausted`,
`StepTrace.peak_*_bytes` and every results row account per tier.

**Why this strengthens the work.** A two-level hierarchy (VRAM over RAM over
SSD, with different promotion costs per level) makes MBOP a genuine **two-level
paging problem with variable page sizes and variable miss costs** — a sharper
instance than a CPU-only machine would have produced. It also makes T2 (a
borrowed GPU box) unnecessary: a 7–8B Q4_K_M is ~4.4–4.9 GB and fits in 6 GB, so
Phase 16's central "orchestration versus one bigger model" comparison runs on
the same machine, measured. That was previously the most fragile dependency in
the plan.

**Given up.** Nothing, but a new obligation: CUDA must stay inside the
`llama_cpp` backend behind a lazy import, or the T0 zero-dependency rule breaks.
Enforced by a test that imports `slmflow` with nothing installed.

---

## 2026-09-10 — Phase 2 split; execution order differs from phase numbering

**Ambiguous.** The brief numbers the full model benchmark as Phase 2, before
architecture (Phase 4) and the core (Phase 5). Executing that literally means
20–50 GB of downloads and days of cold-load measurement producing capability
profiles for models the architecture may never use.

**Chosen.** Split Phase 2 into **2a** (two models, minimal profiling — just
enough to unblock Phase 5) and **2b** (the full sweep, after the vertical slice
is green). Phase 3 depends on 2b and moves with it. Execution order recorded in
`BUILD_BRIEF.md` §6 with a checkpoint gate. Added §2.1, a staged model
acquisition ramp with a per-stage measurement gate and a ~40 GB disk cap.

**Why.** Ground Rule 2: working software over exhaustive planning. Two models is
enough to prove routing, swapping and ensembling are real. Choosing the *best*
models only matters once the machinery works.

**Given up.** Phase 4's architecture is designed against two models' data
instead of ten, so the `ModelCard` capability-profile schema may need one
revision after 2b. A schema revision is cheap; three days benchmarking the wrong
candidate set is not.

---

## 2026-09-10 — A YAML subset parser instead of a PyYAML dependency

**Ambiguous.** The brief requires config-driven YAML pipelines (Phase 5 DoD) and
a core with **zero mandatory native dependencies** (§2). PyYAML is pure Python in
its fallback path but is still a third-party dependency in the import path of
`slmflow.config`.

**Chosen.** Use PyYAML when installed (`slmflow[yaml]`, also pulled by
`[bench]` and `[dev]`); otherwise fall back to `config/minyaml.py`, a
deliberately small parser covering the subset a pipeline config needs: nested
block mappings, block sequences, inline JSON-style flow collections, comments,
quoted and bare scalars, `|` and `>` blocks, and typed scalars.

The parser **refuses** anchors, aliases, merge keys and tab indentation, with a
line number and a pointer to install PyYAML.

**Why refusing rather than approximating.** A config parser that quietly
misreads its input produces a *wrong experiment*, and the wrongness surfaces
weeks later as an inexplicable result. A crash with a line number is strictly
better.

**Given up.** Full YAML compliance in the base install, and some maintenance
burden: the parser is ~250 lines that must stay correct. Mitigated by a test
class covering each supported construct and each refusal.

---

## 2026-09-10 — Learn-then-Test with an exact binomial bound for C3

**Ambiguous.** The brief suggests "conformal prediction / Learn-then-Test — use
it if it fits" for the C3 precision guarantee, without specifying the mechanism.

**Chosen.** Fixed-sequence Learn-then-Test: enumerate candidate thresholds,
compute a one-sided **Clopper-Pearson** (exact binomial) lower bound on
precision over the answered subset at each, walk permissive → strict, return the
first threshold that clears the target. Implemented with a continued-fraction
regularised incomplete beta in stdlib `math`.

**Why exact, not normal-approximate.** Abstention thresholds are chosen on small
answered subsets, where the normal approximation errs in the *optimistic*
direction. An optimistic precision bound is exactly the failure this guarantee
exists to prevent. Fixed-sequence testing also means no multiplicity correction
is needed, so family-wise error stays at `delta` without a Bonferroni penalty
eating the coverage.

**Why the most permissive passing threshold.** It maximises coverage subject to
the guarantee. Threshold 1.0 satisfies any target by answering nothing.

**Given up.** No scipy, so the incomplete beta is hand-rolled (200-iteration
Lentz continued fraction, tested against known values). Split-conformal
prediction sets were considered and rejected: the contract users asked for is
precision on the answered subset, not a set-valued prediction.

---

## 2026-09-10 — Beta-Binomial stopping rule for C4

**Ambiguous.** C4 says "stop as soon as the posterior on the modal answer
crosses a threshold" without specifying the posterior.

**Chosen.** With `m` of `n` samples agreeing, model the majority probability as
`p ~ Beta(m+1, n-m+1)` under a uniform prior and stop when `P(p > 0.5)` exceeds
`posterior_threshold` (default 0.95), subject to `min_samples` (default 2).

**Why.** Well-defined, cheap (one incomplete beta per sample), and it degrades
gracefully at small `n` where a normal approximation would stop far too early.
The `min_samples` floor exists because two identical samples are weak evidence —
`posterior_majority(2, 2)` is below 0.95, which the test suite asserts, so the
rule cannot collapse to "always stop at 1".

**Given up.** A uniform prior is not the true prior over answer distributions.
An empirically fitted prior would stop sooner, and is future work.

---

## 2026-09-10 — Task families chosen for objective metrics

**Ambiguous.** Phase 12 needs a fixed task set; the corpus supports several
framings.

**Chosen.** Three families over the existing DRE-Corpus, each with an objective
metric (Ground Rule 10):

- `repurposing_label` — 3-class exact match on the derived
  `approved`/`investigational`/`hypothesis` label. 140 pairs, plus a 64-item
  held-out split covering 12 drugs absent from training.
- `trial_lookup` — structured extraction, so schema-validity is measurable
  independently of correctness (needed for Phase 10's grammar comparison).
- `grounded_qa` — RAG with groundedness scored against gold PMIDs; retrieval
  quality reported separately as recall@k so failures are attributable.

**Important caveat, recorded so it cannot be forgotten.** The label
distribution is skewed 106/24/10, so the **majority-class baseline is ~0.757**.
`TaskFamily.majority_baseline()` computes it and the results renderer prints it
in the task table. Any accuracy below it is worse than a constant predictor, and
omitting the baseline would make 0.70 look like competence.

**Given up.** No LLM-as-judge family. Ground Rule 10 forbids a small model
judging comparable outputs, and no large judge is available locally.

---

## 2026-09-10 — Abstention is a value, not an exception

**Ambiguous.** Should a pipeline that declines to answer raise?

**Chosen.** `Result.abstained` is a field; only `Result.unwrap()` raises
`AbstainedError`. The CLI exits 3 on abstention.

**Why.** A framework that raised on abstention would push every caller into
`try/except` and make the *safe* path the inconvenient one. Since C3's whole
argument is that abstention is a feature, its ergonomics must be at least as
good as answering.

**Given up.** Callers who want hard failure must remember `.unwrap()`.

---

## 2026-09-10 — `None` propagation instead of sentinel zeros

**Ambiguous.** How to represent an unmeasured cost in code, given Ground Rule 1.

**Chosen.** `Optional[float]` throughout, propagated: `ModelCard.cold_load_s` →
`CostModel.load_seconds` → `Schedule.predicted_seconds` →
`SchedulerRegret.seconds_regret` → `"not measured"` in the rendered table. No
code path converts an unmeasured quantity to `0.0`.

**Why.** It makes Ground Rule 1 a property of the type system rather than a
habit of the author. `mypy --strict` then flags any place that forgets to handle
the `None`.

**Given up.** Verbosity — every consumer handles `None`. Worth it.

---

## Template for future entries

```
## YYYY-MM-DD — <one-line summary>

**Ambiguous.** What was underspecified, and where.
**Chosen.** What was decided, concretely.
**Why.** The reasoning, including what evidence would change it.
**Given up.** The cost, stated plainly.
```
