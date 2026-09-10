# Progress log

Append-only. A Phase Report at the end of every phase: what was built, what was
**measured with numbers**, what did not work, what is deferred and why.

> **Status, 2026-09-10.** Phases 0–12 are **implemented**. Nothing is
> **measured**. The code in this repository was written without being executed:
> no test has run, no model has been downloaded, no benchmark has produced a
> number. Every phase below is therefore marked `code complete, unverified`,
> and the first task on the target machine is `RUN.md` step 2 — making the test
> suite green.
>
> Per Ground Rule 1, no phase may be marked complete until its Definition of
> Done has been checked against a real run on real hardware.

---

## Phase 0 — Repo hygiene and scope resolution — `code complete, unverified`

**Built.** `git init`; `.gitignore` written before the first commit (excludes
`*.gguf`, `models/`, `data.zip`, `.venv/`, `__pycache__/`,
`benchmarks/results/raw/`). Executed Ground Rule 4's demotion: `src/fedslm/` →
`future_work/fedslm/`, `docs/0*.md` → `future_work/docs_fedslm/`, each with a
dated README. Created `docs/PROGRESS.md` and `docs/DECISIONS.md`.

**Measured.** Nothing yet. `docs/ENVIRONMENT.md` **does not exist**: Phase 0's
DoD requires it to contain real probed values, and probing requires running
`slmflow probe` on the target machine. Deliberately not faked.

**Deferred.** `ENVIRONMENT.md` → `RUN.md` step 4.

---

## Phase 1 — Landscape survey and novelty check — `partial`

**Built.** `docs/RESEARCH_SURVEY.md` with a per-claim novelty argument.

**Not done.** The brief requires web search for the survey. This environment had
no search access during implementation, so the survey is written from prior
knowledge and is explicitly marked as needing verification against the current
literature before any claim is defended. **This is the one phase whose DoD is
genuinely unmet**, and it must be closed before Phase 6 per the brief.

---

## Phase 2a — Minimal calibration — `code complete, unverified`

**Built.** `benchmarks/profile_models.py`, one model per invocation by design so
§2.1's acquisition ramp is structurally enforced. Measures cold load, warm
reload, decode and prefill throughput, per-tier footprint, and accuracy on a
task slice. `ModelRegistry` persists cards to JSON with unmeasured fields as
`null`.

**Measured.** Nothing. No model has been downloaded.

---

## Phase 4 — Architecture design — `code complete`

**Built.** `docs/ARCHITECTURE.md`: MBOP stated formally with notation, the
paging-plus-assignment framing, the `Protocol` versus `ABC` rationale, the
module map, the `None`-propagation rule, and a **known limitations** section
listing all seven things a reviewer would otherwise find.

---

## Phase 5 — Core: registry, resource manager, config pipelines — `code complete, unverified`

**Built.** `MemoryBudget` (per-tier vector), `ResourceManager` (residency as a
context manager, `SwapLedger`, pluggable `EvictionPolicy`), `resource/rss.py`
(in-process RSS delta *and* out-of-process daemon RSS *and* `nvidia-smi` VRAM),
`ModelRegistry`, `config/minyaml.py` + `config/loader.py`, `Pipeline` with an
async core and sync facade.

**Tests written for the DoD, not yet run.**
`test_intentional_overcommit_forces_a_swap`,
`test_intentional_overcommit_refuses_when_nothing_fits`,
`test_measured_footprint_over_budget_is_reported`, and the two Hypothesis
property tests asserting the budget invariant over random DAGs and random
access sequences.

**Deferred.** The DoD's "≥2 real local models, end-to-end, with a trace" needs
weights → `RUN.md` steps 5–6. The mock-backed equivalent is
`test_yaml_pipeline_over_two_models_produces_answer_and_trace`.

---

## Phase 6 — SLM-MUX as strategy and baseline — `code complete, unverified`

**Built.** `SLMMuxStrategy` (independent sampling at `T>0`, self-consistency
confidence, validation-accuracy tie-break), `AgentForestStrategy` (flat majority
vote), `SingleStrategy` (trivial baseline), and `select_model_set` implementing
`UnionAcc − λ·Contradiction`.

**Addition beyond the paper, flagged as ours.** `select_model_set` takes an
optional co-residency budget and records `peak_bytes` per candidate set, so a
set whose members cannot be co-resident is visibly expensive on this hardware
rather than silently preferred. Sets over budget are logged, not hidden.

**Deferred.** The DoD is reproducing the paper's *qualitative* pattern
(confidence-based selection beating naive majority voting when models disagree).
Needs real models → `RUN.md` step 8. If it does not reproduce, that must be
stated explicitly in `RESULTS.md`.

---

## Phase 7 — Resource-aware routing and the scheduler — `code complete, unverified`

**Built.** `HeuristicRouter` with a `warm_bonus` term (the resource-aware part
no price-based router has), `LearnedRouter` (one-vs-rest logistic regression in
pure Python, ~20 stdlib-computable features, runs in microseconds on T0),
`CascadeStrategy`, `GreedyScheduler`, `OracleScheduler`,
`NaiveLoadOnDemandScheduler`, plus `RandomRouter`/`StaticRouter`/`OracleRouter`
baselines and both regret functions.

**Measured.** Nothing. `benchmarks/regret.py` and `benchmarks/train_router.py`
are written and will produce scheduler regret, router regret and the
swap-awareness ablation in one table each.

---

## Phase 8 — Fine-tuning integration — `scoped down, code complete`

**Built.** `finetune/adapters.py`: adapter discovery with **mandatory** JSON
provenance sidecars (an adapter without recorded provenance cannot appear in a
results table, so loading one silently would produce an unreportable number),
and `attach_adapter` which degrades gracefully on backends without adapter
support. Imported by nothing in the core, per the decoupling requirement.

**Scope change, stated in writing.** The original plan was off-machine training
only. The corrected hardware (~5 GB usable VRAM) makes QLoRA on a 0.5–1.5B model
*plausible* locally — but per Ground Rule 1 that stays an open question until a
run completes. Nothing here claims local training works.

---

## Phase 9 — RAG integration — `code complete, unverified`

**Built.** `rag/store.py` indexing the real `passages.jsonl` (19,378 passages —
no corpus invented). `BM25Retriever` is pure Python and always available;
`DenseRetriever` (sentence-transformers + FAISS) is behind the `[rag]` extra
with an explicit `release()` because the embedding model competes with the
generator for VRAM. `recall_at_k` reports retrieval quality **separately** from
answer quality so failures are attributable.

**Design note.** BM25 is expected to be competitive here, not a weak baseline:
the queries carry exact drug and indication names, which is where lexical
retrieval beats dense. Worth reporting either way.

---

## Phase 10 — Reliability, abstention, fallback — `code complete, unverified`

**Built.** Four independent confidence signals (`SelfConsistency`,
`LogProbMargin`, `CrossModelAgreement`, `RetrievalSupport`) kept separately
addressable so their ECE can be compared; `TinyVerifier` (selective by
construction — skipped above `skip_above`, because verifying a confident answer
costs a load that may evict the generator); `AbstentionPolicy` with four
independent grounds; `FallbackChain`; `schema_to_gbnf` + a stdlib JSON-schema
validator; and `tune_threshold` implementing the C3 contract.

**Measured.** Nothing. `benchmarks/calibrate.py` will produce the risk-coverage
curve, ECE and Brier per signal, the tuned threshold, and — importantly —
**held-out** precision and coverage, since a threshold evaluated on its own
fitting data always looks calibrated.

---

## Phase 11 — Optimisation — `code complete, unverified`

**Built.** Two optimisations, each with its before/after measurable from the
trace:

1. **Swap-aware batching**, expressed as a scheduling decision:
   `GreedyScheduler(group_by_model=True)` groups sub-tasks by assigned model
   within a topological layer, so each model loads once per layer. The
   `group_by_model=False` variant is the before-case.
2. **Cost-adaptive self-consistency** (C4): `AdaptiveSelfConsistency` reports
   `SamplingBudgetReport` with `drawn` versus `max_samples`, so samples saved is
   read off the trace rather than asserted. `CascadeStrategy` is the second
   form.

---

## Phase 12 — Evaluation harness — `code complete, unverified`

**Built.** `benchmarks/harness.py`. One command runs any configuration over the
fixed task set and emits one table carrying quality **and** cost together.
Nine configurations including three ablations declared as `ablation_of` so the
ablation table derives from the definitions rather than drifting.

**Enforced by construction, not by discipline:**

- ≥3 seeds default, with a warning below that;
- mean ± std across seeds;
- paired bootstrap **and** McNemar run unconditionally against the baseline, so
  no difference can reach the table without a test attached;
- environment record with pinned model digests and probe warnings in every
  artifact;
- `not measured` rendered wherever a value is `None`;
- mock runs **stamped** at the top of `RESULTS.md` so they cannot be cited.

**Not built.** `evals/` frozen golden set. The golden set must be generated from
a real run — inventing expected outputs would breach Ground Rule 1 — so it is
deferred to after `RUN.md` step 6.

---

## Phase Report template

```
## Phase N — <name> — <status>

**Built.** What exists now.
**Measured.** Numbers, with units and n. Or "nothing yet" — never an estimate.
**Did not work.** Failures, including negative results (Ground Rule 9).
**Deferred.** What was cut, and why.
```

---

## Immediate next actions on the target machine

1. `pip install -e ".[dev]"`, then `pytest -m "not real_model"`. Expect
   failures; the code has never run. Log what breaks here.
2. `slmflow probe > docs/ENVIRONMENT.md`, then add exact CPU/GPU model numbers.
3. Close Phase 1: verify the survey against current literature with web search.
4. Model 1 → profile → **read the card before downloading model 2**.
5. Model 2 → the vertical-slice checkpoint in `RUN.md` step 7.
6. Only then Phase 2b, the full sweep.
