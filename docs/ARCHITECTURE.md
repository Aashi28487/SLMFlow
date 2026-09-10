# Architecture

Phase 4 deliverable. Written against two models' worth of design assumptions
per the phase-order amendment in `BUILD_BRIEF.md` §6; the `ModelCard` schema may
need one revision once the Phase 2b sweep lands, and that is an accepted cost.

---

## 1. The problem, stated formally

### The Memory-Bounded Orchestration Problem (MBOP)

Given:

- a task decomposed into a DAG of sub-tasks `T = {t_1 … t_n}` with dependencies
  `D ⊆ T × T`;
- a model pool `M = {m_1 … m_k}` with **measured** per-model profiles: resident
  bytes per memory tier, cold-load seconds, tokens per second, and per-task-type
  accuracy;
- a hard memory ceiling **`B`, which is a vector over memory tiers**, not a
  scalar, and optionally a latency deadline `D_max`;

produce an execution schedule — an assignment `σ: T → M`, an ordering
consistent with `D`, and a residency plan (when each model is loaded and
evicted) — that maximises expected end-to-end quality subject to

```
peak_resident_bytes[tier] ≤ B[tier]   for every tier
```

As an objective:

```
maximise   Σ_i  q(t_i, σ(t_i))  −  λ · cost(schedule)
subject to peak_resident_bytes[tier] ≤ B[tier]  ∀ tier

cost(schedule) = Σ_steps ( load_s(m)·[m not resident] + generate_s(t_i, m) )
```

`q` comes from measured `ModelCard.accuracy_by_task`; `load_s` and
`generate_s` from measured `cold_load_s` / `warm_load_s` / `tokens_per_s`. When
any of those is unmeasured the cost model returns `None` and the planners report
the objective as unavailable rather than substituting zero (Ground Rule 1).

### Why this is a new problem, and what it borrows

The structure is a **paging problem fused with an assignment problem**:

| Classical concept | MBOP counterpart |
|---|---|
| Page | A resident model |
| Variable page size | Models differ by 10× in bytes |
| Variable miss cost | `cold_load_s` differs by 10× across models |
| Page replacement policy | `EvictionPolicy` (LRU is the baseline) |
| Two-level cache hierarchy | **VRAM over RAM over SSD** |
| Assignment problem | Which model runs which sub-task |

Borrowing the right formalism is a strength, not a weakness, and saying so
explicitly is how the contribution stays honest. What has no existing solver is
the **fusion under a hard ceiling with dependency constraints**: LangChain,
LangGraph and AutoGen treat the model as an always-available remote endpoint,
so residency has no representation in their execution model at all. RouteLLM and
FrugalGPT optimise *price per API call*, a setting where switching models is
free. On this hardware switching is the single most expensive operation.

### The two-tier budget, and why it sharpens the claim

The evaluation machine has ~5 GB usable VRAM above ~8–10 GB usable system RAM
above an SSD. A model is GPU-resident, RAM-resident, or on-disk-only, and
promotion cost differs at each level. So `B` is a vector, the invariant is
checked per tier, and every trace records `peak_vram_bytes` and
`peak_ram_bytes` separately.

This is a *harder and more interesting* instance than a CPU-only machine would
have given, because a single scalar "peak memory" number cannot express it. See
the hardware correction in `BUILD_BRIEF.md` §2.

---

## 2. Module map

```
slmflow/
  types.py          value objects, enums, plugin Protocols     [zero deps]
  errors.py         SLMFlowError hierarchy                     [zero deps]
  device.py         DeviceCard: probe this machine             [stdlib only]
  resource/         ★C1
    budget.py       MemoryBudget: the per-tier vector
    residency.py    ResourceManager: residency + swap ledger
    rss.py          RSS/VRAM measurement, in- and out-of-process
    scheduler.py    GreedyScheduler, OracleScheduler, regret
  models/
    registry.py     ModelRegistry: measured cards from disk
    backends/       mock.py, llama_cpp.py, ollama.py  [lazy imports]
  routing/          Phase 7
    features.py     cheap query features (microseconds, stdlib)
    heuristic.py    HeuristicRouter + Random/Static/Oracle baselines
    learned.py      pure-Python logistic regression
  strategies/       Phases 6 + 11
    base.py         BaseStrategy (ABC): shared prompt/sample/trace machinery
    slm_mux.py      SLM-MUX + Agent-Forest + model-set selection
    adaptive.py     ★C4 early stopping, cascade, single baseline
  reliability/      ★C3
    confidence.py   4 independent signals + combiner
    calibration.py  Learn-then-Test threshold tuning, Clopper-Pearson
    abstain.py      AbstentionPolicy, TinyVerifier, FallbackChain
    metrics.py      ECE, Brier, paired bootstrap, McNemar
  agents/schema.py  JSON-schema subset validator + GBNF compiler
  orchestration/    Pipeline: the facade users touch
  telemetry/        JSONL traces
  config/           minyaml.py (YAML subset) + loader.py
  rag/store.py      BM25 (stdlib) + dense (optional extra)
  finetune/         Phase 8 adapter loading, fully decoupled
  cli.py            argparse CLI: probe/doctor/plan/run/inspect/models
```

Dependency direction is strictly downward: `types` and `errors` import nothing
from the package; `resource` imports `types`; `strategies` import `resource`;
`orchestration` imports everything. Nothing imports `orchestration`. `finetune`
is imported by nothing (Phase 8 requires the core stay independent of it).

---

## 3. Design decisions and their reasons

### `Protocol` versus `abc.ABC`

**`Protocol` for the plugin seams** — `Backend`, `Router`, `Strategy`,
`ReliabilityCheck`. Structural typing means a plugin never imports `slmflow`, so
a third-party backend can satisfy the contract with zero dependency on this
library. That is the right shape for a seam intended to be extended by people
who are not us.

**`abc.ABC` where shared implementation genuinely exists** — `BaseStrategy` and
`EvictionPolicy`. `BaseStrategy` owns prompt assembly, residency acquisition,
schema validation, confidence scoring and trace construction; subclasses
override exactly one method, `execute`, which decides how many samples to draw
from which models. Making that a Protocol would force every strategy to
reimplement the shared 100 lines, and one of them would forget to record swap
cost.

### Frozen, slotted dataclasses for every value object

`ModelCard`, `DeviceCard`, `TaskSpec`, `RunTrace`, `StepTrace`, `Generation`,
`Schedule` are all `@dataclass(frozen=True, slots=True)`. A trace is a record of
the past; making it mutable invites code that "fixes up" a measurement after the
fact, which is exactly how fabricated numbers enter a results table. Profiling
therefore produces a *new* card rather than mutating one.

### `None` means "not yet measured", and it propagates

Every cost field on `ModelCard` defaults to `None`. `expected_seconds()` returns
`None` when throughput was never measured. `Schedule.predicted_seconds` is
`None` if any model involved is unprofiled. `SchedulerRegret.seconds_regret` is
`None` if either side is `None`. The results renderer prints `not measured`.

This chain is deliberate: it makes Ground Rule 1 a property of the type system
rather than a habit of the author. There is no code path that turns an
unmeasured quantity into `0.0`.

### Residency is a context manager

```python
with rm.resident(card) as handle:
    handle.backend.generate(card, prompt, ...)
```

Something that must be released deterministically belongs in `__enter__` /
`__exit__`. The manager pins the model against eviction for the block's
duration, charges the load to the `SwapLedger`, and re-measures the footprint.
`MockBackend.generate` **raises** if called without a load, so bypassing the
manager fails loudly — otherwise swap accounting would be best-effort and every
cost number suspect.

### Concurrency is bounded by memory, not CPU count

`Pipeline.concurrency_limit()` computes how many of the smallest candidate
models fit the budget *together*. On a single-slot budget it returns 1 and the
"parallel" execution pattern degenerates to sequential. That is the honest
behaviour on this hardware; a `ThreadPoolExecutor(os.cpu_count())` would produce
a queue of tasks all waiting for the same 5 GB of VRAM.

### Async core, sync facade

`Pipeline.arun` is async and gathers each topological layer under a semaphore;
`Pipeline.run` wraps it in `asyncio.run`. Users are never forced into async.
Backend calls are synchronous and blocking, so they go through
`asyncio.to_thread`.

### A YAML subset parser in the core

The brief wants YAML-configured pipelines *and* zero mandatory dependencies.
PyYAML is used when installed; otherwise `config/minyaml.py` parses the subset a
pipeline config needs. It **refuses** anchors, aliases, merge keys and tabs with
a line number rather than mishandling them, because a config parser that
quietly misreads its input produces a wrong experiment, which is worse than a
crash. See `DECISIONS.md` 2026-09-10.

### Grammar first, validation second

Phase 10's failure class is malformed structured output. Two mechanisms, in
preference order: `schema_to_gbnf()` compiles a JSON-schema subset to GBNF so
invalid output is *impossible* on llama.cpp; `parse_and_validate()` checks
after the fact for backends without grammar support (Ollama). Both work
independently, because the Phase 10 measurement is the schema-validity rate
*before and after* grammar constraints — which requires the ungrammared path to
be real.

`schema_to_gbnf` raises on a schema outside its subset rather than silently
returning `None`. Silently skipping the grammar would weaken the guarantee
invisibly.

### Exceptions carry structure, and messages say what to change

`ResourceExhausted` names the tier, the requested bytes, the available bytes and
the budget, then tells the caller to free a model, raise the budget, or pick a
smaller quantisation. `ModelNotFound` lists the ids that do exist.
`GuaranteeUnmet` reports the best precision actually achievable on the
calibration split. A research framework whose errors are cryptic wastes the
scarcest resource in the project.

---

## 4. The contributions, and where each lives

| Claim | Mechanism | Code | Measured by |
|---|---|---|---|
| **C1** MBOP | two-tier budget; swap-aware greedy scheduler; offline oracle | `resource/budget.py`, `resource/scheduler.py` | `benchmarks/regret.py` → scheduler regret + the naive-LRU ablation |
| **C2** portable profiles | `DeviceCard` × `ModelCard` = routing table; `DeviceTier` as a first-class object | `device.py`, `models/registry.py` | re-calibrate on T0 and compare (needs a device) |
| **C3** abstention contract | Learn-then-Test threshold tuning to a declared precision target | `reliability/calibration.py` | `benchmarks/calibrate.py` → risk-coverage, ECE per signal, held-out precision |
| **C4** cost-adaptive SC | Beta-Binomial early stopping; cascade escalation | `strategies/adaptive.py` | `benchmarks/harness.py` → samples/item at matched accuracy |

### C1 in detail

Two solvers, as the brief requires:

- `GreedyScheduler` ships by default. Two mechanisms distinguish it from naive
  load-on-demand: it **credits resident models with the load cost they avoid**,
  so a marginally weaker warm model can beat a marginally better cold one; and
  it **groups sub-tasks by assigned model within each topological layer**, so
  each model loads once per layer instead of once per sub-task.
- `OracleScheduler` enumerates every topological order crossed with every
  assignment, scoring each through the same `simulate()` function. It is capped
  by `max_instances` and **raises** above that rather than returning a
  non-optimal plan labelled optimal.

Both are scored by one `simulate()`, so a comparison between them compares
decisions, not accounting.

**Scheduler regret** = the gap. `NaiveLoadOnDemandScheduler` is the ablation
baseline: what a framework without a scheduler does. The brief demands this be
reported *even if the answer is "little, because loads are amortised"*.

### C3 in detail

`Precision(0.90)` is a declarative contract, not a label. `tune_threshold`:

1. enumerates candidate thresholds (every observed confidence value);
2. for each, computes an **exact Clopper-Pearson lower bound** on precision
   over the answered subset;
3. walks thresholds permissive → strict and returns the **first** that clears
   the target.

Fixed-sequence testing, so no multiplicity correction is needed and family-wise
error stays at `delta`. Exact rather than normal-approximate because these
decisions are made on small answered subsets where the normal approximation errs
*optimistically* — precisely the failure the guarantee exists to prevent.

Returning the most permissive passing threshold **maximises coverage subject to
the guarantee**. A threshold of 1.0 satisfies any target by answering nothing.

Four independent confidence signals, kept separately addressable so Phase 10 can
compare their ECE rather than assume one is best:

| Signal | Cost | What it catches |
|---|---|---|
| `SelfConsistency` | `k` samples | the SLM-MUX baseline |
| `LogProbMargin` | **1** sample | cheap; the only free signal on CPU |
| `CrossModelAgreement` | `\|M\|` samples | one model confidently and consistently wrong |
| `RetrievalSupport` | free | ungrounded claims on the RAG path |

---

## 5. Measurement, in- and out-of-process

`BUILD_BRIEF.md` §2 forbids process-level assumptions, so `resource/rss.py`
exposes three measurements:

- `process_rss_bytes()` — this process; correct for llama-cpp-python, where the
  weights are in our own address space, measured as an RSS delta around load.
- `rss_by_name("ollama")` — another process; the Ollama daemon holds the weights,
  so measuring our own RSS would report near-zero and **silently understate peak
  memory**.
- `gpu_used_bytes()` — VRAM from `nvidia-smi`, so no CUDA runtime is imported
  into this process.

`OllamaBackend.measure_footprint` prefers `/api/ps`, which reports `size` and
`size_vram` per model and so gives the two-tier split directly. When that is
unavailable it falls back to whole-daemon RSS and **logs `coarse_measurement`**,
because it cannot then attribute bytes to one model. Every function that cannot
measure returns `0` and logs; none returns a guess.

---

## 6. Known limitations

Stated here rather than discovered by a reviewer:

1. **Nothing has been run.** The code was written without execution. `RUN.md`
   step 2 is where correctness is first tested.
2. **Predicted versus measured cost.** `benchmarks/regret.py` computes regret
   from measured *card* values, so it is a model-based prediction. End-to-end
   wall-clock regret requires running both schedules, which the harness does
   via the `ablate_scheduler` configuration.
3. **Ollama grammar.** No GBNF parameter, so grammar-constrained decoding is
   llama.cpp-only. The Phase 10 before/after comparison needs that backend.
4. **`RetrievalSupport` is token overlap**, not entailment. Crude, but objective
   and judge-free, which Ground Rule 10 prefers to a small model judging.
5. **The oracle is exponential.** Capped at `max_instances`; regret is measurable
   only on small instances, as the brief scopes it.
6. **C2 is unproven** without a second device. The tier model and device card
   exist; the transfer claim needs a phone or SBC.
7. **The learned router is one-vs-rest logistic regression.** Chosen because it
   trains on the sparse per-model outcome data Phase 2 actually produces and
   runs in microseconds on T0. A ranker would likely be better with more data.
