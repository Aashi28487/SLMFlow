# SLMFlow — End-to-End Build Brief (v2)

**For:** Claude Code, operating autonomously in `C:\FYP\implementation`
**Deliverable:** a research-grade, `pip install`-able Python library
**Thesis:** *Orchestration of small language models is a scheduling problem under a hard memory budget — and nobody has built the scheduler.*

Read this entire document before writing code. Re-read the relevant phase section before starting that
phase. Three source documents are authoritative and must inform your design:

- `context.md` — the project's research-context document (problem, layers, hardware constraint,
  benchmarking axes, open "Claude Suggestions"). **Authoritative for scope.**
- `slm-mux.pdf` — the SLM-MUX paper (primary related-work baseline and one implemented strategy).
- `Topic_1 SLM.pptx.pdf` — the project presentation. Read it for the framing the supervisor expects.

Existing repo assets you must account for (do not rediscover them):
- `docs/01_PRD.md`, `docs/02_TECH_STACK.md`, `docs/03_UI_LAYOUT.md` — written for that other direction.
- `data/raw/repurposing/` (PubMed abstracts, openFDA labels, ClinicalTrials studies),
  `data/processed/repurposing/` (`passages.jsonl`, `qa_pairs.jsonl`, `repurposing_pairs.jsonl`,
  `DATASET_CARD.md`, `stats.json`), `data/partitions/repurposing/{iid,noniid,test.jsonl}`.
  **This data is reusable and you should reuse it** — it is the corpus for the RAG module (Phase 9),
  the pharma demonstration (Phase 19), and one of the task families in the benchmark harness.
  Read `DATASET_CARD.md` and `stats.json` before designing around it.

---

## 0. The research contribution — read this first

`context.md` §9 currently frames the contribution honestly but weakly: *"the individual ingredients
exist; the contribution is combining them under a hardware constraint."* Combination alone is not a
publishable claim. **Your job is to make the contribution a specific, formal, measurable object.**
Everything you build should serve one of the four claims below. If a feature does not serve one of
them, it is engineering scaffolding — build it minimally and move on.

### C1 — The Memory-Bounded Orchestration Problem (MBOP) — *primary contribution*

Formalise and solve this. Given:
- a task decomposed into a DAG of sub-tasks `T = {t_1 … t_n}` with dependencies,
- a model pool `M = {m_1 … m_k}` with **measured** per-model profiles: resident bytes, cold-load
  seconds, tokens/sec, and per-task-type accuracy (from Phase 2/3 calibration),
- a hard memory ceiling `B` (bytes) and, optionally, a latency deadline `D`,

produce an execution schedule — an assignment of sub-tasks to models, an ordering, and a residency
plan (when each model is loaded and evicted) — that maximises expected end-to-end quality subject to
`peak_resident_bytes ≤ B`.

This is where the novelty lives. Existing frameworks (LangChain, LangGraph, AutoGen) treat the model
as an infinitely available remote endpoint; routing work (RouteLLM, FrugalGPT) optimises *price per
API call*, where switching models is free. **On a laptop or a phone, switching models is the single
most expensive operation in the system**, and no existing orchestrator models it. Note the structural
kinship to classic problems — this is a *caching/paging problem* (model residency ≈ page replacement,
with variable page sizes and variable miss costs) fused with an *assignment problem*. Say so
explicitly in the writeup; borrowing the right formalism is a strength, not a weakness.

Concretely you must deliver:
- a written problem statement with notation, in `docs/ARCHITECTURE.md` and `docs/RESULTS.md`;
- at least **two** solvers: a greedy/heuristic scheduler (ships by default) and an **offline optimal
  or near-optimal oracle** (exhaustive search or ILP on small instances) used only for measurement;
- the headline number: **scheduler regret** = quality/cost gap between your online scheduler and the
  offline oracle, measured on real runs;
- an honest ablation: what does swap-aware scheduling buy over naive "load on demand, evict LRU"?
  Report it even if the answer is "little, because model loads are amortised" — that is a finding.

### C2 — Portable capability profiles across hardware tiers

A **device card** (measured properties of *this* machine) × a **model card** (measured properties of
each model) = the routing table. The research question: *does a routing policy calibrated on one
device transfer to a weaker device after re-calibration, or must the policy itself be relearned?*
Measure this across at least two tiers (see §2 Hardware Tiers). This is what makes the phone claim
real rather than aspirational.

### C3 — Abstention as a declarative contract

Nobody exposes this in an orchestration API. The user writes:

```python
pipe = slmflow.Pipeline.from_config("qa.yaml", guarantee=slmflow.Precision(0.90))
```

and the framework **automatically tunes its confidence thresholds** on a calibration split so that
precision on the answered subset meets the target, then reports the achieved coverage. The
contribution is the mechanism (threshold selection with a distribution-free guarantee — read up on
conformal prediction / Learn-then-Test and use it if it fits) plus the measurement (risk-coverage
curve, ECE, coverage at target precision). This directly attacks SLM-MUX's self-admitted weakness
that confidence is self-consistency alone.

### C4 — Cost-adaptive self-consistency

SLM-MUX multiplies cost by `k` (samples per model) × `|M|` (models). On CPU that is brutal. Implement
**sequential/early-stopping self-consistency**: draw samples one at a time, stop as soon as the
posterior on the modal answer crosses a threshold, and escalate to a second model only when the first
is uncertain (cascade). Report the measured cost reduction at matched accuracy. This is the
optimisation most likely to produce a clean, quotable result.

**Novelty check.** Before Phase 6, run a focused literature check (Phase 1) and write in
`docs/RESEARCH_SURVEY.md` an explicit paragraph per claim: *is this genuinely unaddressed?* If you
find prior work that already does C1 or C3, say so plainly, cite it, and sharpen the claim to the
remaining delta. **Do not defend a claim the literature has already answered.**

---

## 1. Ground rules (apply to every phase)

1. **Never fabricate results.** Every number in any report, README, docstring, or table must come from
   code you actually ran on this machine. If it cannot be measured yet, write `not yet measured` — not
   an estimate that reads like a measurement. Never write a plausible-looking placeholder number.
2. **Working software over exhaustive planning.** Get a thin vertical slice end-to-end (one task, two
   real SLMs, one routing strategy, one metric, one trace file) before generalising anything.
3. **Build the evaluation harness alongside the first strategy, not after.** Every later claim must be
   reproducible by re-running one command.
4. **Resolve the scope conflict yourself, in favour of orchestration.** Move `src/fedslm/` to
   `future_work/fedslm/` and the three `docs/0*_*.md` files to `future_work/docs_fedslm/`, each with a
   one-line `README` note explaining the demotion and the date. Do not delete, do not silently keep
   building both. `data/` stays where it is — it is shared.
5. **Hardware is a design input, not an afterthought.** See §2. Design memory accounting from commit 1.
6. **Local inference only.** `llama.cpp` (via `llama-cpp-python`) and/or Ollama's local HTTP API. No
   hosted LLM API anywhere in the framework's runtime path or its tests — not for judging, not for
   fallback, not "just for the demo". You (Claude Code) may use web search to draft related work.
7. **Small, testable increments; `git init` on your first command; commit after each one.** Never leave
   `main` in a state where `pytest` fails. Never commit model weights or `data.zip` — write
   `.gitignore` first.
8. **Log every judgment call** in `docs/DECISIONS.md`: dated entry, what was ambiguous, what you chose,
   why, what you gave up. This is your audit trail and your future related-work section.
9. **Report negative results.** "Orchestration overhead exceeded its benefit on CPU for task X" is a
   valid, expected, *publishable* finding. Write it down. A brief with only wins is a brief that lied.
10. **Prefer objective metrics over LLM-as-judge.** Where a judge is unavoidable, use the largest model
    available, mark every judge-scored number as `[judge: <model>]` in the output table, and never let
    a small model judge outputs of comparable or larger models.
11. **Reproducibility is a feature.** Fixed seeds, pinned model digests (record the Ollama/GGUF SHA),
    pinned dependency versions, recorded machine specs in every results artifact.

---

## 2. Hardware tiers — the framework must be built for the smallest one

`context.md` §6 names one machine. Generalise it into an explicit tier model, and make the tier a
first-class object in the code (`slmflow.device.DeviceTier`).

| Tier | Device | Memory for models | Backend | Realistic model set |
|---|---|---|---|---|
| **T0 — Phone** | Android via Termux, or ARM SBC | ~1.5–3 GB RAM | `llama.cpp` ARM build / MLC | one 0.5–1.5B model at Q4, **one at a time** |
| **T1 — Dev + evaluation machine** *(the real machine)* | Intel i5 HX-series, discrete GPU ~6 GB VRAM, 16 GB RAM, 512 GB SSD | **~5 GB VRAM** (fast tier) over **~8–10 GB RAM** (slow tier) | `llama.cpp` w/ CUDA offload, Ollama, or pure CPU | 1–2 resident at 0.5B–4B Q4_K_M; **one** 7–8B Q4_K_M GPU-resident for the big-model baseline |
| **T2 — Workstation** *(now largely unnecessary)* | any GPU box you can borrow | >12 GB | same | only if a >8B baseline is specifically wanted — see note |

**Hardware correction, 2026-09-10 — read this before designing the resource manager.**
Earlier versions of this brief and of `context.md` §6 described T1 as *"Intel i5, Iris iGPU, no
CUDA, no usable VRAM"*. **That was wrong.** The real machine has a discrete GPU with ~6 GB VRAM and
working CUDA offload. Two consequences that change the plan:

1. **The memory budget `B` is two-tiered, and this makes C1 stronger, not weaker.** ~5 GB of usable
   VRAM sits above ~8–10 GB of usable system RAM, which sits above the SSD. A model is
   GPU-resident, RAM-resident, or on-disk-only, and the promotion cost differs at each level. So
   MBOP is a **two-level paging problem with variable page sizes and variable miss costs**: model
   `B` as a **vector** `(B_vram, B_ram)`, not a scalar, enforce the invariant per tier, and report
   `peak_vram_bytes` and `peak_ram_bytes` separately in every trace. Say in `ARCHITECTURE.md` that
   the GPU/CPU split is the fast/slow tier of one cache hierarchy.
2. **T2 is no longer needed for the headline baseline.** A 7–8B Q4_K_M is roughly 4.4–4.9 GB and
   fits in 6 GB VRAM at modest context, so Phase 16's central question — *does orchestrating small
   models beat one bigger model?* — is answerable **on this machine, measured**. Do not arrange a
   borrowed GPU box unless a >8B baseline is wanted. Verify the fit by loading the model and
   measuring, never by arithmetic on the file size.

**Design rules that follow, and which are non-negotiable:**

- **The core library must import and run on T0.** Therefore: the `slmflow` core has **zero mandatory
  native dependencies**. No numpy, no pandas, no torch, no pydantic-core in the import path of
  `import slmflow`. Pure Python + stdlib. Everything heavy is an optional extra behind a plugin
  boundary and a lazy import.
- **Model residency is single-slot by default.** Write the resource manager assuming `B` may only fit
  one model. Multi-residency is the optimisation, not the assumption.
- **Every cost number is reported cold and warm.** Load time dominates on T0/T1; hiding it is dishonest.
- **Backends are plugins behind one Protocol.** `llama-cpp-python`, Ollama HTTP, and a deterministic
  `MockBackend` for tests. Adding an MLC/ONNX backend later must not touch the core.
- **No process-level assumptions.** The framework must work when the model runs in-process
  (llama-cpp-python) *and* out-of-process (Ollama daemon), since a phone has neither a daemon nor
  spare RAM for one. Memory accounting must handle both — measure RSS of the right process.
- **Phone support is a claim you must earn.** Per `context.md` §E: if a full T0 run proves infeasible in
  the time available, scope down *explicitly and in writing* to "core library imports and runs one
  0.5B model end-to-end on T0, measured" and report exactly that. Do not quietly drop it and do not
  overclaim mobile support in the README.
- **CUDA is a backend detail and never a core dependency.** GPU offload lives inside the
  `llama_cpp` backend plugin behind a lazy import. `torch`, CUDA wheels and any GPU probe must stay
  out of the `import slmflow` path — the zero-native-dependency rule above is unchanged by the
  presence of a GPU. A `DeviceCard` on a machine with no GPU must still be constructible.
- **Account for VRAM and system RAM separately, from commit 1.** One scalar "peak memory" number
  cannot express this machine. `ResourceManager` tracks both, the budget invariant is checked
  against both, and `ResourceExhausted` names *which* tier was exceeded and by how much.
- **Disk is a third budget.** 512 GB SSD, shared with `data/` and `data.zip`. Cap total GGUF weights
  at ~40 GB, keep a manifest of `(model, quant, digest, bytes, measured profile)`, and **delete
  weights once a model is profiled and rejected** — the recorded digest makes the run
  reproducible without keeping the file. Never commit weights; `.gitignore` covers `*.gguf`.
- **Fine-tuning (Phase 8) is now borderline-feasible locally.** ~5 GB usable VRAM makes QLoRA on a
  0.5–1.5B model plausible. Attempt it locally at ≤1.5B and **measure** whether it completes;
  anything larger goes off-machine per the original plan. Do not claim local training works until a
  run has finished on this machine.

### 2.1 Model acquisition ramp — do not download the whole candidate set up front

A 6–10 model sweep at Q4_K_M is 20–50 GB of downloads and many hours of cold-load and generation
before a single line of the framework is proven. That hammers the machine and the SSD to produce
profiles for models the architecture may never use. **Acquire models in stages, and finish each
stage's measurement before starting the next download.**

| Stage | When | What to pull | Gate before moving on |
|---|---|---|---|
| **A** | Phase 2a | **One** 0.5–1.5B Q4_K_M instruct model | It loads, generates, and its resident bytes / cold-load seconds / tokens-per-sec are **measured and committed** |
| **B** | Phase 2a | **One** deliberately different second model (other family, or ~3B) | Two-model routing and a 2-model ensemble both run |
| **C** | Phase 2b | The remaining candidate sweep, 4–8 models | Only after the vertical slice runs end-to-end with a real trace |
| **D** | Phase 16 | **One** 7–8B Q4_K_M for the big-model baseline | GPU-resident load verified by measurement |

Rules: record the Ollama/GGUF **digest** on download, never re-download a model already in the
manifest, and prefer one extra quantisation of a model you already trust over a new model you do not.

---

## 3. Python engineering standards (fundamentals — these are graded)

This is a library others install, not a script. Build it like one.

**Packaging & layout**
- `src/` layout, `pyproject.toml` (hatchling), PEP 621 metadata, semantic version in one place.
- Optional extras: `slmflow[llama]`, `slmflow[ollama]`, `slmflow[rag]`, `slmflow[bench]`, `slmflow[dev]`.
  Base install must be tiny and pure-Python.
- Console entry point: `slmflow` (Typer or stdlib `argparse` — prefer `argparse` to keep the base
  dependency-free; put Typer in `[cli]` if you want the nicer UX).
- `py.typed` marker. Ship types.

**Language fundamentals — demonstrate them deliberately**
- `typing.Protocol` for the plugin seams (`Backend`, `Router`, `Strategy`, `ReliabilityCheck`), `abc.ABC`
  only where shared implementation genuinely exists. Know why you chose which, and write it in
  `ARCHITECTURE.md`.
- `@dataclass(frozen=True, slots=True)` for value objects (`ModelCard`, `DeviceCard`, `TaskSpec`,
  `RunTrace`). Immutability by default.
- `Enum`/`StrEnum` for closed sets (task type, execution pattern, abstention reason).
- Context managers (`__enter__`/`__exit__`) for model residency — `with rm.resident(model) as h:` is the
  correct API shape for something that must be released deterministically. Use `contextlib`.
- `asyncio` for parallel/ensemble execution, with a sync facade so users are not forced into async.
  Bound concurrency with a semaphore derived from the memory budget — not from CPU count.
- Generators/iterators for streaming tokens and for lazily walking the task DAG.
- Custom exception hierarchy rooted at `SLMFlowError`, with `ResourceExhausted`, `SchemaViolation`,
  `AbstainedError`, `BackendUnavailable`. Error messages must tell the user what to change.
- `functools.lru_cache` / explicit caches for capability lookups; `weakref` where a cache must not pin
  a loaded model in memory.
- `logging` (stdlib) with a JSONL handler for traces. No `print` in library code, ever.
- Full type annotations; `mypy --strict` (or `pyright`) clean on `src/`; `ruff` clean.
- Docstrings in one consistent style (Google or NumPy) on every public symbol, with a runnable example
  on the top-level API.

**Testing**
- `pytest`, with a `MockBackend` that returns scripted, deterministic outputs so **the entire test suite
  runs in seconds with no model downloaded**. Real-model tests live behind `@pytest.mark.slow` and a
  `--run-slow` flag.
- Property-based tests (`hypothesis`) for the scheduler: *no schedule ever exceeds the memory budget* is
  an invariant worth fuzzing.
- A test that intentionally overcommits memory and asserts the resource manager refuses or schedules a
  swap — this test is a deliverable of Phase 5.
- Coverage on `src/slmflow/{resource,routing,reliability}` ≥ 85%. Those are the contribution.

---

## 4. Repository structure

```
implementation/
├── src/slmflow/
│   ├── __init__.py          # public API only: Pipeline, Router, Registry, run(), __version__
│   ├── types.py             # dataclasses, enums, Protocols, exceptions  (no deps)
│   ├── device.py            # DeviceCard: measured RAM/cores/tier; probing
│   ├── models/              # registry, ModelCard capability profiles, backends/
│   │   └── backends/        # llama_cpp.py, ollama.py, mock.py  (lazy-imported plugins)
│   ├── resource/            # memory budget, residency manager, swap accounting, scheduler  ★C1
│   ├── routing/             # task analysis, heuristic router, learned router, cascade policy
│   ├── agents/              # small typed agents; schema/grammar-constrained I/O
│   ├── orchestration/       # execution graph: sequential / parallel / conditional / ensemble
│   ├── reliability/         # confidence, verification, abstention contract, fallback  ★C3
│   ├── strategies/          # pluggable: slm_mux.py, agent_forest.py, cascade.py, single.py
│   ├── telemetry/           # JSONL traces, quality+cost records, run inspector
│   ├── config/              # YAML schema + loader + validation with good error messages
│   ├── rag/                 # optional: local embeddings + vector store  (extra: rag)
│   └── finetune/            # optional: LoRA adapter loading  (training runs off-machine)
├── benchmarks/              # harness.py, tasks/, baselines/, ablations/, analysis/, results/
├── evals/                   # frozen golden set + expected outputs (regression guard)
├── examples/                # quickstart, pharma_demo/, extraction_demo/, qa_demo/
├── docs/
│   ├── PROGRESS.md          # append-only phase log
│   ├── DECISIONS.md         # append-only ambiguity-resolution log
│   ├── ARCHITECTURE.md
│   ├── RESEARCH_SURVEY.md   # Phase 1
│   ├── MODEL_SELECTION.md   # Phase 3
│   ├── RESULTS.md           # the paper's numbers
│   └── COMPARISON.md        # vs LangChain / LangGraph / SLM-MUX
├── tests/
├── future_work/             # fedslm/ + docs_fedslm/  (demoted per Ground Rule 4)
├── data/                    # existing corpus — reused, not regenerated
├── pyproject.toml
├── .gitignore               # models, *.gguf, data.zip, .venv, __pycache__, benchmarks/results/raw
└── README.md
```

---

## 5. The operating loop

Work in phases (§6). Within each phase run this loop until that phase's Definition of Done is met, then
move on — **do not wait for instructions between phases** unless you hit a genuine blocker (§7).

```
LOOP for current phase:
  1. PLAN      — write/update this phase's task list in docs/PROGRESS.md
  2. IMPLEMENT — the smallest coherent unit of work
  3. TEST      — unit tests always; for anything touching models or benchmarks, run it for
                 real on this machine (small n is fine while iterating)
  4. VERIFY    — check the output against this phase's Definition of Done, literally
  5. LOG       — docs/PROGRESS.md (what changed, what passed, what failed)
                 docs/DECISIONS.md (if you made a judgment call)
  6. COMMIT    — descriptive message
  IF Definition of Done fully met -> exit loop, next phase
  ELSE -> identify the single highest-leverage gap, go to 2
```

At the end of every phase append a **Phase Report** to `docs/PROGRESS.md`: what was built, what was
measured (with numbers), what did not work, what is deferred and why.

**Stall rule.** More than ~5 loop iterations on one phase without closing the gap: stop. Write in
`docs/DECISIONS.md` exactly what is blocking you and what would unblock it (compute, a design choice, a
missing model). Then **scope the phase down explicitly, state in writing what you cut**, and move on.
Stalling is worse than a documented reduction in scope.

**Budget guidance.** Phases 1–7 and 10, 12, 16 carry the research contribution — spend time there.
Phases 8, 13, 17, 18 are engineering hygiene — do them correctly but briskly. If time runs short, cut
depth from Phase 8 (fine-tuning) first, then Phase 15 (extra use cases). Never cut Phase 12 or 16.

---

## 6. Phases and Definitions of Done

### Phase order amendment, 2026-09-10 — thin slice before the sweep

**The numbering below is the dependency order, not the execution order.** Running Phase 2's full
model sweep before anything works end-to-end would occupy the machine for days and produce
capability profiles for models the architecture may not even use. Per Ground Rule 2 (working
software over exhaustive planning), Phase 2 splits in two:

- **Phase 2a — minimal calibration, two models.** Profile one model, then a second: resident bytes
  (VRAM *and* RAM), cold-load seconds, warm tokens/sec, and accuracy on **one** small task slice.
  Just enough signal to unblock Phase 5. Nothing more.
- **Phase 2b — full candidate sweep.** Phase 2 exactly as written below, over the remaining
  candidates. Runs **after** the vertical slice is green.
- **Phase 3** consumes 2b's complementarity data, so it moves with 2b.

**Execution order:**

```
Phase 0   repo hygiene, scope demotion, ENVIRONMENT.md probed
   |
Phase 1   landscape survey + novelty check
   |
Phase 4   architecture design  (two-tier memory budget baked in)
   |
Phase 2a  model #1: download, profile, commit numbers
   |
Phase 5*  registry + resource manager + MockBackend + config loader
   |          (whole test suite green with zero model weights present)
Phase 2a  model #2: download, profile, commit numbers
   |
Phase 5   complete: YAML pipeline over 2 real models, end-to-end, real trace
   |
Phase 7*  heuristic routing only — no learned router, no oracle yet
   |
=== CHECKPOINT: vertical slice runs from one command and emits a real trace ===
   |
Phase 2b  full model sweep        -> Phase 3  complementarity + model selection
   |
Phase 6   SLM-MUX + ensemble + single baselines
   |
Phase 7   complete: learned router, swap-aware scheduler, offline oracle, regret
   |
Phases 8-20 as written
```

`*` = partial delivery of that phase; its Definition of Done is met on the second visit, not the
first. Do **not** mark a phase complete in `PROGRESS.md` on a partial pass — log it as
`Phase 5 (partial)` and state what remains.

**What this trades away, stated honestly:** Phase 4's architecture gets designed against two models'
data instead of ten, so the `ModelCard` capability-profile schema may need one revision once 2b
lands. That is an acceptable and expected cost — a schema revision is cheap, three days of
benchmarking the wrong models is not. **Log this reordering in `docs/DECISIONS.md`** per Ground
Rule 8, including this trade-off.

### Phase 0 — Repo hygiene and scope resolution
**Do:** `git init`. Write `.gitignore` **before** the first commit (exclude `*.gguf`, `models/`,
`data.zip`, `.venv/`, `__pycache__/`, `benchmarks/results/raw/`). Execute Ground Rule 4's demotion of
`src/fedslm/` and `docs/0*.md` to `future_work/`. Create `docs/PROGRESS.md` and `docs/DECISIONS.md`
with their first entries. Capture the machine's specs (CPU model, cores, total RAM, OS, Python version)
into `docs/ENVIRONMENT.md` by running code, not by assuming.
**DoD:** clean repo, first commit, scope conflict resolved in writing, `ENVIRONMENT.md` contains real
probed values.

### Phase 1 — Landscape survey and novelty check
**Do:** Web-search and summarise: LangChain, LangGraph, AutoGen, CrewAI, DSPy, RouteLLM, FrugalGPT,
Belcak et al. 2025 (*Small Language Models are the Future of Agentic AI*), MAST (Cemri et al. 2025),
llama.cpp, Ollama, MLC-LLM, and any SLM-specific orchestration project you find. Read `slm-mux.pdf`
closely and extract **in your own words** (no verbatim reproduction): the method (independent
generation, self-consistency confidence, validation-accuracy tie-break), the selection objective
`UnionAcc − λ·Contradiction`, the theoretical bound, and every limitation the authors state themselves.
Check whether SLM-MUX released code — reproducing beats reimplementing.
**Then do the novelty check from §0** on each of C1–C4.
**Write:** `docs/RESEARCH_SURVEY.md` — comparison table extending `context.md` §3.3, gap analysis, an
adopt-vs-contribute table, and the per-claim novelty paragraphs.
**DoD:** survey exists, cites real retrievable sources with URLs/DOIs, names **≥5 concrete gaps** each
traceable to a later phase, and states plainly for each of C1–C4 whether it survives the novelty check.

### Phase 2 — Benchmark candidate SLMs on this hardware
**Do:** Select 5–8 real, freely-downloadable quantised models in 0.5B–4B spanning different strengths
(general instruct, math/reasoning, code, and at least one ≤1B model for cascade and T0 experiments).
Pull via Ollama or GGUF. Measure per model: **cold load time, peak RSS while resident, tokens/sec
(prefill and decode separately), and accuracy on ~50–100 items** from each of 2–3 public benchmark
slices (GSM8K / MATH / GPQA-style, or documented substitutes). Record the model digest/SHA.
**Write:** `benchmarks/single_model_results.md` plus raw CSV/JSON that later phases re-load rather than
re-measure.
**DoD:** `benchmarks/run_single_model_bench.py` regenerates the table from scratch; the table contains
real numbers; raw data is committed; every model's exact version is pinned.

### Phase 3 — Analyse complementarity; select models per use case
**Do:** Analyse Phase 2 data for **complementarity** (models strong on disjoint slices — mirror SLM-MUX
Figure 4) versus **strict dominance** (one model better everywhere → pairing adds nothing, and saying so
is a result). Select 2–4 model combinations and 1–2 use-case profiles ("fast triage cascade",
"high-reliability pharma QA", "T0 phone single-model").
**DoD:** `docs/MODEL_SELECTION.md` names the chosen models and combinations with data-backed reasoning
and a complementarity matrix/plot. Later phases reference this file; they do not re-derive choices.

### Phase 4 — Architecture design
**Do:** Turn `context.md` §4–§5 into concrete interfaces: `Backend`, `Model`/`ModelCard`, `DeviceCard`,
`Agent`, `Router`, `Strategy`, `ReliabilityCheck`, `ResourceManager`, `Scheduler`, `TelemetryRecorder`.
Write the MBOP formalism (§0/C1) here. Decide and document the YAML config format for models, routing
policy, pipeline shape, and guarantees. Justify Protocol-vs-ABC per seam.
**DoD:** `docs/ARCHITECTURE.md` with a module diagram and full interface signatures; `pip install -e .`
works; `import slmflow` succeeds in **under 200 ms** with zero native deps; stub classes match the
documented design; near-empty tests pass; `mypy --strict` clean.

### Phase 5 — Core: registry, resource manager, config-driven pipelines  ★C1
**Do:** Implement the model registry (capability profiles loaded from Phase 2 data), the **resource
manager** (memory budget, residency as a context manager, load/unload, swap-cost accounting, RSS
measurement that works for both in-process and daemon backends), and the config-driven pipeline builder
expressing sequential / parallel / conditional / ensemble execution over a shared typed context.
**DoD:** a YAML-configured pipeline over ≥2 real local models runs end-to-end on a real task, producing
a final answer **plus a structured trace**; the resource manager provably refuses or schedules a swap
for a pipeline that would exceed the budget, verified by an intentional-overcommit test; a hypothesis
property test asserts the budget invariant holds for randomly generated task DAGs.

### Phase 6 — SLM-MUX as strategy and baseline
**Do:** Implement SLM-MUX faithfully in `strategies/slm_mux.py`: independent generation at T>0,
self-consistency confidence, validation-accuracy tie-break, and the `UnionAcc − λ·Contradiction`
selection search. Also implement one contrasting ensemble (Agent-Forest-style flat majority vote) and a
trivial `single.py` baseline.
**DoD:** on the Phase 2 slices your implementation reproduces the paper's **qualitative** pattern
(confidence-based selection beats naive majority voting when models disagree). Exact numbers will differ
— different models, different hardware. Log both your numbers and the paper's side by side in
`docs/RESULTS.md`, and state explicitly if the pattern does **not** reproduce.

### Phase 7 — Resource-aware routing and the scheduler  ★C1
**Do:** Implement per-query routing: pick model(s) from capability profile **plus live memory/latency
state**, not a fixed set. Start heuristic (rules over the Phase 3 complementarity matrix), then a
lightweight learned router (logistic regression / gradient boosting over cheap query features) trained on
your own Phase 2 data — no neural router, it must run on T0. Implement **cascade/escalation** (cheapest
model first, escalate on low confidence) as a routing policy. Implement the **swap-aware scheduler** and
the **offline oracle scheduler** for regret measurement.
**DoD:** the router is measurably compared against random and static baselines on **cost-normalised
accuracy** (accuracy/second and accuracy/GB-peak) on ≥1 task family, with honest reporting either way;
**router regret** (vs. an oracle that picks the best model in hindsight) is computed and reported; and
**scheduler regret** (vs. the offline optimal schedule) is computed on small instances.

### Phase 8 — Fine-tuning integration *(optional module, decoupled)*
**Do:** A LoRA fine-tuning path for a narrow task (e.g. the extraction format used by the pharma demo),
**trained off-machine** (Colab-class, per `context.md` §6) with the framework loading the resulting
adapter locally at inference. The orchestration core must not import or require anything from here.
**DoD:** one documented, script-reproducible example where a fine-tuned adapter improves a measured
metric over the base model on a narrow task. If off-machine training proves impractical, scope down to
"adapter loading implemented and tested against a published adapter" and say so.

### Phase 9 — RAG integration
**Do:** Local embedding model + local vector store (FAISS or Chroma, fully offline), usable by any
agent. **Index `data/processed/repurposing/passages.jsonl`** — do not invent a corpus. Pipeline:
retrieval → extraction → summarisation → verification against retrieved evidence, with citations.
**DoD:** a RAG-backed agent answers questions over the local corpus with citations, and a
**groundedness check** (do the answer's claims trace to retrieved spans?) is *implemented and measured*
on `qa_pairs.jsonl`, not asserted. Report the retrieval quality (recall@k) separately from answer
quality so failures are attributable.

### Phase 10 — Reliability: verification, abstention contract, fallback  ★C3
**Do:** Confidence signals beyond bare self-consistency (log-prob margin, verifier agreement, retrieval
support, cross-model contradiction rate) — flag this explicitly as addressing SLM-MUX's stated weakness.
A **tiny verifier** pattern (0.5B checking a 3B generator, per `context.md` §D). **Selective
verification** (verify only low-confidence claims). Abstention when confidence is weak. Fallback
(alternate model / re-route) on failure or resource error. **Grammar-constrained decoding (GBNF) or JSON
schema enforcement at every agent I/O boundary** to eliminate the malformed-output failure class. Then
build the **declarative guarantee** from C3: threshold auto-tuning on a calibration split to hit a
user-declared precision target.
**DoD:** risk-coverage curve plotted from real data; **ECE measured** for ≥1 confidence signal (and
ideally compared across signals); abstention demonstrably improves precision on the answered subset
versus never abstaining; schema-validity rate measured before and after grammar constraints;
`Pipeline(guarantee=Precision(p))` achieves the target precision on held-out data, with achieved
coverage reported.

### Phase 11 — Optimisation  ★C4
**Do:** Model-swap-aware batching (group sub-tasks by model so each loads once). Cost-adaptive
self-consistency with early stopping (C4). Plus any optimisation Phases 2–10 revealed (prompt caching,
KV-cache reuse across sub-tasks, prefill batching, quantisation-level selection per task).
**DoD:** ≥2 optimisations implemented, each with a measured before/after cost number on real runs, and
no unreported accuracy regression — an accuracy cost that is measured and stated is acceptable; an
unmeasured one is not.

### Phase 12 — Formalise the evaluation harness
**Do:** Finalise `benchmarks/harness.py`: run **any** configuration against a fixed task set and emit one
table carrying quality **and** cost together (`context.md` §8). Add ablations via the same harness
(no router / no verifier / no fallback / no swap-aware scheduler / no abstention). Add the frozen golden
set in `evals/`. Fixed seeds, pinned model digests, **mean ± std over ≥3 seeds**, and a significance test
(paired bootstrap or McNemar) wherever you claim a difference.
**DoD:** one command reproduces the full suite and regenerates `docs/RESULTS.md`; the ablation table
shows each component's measured contribution; the golden set passes in a local CI-style run; no claimed
difference lacks a significance test.

### Phase 13 — Deployment and packaging
**Do:** Versioned, installable library. CLI entry point (`slmflow run`, `slmflow bench`, `slmflow
inspect`, `slmflow devicecard`). At least one deployment path matched to the hardware constraint (local
CLI and/or a small local HTTP service). **Attempt the T0 phone path** (Termux + llama.cpp ARM); if
infeasible, scope down explicitly per §2 and write exactly what was and was not achieved.
**DoD:** `pip install .` works in a clean venv; the documented quickstart runs a real pipeline within
minutes on T1; the T0 result — success or a documented scoped-down outcome — is written in
`docs/RESULTS.md`.

### Phase 14 — Monitoring and observability
**Do:** `telemetry/`: structured JSONL trace per run recording every sub-task, model choice, **and the
reason for the choice** (needed for debugging and for the paper's qualitative examples). Record per
`context.md` §8: latency (cold and warm), peak RAM, CPU utilisation, load/swap count and time, tokens
generated, model call count. `slmflow inspect` renders recent runs. **Skip energy measurement** unless
you have a real method — do not fake it.
**DoD:** every framework run emits a trace artifact and a telemetry record; both are inspectable via a
documented command; a worked qualitative example (one trace, explained) appears in `docs/RESULTS.md`.

### Phase 15 — Breadth: other use cases
**Do:** Exercise the framework on ≥2 task types beyond pharma — e.g. general QA/reasoning and structured
extraction — to check it is not overfit to one workflow.
**DoD:** ≥2 additional worked examples in `examples/`, each with measured results in `docs/RESULTS.md`,
and an honest note on anything the framework made *harder* rather than easier.

### Phase 16 — The headline validation
**Do:** Run the full comparison matrix from `context.md` §8: each single SLM alone; **one larger single
model of comparable total memory footprint**; SLM-MUX as baseline; a **minimal LangGraph-style static
pipeline you actually implement** (do not assert it would be worse); and SLMFlow's routed system. Locate
the quality-vs-cost break-even point.
**DoD:** Pareto frontier (quality vs. latency, quality vs. peak RAM) plotted from real data; a written,
honest answer to *"when does orchestrating small models beat one bigger model on this laptop?"* —
**including the regime where it does not**; the break-even point stated numerically with its conditions.

### Phase 17 — Adoption polish
**Do:** Clean public API (`slmflow.Pipeline`, `slmflow.Router`, `slmflow.Registry`), sensible defaults,
config templates, and error messages that name the fix. Auto-generate a starter config from a device
probe (`slmflow init`).
**DoD:** following **only** the README quickstart, you build a working pipeline in under ~15 minutes —
verify this literally, from a clean venv, without reading source; log what tripped you up and fix it.

### Phase 18 — Documentation
**Do:** `README.md` (what/why/quickstart/limitations), `docs/ARCHITECTURE.md` kept current, API reference
from docstrings, `docs/RESULTS.md`, `docs/DECISIONS.md`, and `docs/COMPARISON.md` (vs
LangChain/LangGraph/SLM-MUX, drawn from Phase 1).
**DoD:** all exist, match the actual code, and contain **no placeholder number presented as real**. The
README states the hardware tiers honestly, including what does not work.

### Phase 19 — Pharma demonstration
**Do:** End-to-end (`context.md` §7) over the existing `data/` corpus: private-style local documents →
retrieval → extraction/summarisation agents → verification against source → answer with citations, or
abstention. Frame it plainly in the demo's own README: **a demonstration of the framework, not a
clinical, diagnostic, or medical-advice product.** Use only the synthetic/public-domain corpus already in
`data/` — never real confidential data.
**DoD:** runnable demo script/notebook; `examples/pharma_demo/README.md` with a walkthrough and **real
captured output**, including at least one example where the framework correctly abstains.

### Phase 20 — Final comparative writeup
**Do:** SLMFlow vs LangChain/LangGraph-style static pipeline vs raw SLM-MUX vs single-model baselines vs
larger-single-model, using Phase 16 data. Be explicit about where SLMFlow does **not** win.
**DoD:** `docs/RESULTS.md` ends with a summary table and an honest "what worked / what didn't / what's
future work" section, cross-referenced to `docs/DECISIONS.md`, and a paragraph per claim C1–C4 stating
whether the evidence supports it, partially supports it, or does not.

---

## 7. When to stop and ask

Interrupt the loop **only** when:
- a choice would destroy something not recoverable from git history, or would materially change the
  project's thesis beyond what Ground Rule 4 already settles;
- you need something you cannot obtain yourself — a paid resource, credentials, hardware, or a physical
  Android device for the T0 test;
- you have scoped a phase down twice in a row without closing the gap. Then surface a status summary and
  a recommendation, and wait.

Everything else — model choices, task sets, config format, benchmark substitutes, library choices — is
yours to decide and log in `docs/DECISIONS.md`. **Prefer a documented, reasonable default over stalling
to ask.**

---

## 8. Definition of Done for the whole project

- Every phase has a Phase Report in `docs/PROGRESS.md`.
- `docs/RESULTS.md` presents real, reproducible numbers for: single-model baselines, SLM-MUX baseline,
  a static-pipeline baseline, a larger-single-model baseline, and SLMFlow's routed system — with the
  Pareto frontier, the break-even discussion, router regret, scheduler regret, risk-coverage, and ECE.
- The framework is `pip install`-able; base install is pure-Python; a fresh user runs the quickstart and
  the pharma demo from README instructions alone.
- The T0 (phone) claim is either demonstrated with measurements or explicitly scoped down in writing.
- `docs/DECISIONS.md` and `docs/RESEARCH_SURVEY.md` together give an honest, source-grounded account of
  what is adopted from prior work versus original to SLMFlow, including any claim the literature already
  answered.
- **No result, table, or claim anywhere in `docs/` or `README.md` is unreproducible from a script in the
  repo.**

Begin with Phase 0, then Phase 1.
