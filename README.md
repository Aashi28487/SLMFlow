# SLMFlow

**Memory-bounded orchestration of small language models on local hardware.**

Orchestrating small models on a laptop or a phone is a **scheduling problem
under a hard memory budget** — and nobody has built the scheduler.

Existing frameworks (LangChain, LangGraph, AutoGen) treat the model as an
always-available remote endpoint. Routing work (RouteLLM, FrugalGPT) optimises
*price per API call*, a setting where switching models is free. On a laptop with
6 GB of VRAM, **switching models is the single most expensive operation in the
system**, and no existing orchestrator represents it at all.

SLMFlow does. Residency, swap cost and a per-tier memory ceiling are first-class
objects in the execution model, not deployment details.

> ### Status: alpha. Code works; benchmark numbers are being re-measured
>
> The test suite passes (`pytest -m "not real_model"`). A first benchmark run on
> 2026-09-13 produced accuracy 0.000 everywhere because of scorer and setup
> bugs, which have since been fixed; **those numbers are invalid and must not
> be cited.** The full benchmark is being re-run on a stronger machine. Until
> `docs/RESULTS.md` is regenerated from that run, treat it as stale.
>
> Per Ground Rule 1, no number appears in this repository unless it came from
> code actually run on real hardware. Start at **[`RUN.md`](RUN.md)**; the
> one-command run is `python -m benchmarks.run_all`.

---

## What it does

```python
from slmflow import MemoryBudget, ModelRegistry, Pipeline, Precision

# A budget that fits either model, but not both at once. This is the normal
# case on consumer hardware, not the degenerate one.
pipeline = Pipeline.from_config("examples/repurposing_label.yaml",
                                guarantee=Precision(0.90))

policy = pipeline.calibrate(calibration_items)   # tunes its own threshold
print(policy.achieved_coverage)                  # and tells you what it cost

result = pipeline.run({"drug": "metformin", "indication": "cancer"})
print(result.answer, result.confidence, result.abstained)
print(result.trace.total_swaps, result.trace.swap_seconds)   # the hidden cost
```

Or see the whole library work with no model weights at all:

```bash
python examples/quickstart.py
```

---

## The four contributions

| | Claim | Mechanism | Measured by |
|---|---|---|---|
| **C1** | **The Memory-Bounded Orchestration Problem.** Formalised as paging (variable page sizes, variable miss costs) fused with assignment, under a *vector* budget over VRAM/RAM/disk. | Swap-aware greedy scheduler that ships, plus an offline oracle for measurement | **scheduler regret**, and an honest ablation against naive load-on-demand + LRU |
| **C2** | **Portable capability profiles.** A measured device card × a measured model card *is* the routing table. Does a policy calibrated on one device transfer to a weaker one? | `DeviceCard`, `DeviceTier`, per-tier budgets | needs a second device; scoped down in writing if none materialises |
| **C3** | **Abstention as a declarative contract.** `guarantee=Precision(0.90)` and the framework tunes its own thresholds to meet it, then reports achieved coverage. No orchestration framework exposes this. | Learn-then-Test with an exact Clopper-Pearson bound | risk–coverage curve, ECE per signal, **held-out** precision and coverage |
| **C4** | **Cost-adaptive self-consistency.** SLM-MUX costs `k × \|M\|` generations with no early exit; on CPU that is brutal. | Beta-Binomial early stopping; cascade escalation | samples drawn per item at matched accuracy |

---

## Design constraints that shaped everything

- **The core imports and runs on a phone.** `import slmflow` pulls **zero**
  third-party packages. No numpy, no torch, no pydantic-core. The incomplete
  beta function, Clopper-Pearson bound, logistic-regression router, ECE, paired
  bootstrap, McNemar test, YAML parser and JSON-schema validator are all stdlib
  `math` — because they have to run on T0.
- **Single-slot residency by default.** The resource manager assumes the budget
  may fit exactly one model. Multi-residency is the optimisation, never the
  assumption.
- **Local inference only.** llama.cpp or Ollama. No hosted API anywhere in the
  runtime path, the tests, the judging or the fallback — not "just for the demo".
- **Every cost reported cold and warm.** Load time dominates on this hardware;
  hiding it would be dishonest.
- **`None` means "not yet measured", and it propagates.** From `ModelCard` through
  the cost model, the schedule, the regret calculation, to `not measured` in the
  rendered table. There is no code path that turns an unmeasured quantity into
  `0.0`. Ground Rule 1 is enforced by the type system, not by discipline.

---

## Install

```bash
pip install -e ".[dev]"        # everything needed to run the full test suite
pip install -e ".[ollama]"     # + a local Ollama daemon
pip install -e ".[llama]"      # + in-process llama.cpp, GBNF grammars, log-probs
pip install -e ".[rag]"        # + dense retrieval (BM25 needs nothing)
```

The base install is intentionally tiny and pure-Python. See
[`docs/TECH_STACK_SETUP.md`](docs/TECH_STACK_SETUP.md) for what to install on a
real machine and, more usefully, **what you do not need**.

## Command line

```bash
slmflow probe                       # measured device card
slmflow doctor                      # what works, what is missing, what to fix
slmflow plan   config.yaml          # the residency plan, before paying for it
slmflow run    config.yaml --mock   # run it; --mock needs no weights
slmflow inspect runs/trace.jsonl    # summarise a trace
slmflow models                      # registered cards, flagging unmeasured ones
```

Every benchmark script takes `--mock`, so you can prove a command works before
it costs you a model load.

## Documentation

| | |
|---|---|
| [`RUN.md`](RUN.md) | **start here** — the order to run things in, on the target machine |
| [`docs/TECH_STACK_SETUP.md`](docs/TECH_STACK_SETUP.md) | what to install, tiered by what it buys |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | MBOP stated formally; design rationale; known limitations |
| [`docs/RESULTS.md`](docs/RESULTS.md) | the numbers (currently all `not yet measured`) |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | every judgment call, dated, with what was given up |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | phase-by-phase status |
| [`docs/RESEARCH_SURVEY.md`](docs/RESEARCH_SURVEY.md) | novelty check per claim — **needs literature verification** |
| [`docs/COMPARISON.md`](docs/COMPARISON.md) | versus LangChain / LangGraph / SLM-MUX, including what we lack |
| [`BUILD_BRIEF.md`](BUILD_BRIEF.md) | the specification this was built against |

## The test suite runs with no model weights

```bash
pytest -m "not real_model"
```

`MockBackend` simulates per-model latency, cold-versus-warm load cost,
inter-model disagreement and sampling spread, so residency, scheduling, routing,
ensembling, calibration and abstention are all testable on any machine.

## Data

The DRE-Corpus in `data/` is reused, not regenerated: 19,378 passages from
PubMed, ClinicalTrials.gov and openFDA, with 140 drug × indication pairs carrying
derived `approved` / `investigational` / `hypothesis` labels and a 64-item
held-out split over 12 unseen drugs. See `data/processed/repurposing/DATASET_CARD.md`.

The label distribution is skewed 106/24/10, so the **majority-class baseline is
0.757**. Every results table prints it first, because an accuracy of 0.70 would
otherwise look like competence.

## Licence

MIT.
