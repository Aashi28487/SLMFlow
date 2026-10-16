# SLMFlow

**Memory-bounded orchestration of small language models on local hardware.**

SLMFlow is a pure-Python library for running pipelines of small language models
(roughly 0.5B-8B parameters) on a machine whose memory cannot hold every model at
once. It treats model residency (which model is loaded, what it cost to load,
what must be evicted) as part of the execution plan, and it can decline to
answer when its confidence is too low.

> **Status: alpha (0.1.x).** The library and its test suite work. **Performance
> advantages are not established.** The first benchmark run was invalid (scorer
> and setup bugs, since fixed). A later 20-item smoke run on one laptop with
> 0.5B-3B models showed no accuracy gain over a single model, and several
> configurations scored 0.000. Do not read this project as evidence that it is
> faster, more accurate or more memory-efficient than anything else. See
> [Limitations](#limitations).

## What it does

- **Residency and budgets.** A two-tier (VRAM / RAM) memory budget, a resource
  manager that loads and evicts models (LRU or largest-first) and a ledger that
  records every load and its cost.
- **Scheduling.** A swap-aware greedy scheduler that orders and groups steps to
  reduce model loads, an offline oracle and a naive baseline for measuring
  *scheduler regret*.
- **Routing.** Heuristic, learned (pure-Python logistic regression), random,
  static and oracle routers.
- **Strategies.** Single model, SLM-MUX, agent-forest voting, cascade and
  adaptive self-consistency (Beta-Binomial early stopping).
- **Reliability.** Confidence signals, `Precision(target)` threshold tuning with
  a Clopper-Pearson lower bound, abstention policy, ECE, Brier score, paired
  bootstrap and McNemar tests.
- **Backends.** Ollama (HTTP daemon), llama.cpp (in-process) and a mock backend.
- **Zero required dependencies.** `import slmflow` loads nothing outside the
  standard library. Optional extras add backends and tooling.

## Requirements

- Python **3.10, 3.11, 3.12 or 3.13** (the CI matrix).
- No third-party packages for the core.
- A real backend only if you want real inference (see below).

## Installation

From PyPI (once released) or from a wheel:

```bash
pip install slmflow                       # core only
pip install path/to/slmflow-0.1.1-py3-none-any.whl
```

Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `slmflow[ollama]` | nothing extra (stdlib HTTP) | Ollama backend; you still need the Ollama daemon |
| `slmflow[llama]` | `llama-cpp-python` | in-process llama.cpp backend and GBNF grammars |
| `slmflow[yaml]` | `PyYAML` | full YAML syntax (a minimal built-in parser covers simple configs) |
| `slmflow[rag]` | `sentence-transformers`, `faiss-cpu`, `numpy` | dense retrieval (BM25 needs nothing) |
| `slmflow[bench]` | `numpy`, `matplotlib`, `psutil`, `PyYAML` | benchmark scripts (source checkout only) |
| `slmflow[dev]` | pytest, mypy, ruff, hypothesis, ... | running the test suite |

### Backend prerequisites

| Backend | You need |
|---|---|
| `mock` | nothing; scripted answers for tests and demos (never real inference) |
| `ollama` | [Ollama](https://ollama.com) installed and running (`ollama serve`), models pulled (`ollama pull <model>`). Ollama cannot take GBNF grammars; JSON mode is used for schema steps. |
| `llama_cpp` | `pip install "slmflow[llama]"` and GGUF files in a model directory |

If a backend is missing, SLMFlow raises `BackendUnavailable` with the command to
fix it, rather than returning an empty or wrong answer.

## Quickstart (no model weights, mock backend)

This runs anywhere. The `MockBackend` returns a scripted answer, so **the output
is not a model measurement**.

```python
from slmflow import MemoryBudget, ModelRegistry, Pipeline
from slmflow.models.backends.mock import MockBackend, make_pool

registry = ModelRegistry(make_pool(
    [("small", 0.5, 400_000_000), ("large", 3.0, 2_200_000_000)],
    accuracy={"small": {"classify": 0.62}, "large": {"classify": 0.81}},
))
registry.register_backend("mock", MockBackend(default_answer="investigational"))

pipeline = Pipeline.from_dict(
    {"name": "quickstart",
     "steps": [{"name": "label", "task_type": "classify", "prompt": "Is it?"}]},
    registry=registry,
    budget=MemoryBudget.single_slot(2_500_000_000),
)
result = pipeline.run()
print(result.answer, result.trace.total_swaps)
```

With real weights, build the registry from profiled model cards and optionally
declare a precision target:

```python
from slmflow import ModelRegistry, Pipeline, Precision

registry = ModelRegistry.from_file("model_cards.json")      # your profiled cards
pipe = Pipeline.from_config("pipeline.yaml", registry=registry,
                            guarantee=Precision(0.90))
policy = pipe.calibrate(calibration_items)                  # tunes the abstention threshold
result = pipe.run({"drug": "metformin", "indication": "cancer"})
print(result.answer, result.confidence, result.abstained)
```

Model cards come from `python -m benchmarks.profile_models` in a source
checkout (the `benchmarks/` package is not part of the wheel).

## Command line

```bash
slmflow --version
slmflow probe                          # measured device card
slmflow doctor                         # backends, budget, model cards, what to fix
slmflow plan config.yaml --mock        # residency plan without running
slmflow run  config.yaml --mock        # prints a MOCK BACKEND notice on stderr
slmflow inspect runs/trace.jsonl       # summarise a trace
slmflow models --cards model_cards.json
```

Exit codes: `0` success, `2` a library error (one actionable line on stderr),
`3` the pipeline abstained.

## Running the tests

Unit tests use the mock backend and need no model, no network and no Ollama:

```bash
git clone https://github.com/Aashi28487/SLMFlow && cd SLMFlow
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -m "not real_model"
```

Smoke test against a real backend (needs Ollama running and a model pulled):

```bash
pip install -e ".[dev,bench]"
python -m benchmarks.run_all --fresh --seeds 1 --limit 20
```

Check the output yourself: every accuracy should be above the majority-class
baseline printed in the results table before you trust it. See
[RUN.md](https://github.com/Aashi28487/SLMFlow/blob/main/RUN.md).

## Limitations

- **No demonstrated performance advantage.** Orchestration strategies
  (ensembles, cascade, adaptive self-consistency) did not beat a single
  0.5B model in our smoke run, and ensembles were 13-18x slower. Scheduler
  benefit depends on real model-load cost, and our profiling of load time was
  still being corrected when this version was prepared.
- **Small models are weak at some tasks.** A 0.5B model could not do structured
  trial lookup at all in our runs.
- Ollama gives approximate memory figures; llama.cpp gives more precise ones.
- Dense retrieval and grammar-constrained decoding need extras and (for
  grammars) the llama.cpp backend.
- The benchmark scripts and DRE-Corpus data live in the repository and are not
  installed with the package.
- Local inference only: there is no cloud-model backend.

## Documentation and project links

- Repository: <https://github.com/Aashi28487/SLMFlow>
- Architecture: <https://github.com/Aashi28487/SLMFlow/blob/main/docs/ARCHITECTURE.md>
- Results (measured numbers only): <https://github.com/Aashi28487/SLMFlow/blob/main/docs/RESULTS.md>
- Design decisions: <https://github.com/Aashi28487/SLMFlow/blob/main/docs/DECISIONS.md>
- Release procedure: <https://github.com/Aashi28487/SLMFlow/blob/main/docs/RELEASE.md>
- Issues: <https://github.com/Aashi28487/SLMFlow/issues>
- Changelog: <https://github.com/Aashi28487/SLMFlow/blob/main/CHANGELOG.md>

## Authors and licence

Aashi, Sneha, Rutvik and Prakriti. Released under the MIT licence; see
[LICENSE](https://github.com/Aashi28487/SLMFlow/blob/main/LICENSE).
