# RUN.md — running SLMFlow on the target machine

Every command here is meant to be run on **the evaluation machine**
(Intel i5 HX-series, discrete GPU ~6 GB VRAM, 16 GB RAM, 512 GB SSD).
Nothing in this repository has been executed yet: the code was written without
running it, so **Step 1 and Step 2 are where you find out whether it works.**

Read `docs/TECH_STACK_SETUP.md` first for what to install. This file is the
order to run things in.

Time estimates assume the machine above and are rough.

---

## 0. Where you are

| | |
|---|---|
| Implemented | Phases 0–12 of `BUILD_BRIEF.md` (code, tests, harness, docs) |
| Measured | **nothing** — every number in `docs/RESULTS.md` reads `not yet measured` |
| Model weights | none downloaded; `.gitignore` excludes them |
| Tests | written, **never run** |

So the first real milestone is not a result. It is *green tests*.

---

## 1. Install and verify the code even parses (5 minutes, no models)

```bash
cd C:\FYP\implementation

python -m venv .venv
.venv\Scripts\activate            # PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Then the three checks that cost nothing:

```bash
python -c "import slmflow; print(slmflow.__version__)"
ruff check src tests benchmarks
mypy
```

`import slmflow` must succeed with **no third-party packages** installed beyond
pytest — that is the T0 constraint, and if it fails, something leaked a
dependency into the core import path.

---

## 2. Run the test suite (2–5 minutes, still no models)

```bash
pytest -m "not real_model"
```

This is the gate. The whole suite runs on `MockBackend`, so it needs **zero
model weights**. It covers the two-tier budget, forced eviction, swap
accounting, the schedulers and their regret, the routers, schema/GBNF
compilation, the calibration mathematics, the abstention policy, the
significance tests, the pipeline end-to-end, and the CLI.

Expect failures on the first run. The code was never executed; treat each one
as a normal bug, fix it, and re-run. `docs/PROGRESS.md` has a section to log
what you hit.

Useful while fixing:

```bash
pytest -x -q                       # stop at the first failure
pytest tests/test_resource.py -v   # just the C1 core
pytest --collect-only -q           # what exists, without running it
```

---

## 3. See it work without any model (1 minute)

```bash
python examples/quickstart.py
slmflow probe
slmflow doctor
```

`quickstart.py` forces a model swap and prints the cost, then computes
scheduler regret against the offline oracle. Every number it prints comes from
`MockBackend` and it says so — do not cite them.

`slmflow doctor` is the important one: it reports the probed device card, the
derived two-tier budget, which backends are available, and what is missing.

**Check the VRAM line.** If `nvidia-smi` is not on PATH, the probe falls back
to a Windows WMI field that saturates at 4 GB and it will warn you. Fix it by
installing the NVIDIA driver tools, or pin the budget by hand:

```bash
set SLMFLOW_VRAM_BUDGET=5GB
set SLMFLOW_RAM_BUDGET=8GB
```

---

## 4. Record the machine (Phase 0 DoD, 1 minute)

```bash
slmflow probe > docs\ENVIRONMENT.md
```

Then open it and add the exact CPU and GPU model numbers if the probe left them
vague. Phase 0's Definition of Done requires `ENVIRONMENT.md` to hold **real
probed values**, and until this file exists the results artifacts have no
machine to attribute themselves to.

---

## 5. Model 1 — the first real measurement (30–60 minutes)

Per `BUILD_BRIEF.md` §2.1, **one model, then stop and look at the numbers.**

```bash
ollama serve                       # leave running in another terminal
ollama pull qwen2.5:0.5b-instruct-q4_K_M
ollama list                        # record the digest

python -m benchmarks.profile_models ^
    --model qwen2.5:0.5b-instruct-q4_K_M ^
    --backend ollama ^
    --params-b 0.5 ^
    --quant Q4_K_M ^
    --repeats 5 ^
    --slice 20
```

This writes `benchmarks/results/model_cards.json` with measured
`cold_load_s`, `warm_load_s`, `tokens_per_s`, per-tier footprint, and accuracy
on a 20-item slice of the real corpus.

Then run the slice end to end:

```bash
slmflow run examples/repurposing_label.yaml ^
    --input drug=metformin --input indication=cancer ^
    --trace runs\first_real.jsonl

slmflow inspect runs\first_real.jsonl
```

**Stop here and read the card.** Three things worth knowing before you spend
another download:

- Is `cold_load_s` big enough that scheduling matters? If a 0.5B model loads in
  0.3 s, C1's swap-awareness may buy little — which is a *finding*, not a
  failure (Ground Rule 9).
- Is accuracy above the majority-class baseline? The harness prints it. The
  corpus is 76 % `investigational`, so anything below ~0.76 is worse than a
  constant.
- Does the schema hold? `slmflow inspect` shows the schema-validity rate.

---

## 6. Model 2 — enough for routing (30–60 minutes)

Pick something **deliberately different**, not just bigger:

```bash
ollama pull llama3.2:3b-instruct-q4_K_M

python -m benchmarks.profile_models ^
    --model llama3.2:3b-instruct-q4_K_M ^
    --backend ollama --params-b 3.0 --quant Q4_K_M --slice 20

slmflow models                     # both cards, side by side
```

Now two models exist, so routing, ensembling and swapping are all real:

```bash
python -m benchmarks.regret --sizes 2,3,4
python -m benchmarks.train_router --family repurposing_label --limit 60
```

`regret` gives you the **C1 headline** (greedy vs oracle) and the
**swap-awareness ablation** (greedy vs naive LRU) in one table.

---

## 7. CHECKPOINT — the vertical slice is green

Per the phase-order amendment in `BUILD_BRIEF.md` §6, stop here and confirm:

- [ ] `pytest -m "not real_model"` passes
- [ ] `docs/ENVIRONMENT.md` has real probed values
- [ ] two model cards are profiled (`slmflow models` shows no `not measured`)
- [ ] one YAML pipeline runs end to end over both models and writes a trace
- [ ] `benchmarks/results/regret.json` exists
- [ ] `docs/PROGRESS.md` has a Phase Report for each phase reached

**Only now** is the full model sweep worth its download budget.

---

## 8. The full suite (Phases 2b, 3, 6, 7, 10, 11, 12)

```bash
# Phase 2b: the rest of the candidate set, one at a time, per the ramp
python -m benchmarks.profile_models --model <next-model> --backend ollama --params-b <n>

# Phase 10 / C3: risk-coverage, ECE per signal, the precision contract
python -m benchmarks.calibrate --family repurposing_label --limit 80 --target 0.9

# Phase 12: everything, 3 seeds, regenerating docs/RESULTS.md
python -m benchmarks.harness --all --seeds 3 --retrieval
```

The harness writes `docs/RESULTS.md`, `benchmarks/results/summary.json`, and
per-run JSONL traces under `benchmarks/results/raw/`. It runs the paired
bootstrap and McNemar automatically, so no difference reaches the table without
a significance test attached.

Held-out numbers, for anything you intend to cite:

```bash
python -m benchmarks.harness --all --seeds 3 --split test
python -m benchmarks.train_router --split test
```

### The big-model baseline (Phase 16)

Your 6 GB VRAM makes this runnable **on the same machine** — a 7–8B Q4_K_M is
~4.4–4.9 GB and fits. This is the comparison the whole thesis turns on:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
python -m benchmarks.profile_models --model qwen2.5:7b-instruct-q4_K_M ^
    --backend ollama --params-b 7.0
python -m benchmarks.harness --family repurposing_label --seeds 3
```

Verify the fit by loading it and reading `slmflow models`, never by arithmetic
on the file size.

---

## 9. The pharma demonstration (Phase 19)

```bash
python examples/pharma_demo.py --drug metformin --indication cancer
```

Retrieval over the real corpus, four-step pipeline, cited output, measured
cost. Add `--mock` to see the shape without weights.

---

## Dry-run everything first

Every script takes `--mock`, which substitutes `MockBackend`. Use it to prove a
command works before it costs you a model load:

```bash
python -m benchmarks.harness --all --limit 10 --seeds 1 --mock
python -m benchmarks.regret --mock
python -m benchmarks.train_router --mock --limit 40
python -m benchmarks.calibrate --mock --limit 40
```

Mock output is **stamped** as mock in `docs/RESULTS.md`. Never cite it.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `import slmflow` fails | A dependency leaked into the core import path. Find it: `python -X importtime -c "import slmflow"`. Nothing third-party may appear. |
| `ScheduleInfeasible: No registered model fits` | Budget too small or cards unprofiled. `slmflow doctor` prints the arithmetic. Override with `SLMFLOW_RAM_BUDGET`. |
| VRAM reported as 4 GB or 0 | `nvidia-smi` not on PATH. Install the driver tools, or set `SLMFLOW_VRAM_BUDGET=5GB`. |
| `BackendUnavailable: could not reach the Ollama daemon` | `ollama serve` is not running. Confirm with `ollama list`. |
| Ollama memory looks like the whole daemon | `/api/ps` was unavailable, so RSS of all `ollama` processes was attributed to one model. The log says `coarse_measurement`. Prefer the llama.cpp backend for precise per-model figures. |
| Everything abstains | The step `threshold` is above the achievable confidence. Lower it, or run `benchmarks/calibrate.py` and use the tuned threshold. |
| Grammar ignored | Ollama has no GBNF parameter; only llama.cpp constrains decoding. Use `--backend llama_cpp` for the Phase 10 grammar comparison. |
| Oracle refuses to plan | Correct behaviour: the search space exceeded `max_instances`. Shrink the DAG; the oracle is a measurement tool for small instances. |

---

## What to log as you go

`BUILD_BRIEF.md` Ground Rule 8 requires a dated entry in `docs/DECISIONS.md`
for every judgment call, and a Phase Report appended to `docs/PROGRESS.md` at
the end of each phase: what was built, what was measured **with numbers**, what
did not work, what is deferred and why.

Ground Rule 1 is the one that protects the thesis: **if it is not measured,
write `not yet measured`.** Never a plausible-looking placeholder.
