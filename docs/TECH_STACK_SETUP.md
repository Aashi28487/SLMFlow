# Tech stack — what to install on the target machine

Target machine: **Intel i5 HX-series, discrete GPU with ~6 GB VRAM, 16 GB RAM,
512 GB SSD, Windows.**

Install in the order below. Everything in **Tier 0** is required; each later
tier buys a specific capability and can wait until you need it.

The design constraint that shapes this whole list: `import slmflow` must work
with **nothing installed but Python**. Every heavy package is optional and
lazily imported, because the T0 phone tier cannot install numpy or torch. So a
partial install is a valid install.

---

## Tier 0 — required to run anything (~250 MB)

| What | Version | Why | How |
|---|---|---|---|
| **Python** | 3.10–3.13 | 3.10 is the floor (`slots=True` dataclasses, PEP 604 unions). **Avoid 3.14 for now** — `llama-cpp-python` and `faiss-cpu` wheels lag a new release by months. 3.12 is the safe choice. | [python.org](https://www.python.org/downloads/) — tick **Add python.exe to PATH** |
| **Git** | any | Ground Rule 7: commit after each increment | [git-scm.com](https://git-scm.com/download/win) |
| **pip + venv** | bundled | isolation | `python -m venv .venv` |

```powershell
cd C:\FYP\implementation
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -c "import slmflow; print(slmflow.__version__)"
```

`[dev]` pulls pytest, pytest-asyncio, hypothesis, mypy, ruff, psutil, PyYAML.
That is enough to run the **entire test suite with no model weights**.

> **Note on 3.14.** This repo was authored on a machine running Python 3.14.5.
> The library targets `>=3.10` and uses nothing newer than 3.10 features, so it
> should import fine on 3.14 — but `llama-cpp-python` may have no wheel there.
> If Tier 2 fails to install, that is why; drop to 3.12.

---

## Tier 1 — required for real models (~700 MB + weights)

Two backends. **Start with Ollama**: it is one installer, it handles GPU offload
itself, and it will not make you build a C++ extension.

### Ollama (recommended first)

| What | Why | How |
|---|---|---|
| **Ollama** | out-of-process daemon; manages GGUF downloads and CUDA offload for you | [ollama.com/download/windows](https://ollama.com/download/windows) |

```powershell
ollama --version
ollama serve                 # leave running in its own terminal
ollama pull qwen2.5:0.5b-instruct-q4_K_M
ollama list                  # record the digest for reproducibility
```

Then: `pip install "slmflow[ollama]"` (adds `httpx`; the stdlib fallback works
without it, just slower).

**Trade-off, and it matters for the thesis.** Ollama holds weights in the
*daemon's* address space, so per-model memory comes from its `/api/ps`
endpoint. When that is unavailable, SLMFlow falls back to summing RSS of all
`ollama` processes and **logs `coarse_measurement`** — it cannot attribute
bytes to one model. Ollama also has **no GBNF parameter**, so grammar-constrained
decoding is unavailable there; only JSON-format mode is. For the Phase 10
grammar before/after comparison and for precise per-model memory, use
llama.cpp.

### llama-cpp-python (needed for grammars, log-probs and exact memory)

| What | Why | How |
|---|---|---|
| **llama-cpp-python** | in-process inference; GBNF grammars; log-probs; exact RSS-delta accounting | see below |
| **NVIDIA driver + `nvidia-smi`** | **required** for honest VRAM measurement | ships with the GeForce/Studio driver |

CPU-only build (always works):

```powershell
pip install llama-cpp-python
```

CUDA build, to use your 6 GB VRAM:

```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install --no-cache-dir --force-reinstall llama-cpp-python
```

The CUDA build needs the **CUDA Toolkit** (12.x) and **Visual Studio Build
Tools** with the C++ workload. If that is more than you want to fight, there
are prebuilt CUDA wheels:

```powershell
pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**Verify `nvidia-smi` is on PATH before profiling anything:**

```powershell
nvidia-smi --query-gpu=name,memory.total --format=csv
```

If this fails, SLMFlow falls back to a Windows WMI field (`AdapterRAM`) that is
32-bit and **saturates at 4 GB**. `slmflow doctor` warns you. Either fix the
PATH or pin the budget: `$env:SLMFLOW_VRAM_BUDGET="5GB"`.

### Model weights — budget the disk

512 GB shared with `data/`, so `BUILD_BRIEF.md` §2 caps total GGUF at ~40 GB.

| Model class | Q4_K_M size | Fits 6 GB VRAM? | Role |
|---|---|---|---|
| 0.5–1.5B | 0.4–1.1 GB | comfortably | Stage A/B; the T0 candidate; the tiny verifier |
| 3–4B | 2.0–2.7 GB | yes | the workhorse |
| 7–8B | 4.4–4.9 GB | yes, at modest context | **the Phase 16 big-model baseline** |
| 13B+ | 7.5 GB+ | no | out of scope; do not download |

Follow the §2.1 ramp: one model, measure, commit the card, *then* decide on the
next download. Record the digest and delete weights you have profiled and
rejected — the digest keeps the run reproducible without the file.

---

## Tier 2 — required for the RAG phase (~500 MB)

```powershell
pip install "slmflow[rag]"       # sentence-transformers, faiss-cpu, numpy
```

**You may not need this.** `BM25Retriever` is pure Python, always available,
and is a genuinely strong baseline on this corpus because the queries carry
exact drug and indication names — precisely where lexical retrieval beats dense.
Install Tier 2 only when you want the dense-vs-lexical comparison. If
`faiss-cpu` has no wheel for your Python, that is another reason to use 3.12.

Note the memory interaction: the embedding model is *also* a resident model. On
6 GB VRAM it competes with the generator, which is why `DenseRetriever` has an
explicit `release()`. Index first, release, then generate.

---

## Tier 3 — required for plots and the phone tier

| What | For | How |
|---|---|---|
| `matplotlib` | risk-coverage curves, reliability diagrams | `pip install "slmflow[bench]"` |
| **Termux** + `llama.cpp` ARM | the T0 phone claim (C2) | Android device, separate setup |

The T0 tier needs an actual Android device. Per `BUILD_BRIEF.md` §2, if a full
T0 run is infeasible, scope down **in writing** to "core library imports and
runs one 0.5B model end-to-end on T0, measured" — do not quietly drop it and do
not overclaim mobile support in the README.

---

## Explicitly NOT needed

Worth stating, because each absence is a design decision:

| Not needed | Why |
|---|---|
| **torch / transformers** | inference is GGUF via llama.cpp or Ollama. Torch would add ~2.5 GB and break the T0 constraint. |
| **Any hosted LLM API key** | Ground Rule 6 forbids it in the runtime path, tests, judging and fallback. No key, no network call, no account. |
| **scipy / sklearn** | the incomplete beta, Clopper-Pearson bound, logistic regression, ECE, paired bootstrap and McNemar are all implemented in stdlib `math` so they run on T0. |
| **jsonschema** | the schema validator is a documented stdlib subset. |
| **PyYAML** (strictly) | there is a built-in YAML subset parser. Install it (`[yaml]`) for full YAML; the parser refuses anchors and aliases rather than misreading them. |
| **Docker** | a venv is enough; a container would hide the memory measurements the research depends on. |
| **CUDA Toolkit** (unless building llama.cpp from source) | Ollama and the prebuilt wheels bundle what they need. |

---

## Verify the install

```powershell
python -c "import slmflow; print(slmflow.__version__)"   # must work bare
pytest -m "not real_model"                                # no weights needed
slmflow doctor                                            # the real check
python examples/quickstart.py                             # end to end, mocked
```

`slmflow doctor` prints the probed device card, the derived two-tier budget,
which backends are live, how RSS will be measured, which model cards exist and
which fit — and exits non-zero listing what still needs attention. It is the
one command to run after any environment change.

---

## Environment variables

| Variable | Effect |
|---|---|
| `SLMFLOW_VRAM_BUDGET` | override the probed VRAM budget (`5GB`, or bytes) |
| `SLMFLOW_RAM_BUDGET` | override the probed RAM budget |

Use them when the probe is unreliable, or to *force* a small budget deliberately
— running the suite under an artificially tight ceiling is how you show the
scheduler earning its keep.

---

## Summary: minimum path to a first real number

```powershell
# 1. Python 3.12, Git
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -m "not real_model"

# 2. Ollama, one small model
ollama serve
ollama pull qwen2.5:0.5b-instruct-q4_K_M
python -m benchmarks.profile_models --model qwen2.5:0.5b-instruct-q4_K_M `
    --backend ollama --params-b 0.5
```

Total: ~1 GB of downloads and roughly an hour, most of it waiting. See
`RUN.md` for what to do with it.
