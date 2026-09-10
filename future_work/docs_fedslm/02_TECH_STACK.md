# FedSLM — Tech Stack & Installation

**Companion to** `01_PRD.md` · v1.0 · 10 August 2026

This document covers two things: **what we are using and why**, and **exactly what you have to install**. Section 8 is the checklist — if you only read one section, read that one.

---

## 1. Your machine, and what it means

Detected on this laptop:

| | |
|---|---|
| CPU | Intel Core i5-1335U (10 cores, low-power laptop chip) |
| GPU | Intel Iris Xe integrated — **no CUDA, no usable VRAM for training** |
| RAM | 16 GB |
| Disk free | 230 GB |
| Python | **3.14.5 — too new. PyTorch has no wheels for it.** |
| OS | Windows 11 |

**Two consequences, and they shape everything below.**

**Consequence 1 — you cannot fine-tune locally.** Iris Xe has no CUDA. Training even a 0.5B model with LoRA on this CPU would take days per federated round. So the work splits across two environments:

```
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│  YOUR LAPTOP  (CPU, Windows)          │   │  GOOGLE COLAB  (free T4, 16GB VRAM)   │
│                                       │   │                                       │
│  · dataset download / clean / split   │   │  · single-SLM fine-tuning             │
│  · FAISS retrieval index              │   │  · federated simulation (Flower)      │
│  · Evidence Agent (CPU embeddings)    │   │  · all training rounds                │
│  · evaluation + metric computation    │   │                                       │
│  · Streamlit dashboard                │   │  outputs: LoRA adapters (~8 MB each)  │
│  · writing code, git, docs            │   │  ↓ download to laptop for inference   │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
```

The adapters are only ~8 MB, so moving results back and forth is trivial. Inference with a 0.5B model **does** run acceptably on your CPU (a few seconds per answer) — so the demo works on your laptop with no internet.

**Consequence 2 — you must install Python 3.11.** Do not uninstall 3.14; just add 3.11 alongside it and point the project at it. Instructions in §8.

---

## 2. Stack at a glance

| Layer | Choice | Why this one |
|---|---|---|
| Language | **Python 3.11** | Last version with complete, stable support across torch + flwr + bitsandbytes |
| Env manager | **venv** (stdlib) | No extra install; conda is optional and heavier |
| DL framework | **PyTorch 2.x** | Everything below is built on it |
| Model hub / training | **Hugging Face** — transformers, datasets, peft, trl, accelerate | The standard stack; PEFT gives us LoRA in ~5 lines |
| Quantization | **bitsandbytes** (Colab only) | 4-bit NF4 base weights → QLoRA fits a free T4 |
| Federated learning | **Flower (`flwr`)** | Framework-agnostic, simulation mode, small API — you write two methods and it works |
| Retrieval | **sentence-transformers + faiss-cpu** | MiniLM embeddings, exact search over ~50k passages; runs on your CPU in ms |
| Metrics | **scikit-learn + evaluate** | Standard, citable implementations — no hand-rolled F1 |
| Config | **Hydra / OmegaConf** (YAML) | One config tree; experiment variants become CLI overrides, which is what makes runs reproducible |
| Tracking | **CSV + JSON on disk** | No account, no network, works offline, trivially auditable. W&B optional. |
| Plotting | **matplotlib + plotly** | matplotlib for the report figures, plotly for interactive dashboard charts |
| UI | **Streamlit** | Pure Python — no Node, no JS build step, no second language |
| Testing | **pytest** | Needed for the privacy assertions in §FL7 of the PRD |
| Formatting | **ruff** | Lint + format in one tool, very fast |

**What we deliberately did not use:**

| Rejected | Reason |
|---|---|
| LangChain / LangGraph | Heavy abstraction over what is a 40-line sequential loop. Adds dependency churn and hides the logic your examiner needs to see. |
| CrewAI / AutoGen | Multi-agent frameworks designed for autonomous agent swarms. Our four components are a fixed pipeline. Wrong tool. |
| Ollama / llama.cpp | Great for chat, awkward for training and for extracting token logprobs — which the Clinical Agent must produce. |
| FedML / FATE | Heavier than Flower with worse docs for a student project. |
| Weights & Biases (required) | Needs an account and network. Kept as an **optional** flag, not a dependency. |
| Docker | Adds a layer of debugging on Windows for no benefit in a single-machine prototype. |
| A database | `results/` on disk is enough. Postgres for a prototype is ceremony. |

---

## 3. Repository layout

```
C:\FYP\implementation\
├── docs\
│   ├── 01_PRD.md
│   ├── 02_TECH_STACK.md          ← you are here
│   └── 03_UI_LAYOUT.md
├── configs\
│   ├── config.yaml               # root config
│   ├── model\qwen05b.yaml
│   ├── data\meddialog.yaml, pubmedqa.yaml, medmcqa.yaml
│   ├── federated\fedavg.yaml, fedprox.yaml
│   └── experiment\e1_comparison.yaml, e3_iid_vs_noniid.yaml
├── src\fedslm\
│   ├── data\                     # download, clean, preprocess, partition, stats
│   ├── models\                   # loading, LoRA setup, quantization
│   ├── training\                 # baseline trainer, local client trainer
│   ├── agents\
│   │   ├── base.py               # AgentState dataclass
│   │   ├── clinical.py
│   │   ├── evidence.py
│   │   ├── safety.py
│   │   ├── verifier.py
│   │   └── orchestrator.py
│   ├── federated\
│   │   ├── client.py             # Flower NumPyClient
│   │   ├── server.py             # strategy + aggregation
│   │   └── strategies.py         # FedAvg, FedProx
│   ├── retrieval\                # FAISS index build + search
│   ├── evaluation\               # metrics, reliability scoring, comparison runner
│   └── utils\                    # seeding, logging, io
├── experiments\                  # one script per experiment ID (e1_…, e2_…)
├── notebooks\                    # Colab notebooks for GPU phases
├── app\                          # Streamlit dashboard
│   ├── Home.py
│   ├── pages\1_Ask.py … 5_Runs.py
│   ├── components\
│   └── assets\theme.css
├── tests\
├── data\                         # gitignored
│   ├── raw\  processed\  partitions\
│   │            ├── noniid\hospital_a\ hospital_b\ hospital_c\
│   │            └── iid\hospital_a\ hospital_b\ hospital_c\
│   └── index\                    # FAISS index
├── checkpoints\                  # gitignored — LoRA adapters
├── results\                      # committed — the evidence for every number
├── requirements.txt              # laptop (CPU)
├── requirements-gpu.txt          # Colab (CUDA)
└── README.md
```

**Two requirements files on purpose.** The laptop install must not pull CUDA wheels (2+ GB it cannot use) and must not pull `bitsandbytes` (needs CUDA). Colab installs the GPU set.

---

## 4. Component detail

### 4.1 Model layer
- `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face Hub. Apache 2.0, no access request, no token needed.
- Loaded 4-bit NF4 with double quantization on Colab; loaded fp32/bf16 on the laptop for CPU inference.
- LoRA via `peft`: r=16, α=32, dropout 0.05, targeting `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- **`get_peft_model_state_dict()` is the exact set of tensors we federate.** Nothing else crosses the wire.

### 4.2 Federated layer
- Flower **simulation** (`flwr.simulation.start_simulation`), 3 clients, backed by Ray.
- Two methods per client: `fit()` (train locally, return adapter weights) and `evaluate()` (score the global adapter on a local val split).
- Server strategy: `FedAvg` weighted by `num_examples`. `FedProx` subclasses it with a proximal term added to the local loss.
- **Windows caveat:** Ray on Windows is not reliable. Run the federated simulation **on Colab** (Linux) — this is where it belongs anyway, since it needs the GPU. If you want it locally, use WSL2.
- `client_resources={"num_gpus": 1.0}` forces one client on the GPU at a time — clients run sequentially. That is intentional: it is what makes 3 clients fit in 16 GB of VRAM.

### 4.3 Retrieval layer
- `sentence-transformers/all-MiniLM-L6-v2` — 22M params, 384-dim, CPU-fast.
- `faiss-cpu` with a flat inner-product index over normalised embeddings (= exact cosine search). At ~50k passages, exact search takes milliseconds; approximate indexing would add complexity for no gain.
- Index built once and saved to `data/index/`. The dashboard loads it in ~2 s.

### 4.4 Evaluation layer
- `scikit-learn` for accuracy / precision / recall / macro-F1.
- Grounding and hallucination: claim-level check against retrieved passages, using a lightweight NLI model with an n-gram-overlap fallback.
- Latency: `time.perf_counter()` around the orchestrator, p50/p95 over the test set.
- Communication cost: measured from the byte length of serialised parameter arrays, not estimated from parameter counts.
- Every run writes `results/<experiment_id>/{metrics.json, predictions.jsonl, config.yaml, env.json}`. `env.json` records package versions and the git commit — that is what makes a number reproducible.

---

## 5. Free resources — everything, and none of it costs money

| Need | Free option | Limits |
|---|---|---|
| GPU for training | **Google Colab (free)** — T4, 16 GB VRAM | ~12 h sessions, can disconnect; usage-limited if heavy |
| GPU backup | **Kaggle Notebooks** — P100 or 2× T4 | 30 h/week, more stable sessions, better for long federated runs |
| Models | **Hugging Face Hub** | Free, no token needed for Qwen2.5 |
| Datasets | **Hugging Face Hub** (`datasets` library) | Free |
| Storage for checkpoints | **Google Drive** (15 GB free), mounted in Colab | Our adapters are ~8 MB, so this is generous |
| Code hosting | **GitHub** free | Private repos included |
| Experiment tracking | Local CSV/JSON — **or** W&B free personal tier | W&B free tier is unlimited for personal projects |
| Hosting the demo | **Streamlit Community Cloud** free | 1 GB RAM — enough for the dashboard reading saved results, **not** enough to run the model. Local demo for the viva. |
| Editor | **VS Code** free | — |

**Total cost of this project: $0.** There is no paid dependency anywhere in the stack.

**Colab survival tips** (you will need these in Phase 6):
- Save the global adapter to Google Drive after **every** round, not at the end.
- Keep the browser tab open; Colab disconnects idle sessions.
- Run federated rounds in batches of 3–4 per session rather than all 10 at once.
- If you hit the usage limit, switch to Kaggle for a day — the code is identical.

---

## 6. Package list

### 6.1 `requirements.txt` — laptop (CPU, Windows)

```txt
# ── core ───────────────────────────────────────────────
torch==2.5.1                      # CPU build, see install note below
transformers==4.46.3
datasets==3.1.0
accelerate==1.1.1
peft==0.13.2
sentencepiece==0.2.0
safetensors==0.4.5

# ── retrieval ──────────────────────────────────────────
sentence-transformers==3.2.1
faiss-cpu==1.9.0

# ── federated (client code + config; simulation runs on Colab)
flwr==1.13.0

# ── evaluation ─────────────────────────────────────────
scikit-learn==1.5.2
evaluate==0.4.3
numpy==1.26.4
pandas==2.2.3
scipy==1.14.1

# ── config & utils ─────────────────────────────────────
hydra-core==1.3.2
omegaconf==2.3.0
python-dotenv==1.0.1
tqdm==4.67.0
rich==13.9.4

# ── ui & plots ─────────────────────────────────────────
streamlit==1.40.1
plotly==5.24.1
matplotlib==3.9.2
seaborn==0.13.2

# ── dev ────────────────────────────────────────────────
pytest==8.3.3
ruff==0.7.4
ipykernel==6.29.5
jupyter==1.1.1
```

### 6.2 `requirements-gpu.txt` — Colab / Kaggle additions

```txt
bitsandbytes==0.44.1              # 4-bit quantization — CUDA only
trl==0.12.1                       # SFTTrainer
flwr[simulation]==1.13.0          # pulls Ray for the federated simulation
ray[default]==2.38.0
```

> `bitsandbytes` will not install usefully on your laptop — it needs CUDA. That is expected and not a problem, because all quantized training happens on Colab.

---

## 7. Version pinning policy

Every version above is pinned exactly. This is not pedantry — a reproducibility requirement in the PRD says a third party must reproduce your numbers, and an unpinned `transformers` will silently change tokenizer behaviour between your experiment and their rerun.

`results/*/env.json` records the resolved versions at run time, so if a number ever looks wrong you can check whether the environment moved under you.

---

## 8. INSTALLATION — do this in order

### Step 1 — Install Python 3.11

Your 3.14.5 will not work with PyTorch. Install 3.11 alongside it.

1. Download **Python 3.11.9 (Windows installer, 64-bit)** from python.org.
2. During install: tick **"Add python.exe to PATH"** is *optional* here — we will call it by full path, which avoids clobbering your 3.14.
3. Verify:

```powershell
py -3.11 --version
# expected: Python 3.11.9
```

If `py -3.11` is not found, use the full path, typically:
`C:\Users\Aashi\AppData\Local\Programs\Python\Python311\python.exe`

### Step 2 — Install Git

Download from git-scm.com. Verify:

```powershell
git --version
```

The project is not yet a git repository — we will `git init` in Phase 1.

### Step 3 — Create the virtual environment

```powershell
cd C:\FYP\implementation
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version          # must print 3.11.9
python -m pip install --upgrade pip
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Step 4 — Install PyTorch (CPU build) first, separately

Install torch *before* the rest, from the CPU index. Otherwise pip pulls the ~2.5 GB CUDA build you cannot use.

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.5.1+cpu False      ← False is correct on this laptop
```

### Step 5 — Install everything else

```powershell
pip install -r requirements.txt
```

Roughly 2–3 GB of downloads. Give it 10–20 minutes.

### Step 6 — Verify the install

```powershell
python -c "import transformers, datasets, peft, flwr, faiss, sentence_transformers, streamlit, sklearn; print('all imports OK')"
```

### Step 7 — Download the model once (~1 GB, then cached)

```powershell
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; m='Qwen/Qwen2.5-0.5B-Instruct'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m); print('model cached')"
```

Cached to `C:\Users\Aashi\.cache\huggingface\`.

### Step 8 — Set up Colab (no local install needed)

1. Sign in to **colab.research.google.com** with a Google account.
2. **Runtime → Change runtime type → T4 GPU**.
3. First cell of every notebook:

```python
!nvidia-smi
!pip install -q transformers==4.46.3 datasets==3.1.0 peft==0.13.2 accelerate==1.1.1 \
                bitsandbytes==0.44.1 trl==0.12.1 "flwr[simulation]==1.13.0" \
                sentence-transformers==3.2.1 faiss-cpu==1.9.0
from google.colab import drive; drive.mount('/content/drive')
```

Colab ships torch pre-installed with CUDA — do not reinstall it.

### Step 9 — VS Code (recommended)

Install VS Code + the **Python** and **Jupyter** extensions. Then `Ctrl+Shift+P → Python: Select Interpreter →` pick `.venv`.

---

## 9. Install checklist

| # | Item | Where | Required? |
|---|---|---|---|
| 1 | Python 3.11.9 | python.org | **Yes** — 3.14 will not work |
| 2 | Git | git-scm.com | **Yes** |
| 3 | `.venv` virtual environment | local | **Yes** |
| 4 | PyTorch 2.5.1 **CPU build** | pip, CPU index | **Yes** — install before other packages |
| 5 | `requirements.txt` | pip | **Yes** |
| 6 | Qwen2.5-0.5B-Instruct | HF cache | **Yes** |
| 7 | Google account + Colab | web | **Yes** — this is your GPU |
| 8 | Google Drive | web | **Yes** — checkpoint storage |
| 9 | VS Code + Python extension | code.visualstudio.com | Recommended |
| 10 | Kaggle account | kaggle.com | Recommended — Colab backup |
| 11 | GitHub account | github.com | Recommended — version control |
| 12 | CUDA toolkit | — | **No.** Never installed locally. Colab provides it. |
| 13 | Docker | — | **No.** |
| 14 | Weights & Biases | — | **No.** Optional only. |

---

## 10. Known Windows friction, and the fix for each

| Problem | Fix |
|---|---|
| `bitsandbytes` fails to import | Expected — CUDA only. Never import it in laptop code paths; guard it behind a device check. |
| Ray / `flwr[simulation]` unstable | Do not run the simulation on Windows. It belongs on Colab. WSL2 if you must run it locally. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Long-path errors during pip install | Enable long paths: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force` (needs admin) |
| `datasets` download is slow or stalls | `$env:HF_HUB_ENABLE_HF_TRANSFER=1` after `pip install hf_transfer` |
| Disk fills up | HF cache lives in `C:\Users\Aashi\.cache\huggingface\`. You have 230 GB free; the whole project needs under 30 GB. |
| Tokenizer parallelism warnings | `$env:TOKENIZERS_PARALLELISM="false"` |

---

## 11. Running things

| Task | Where | Command |
|---|---|---|
| Download datasets | Laptop | `python -m fedslm.data.download` |
| Clean + preprocess | Laptop | `python -m fedslm.data.preprocess` |
| Partition into hospitals | Laptop | `python -m fedslm.data.partition partition=noniid` |
| Dataset statistics | Laptop | `python -m fedslm.data.stats` |
| Build FAISS index | Laptop | `python -m fedslm.retrieval.build_index` |
| Fine-tune baseline | **Colab** | `notebooks/04_baseline_finetune.ipynb` |
| Federated training | **Colab** | `notebooks/06_federated.ipynb` |
| Run the agent pipeline | Laptop | `python -m fedslm.agents.orchestrator --query "..."` |
| Run experiment E1 | Laptop | `python -m experiments.e1_comparison` |
| Launch the dashboard | Laptop | `streamlit run app/Home.py` |
| Tests | Laptop | `pytest tests/ -v` |

---

**Next document:** `03_UI_LAYOUT.md` — the dashboard design specification.
