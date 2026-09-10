# FedSLM — Product Requirements Document

**A Federated Framework for Reliable Multi-Agent Small Language Models in Healthcare**

| | |
|---|---|
| **Document** | PRD v1.0 |
| **Date** | 10 August 2026 |
| **Status** | Draft — awaiting approval before implementation |
| **Project type** | Final-year research prototype |
| **Target hardware** | Intel i5-1335U · 16 GB RAM · Iris Xe (no CUDA) + Google Colab free tier |

> **Scope note.** This is a research prototype for evaluating an architecture. It is **not** a medical device, not a diagnosis system, and must never be presented as clinical advice. Every UI surface carries this disclaimer.

---

## 1. What FedSLM is, in plain language

Three problems in healthcare AI, and the one system that addresses all three.

**Problem 1 — Hospitals cannot share patient data.** Privacy law and hospital policy mean Hospital A's records never leave Hospital A. So each hospital trains on its own small dataset and gets a weak model.

**Problem 2 — Big models are unaffordable.** A hospital cannot run a 70-billion-parameter model on a ward workstation, and sending queries to a cloud API sends patient text off-site.

**Problem 3 — Language models make things up.** A single model produces one answer with no evidence, no safety check, and no way to tell a good answer from a confident-sounding wrong one.

**FedSLM's answer to each:**

| Problem | Mechanism | What it means concretely |
|---|---|---|
| Cannot share data | **Federated learning** | Each hospital trains locally. Only *model weight updates* (a few MB of LoRA adapter) travel to the server. Raw text never leaves. |
| Cannot afford big models | **Small language model + LoRA** | A 0.5B-parameter model, fine-tuned with low-rank adapters, runs on a laptop. |
| Models hallucinate | **Multi-agent + verifier** | Four components inspect the answer in sequence and stamp it PASS, REVIEW, or FAIL before the user sees it. |

The research claim we will test:

> Combining federated training with a multi-agent verification pipeline produces answers that are measurably more accurate, better grounded in evidence, and safer than a single fine-tuned SLM — at a communication cost small enough to be practical.

We test this claim by building four systems and measuring all of them under identical conditions. **We do not assume FedSLM wins.** If it does not, that is the result we report.

---

## 2. Goals and non-goals

### 2.1 Goals

| ID | Goal | Success looks like |
|---|---|---|
| G1 | A working end-to-end prototype, runnable by a third party | A fresh clone + one setup command + documented run steps reproduce every reported number |
| G2 | Three real public healthcare datasets, cleaned and partitioned non-IID across 3 simulated hospitals | `data/processed/` and `data/partitions/` exist with a printed statistics report |
| G3 | A single-SLM fine-tuned baseline | Measured accuracy on a held-out test set, checkpoint saved |
| G4 | A four-component multi-agent pipeline with PASS/REVIEW/FAIL verdicts | Any query returns a final answer plus a full agent trace |
| G5 | Federated training with Flower — FedAvg across 3 clients, ≥5 rounds | Per-round loss, accuracy, wall-clock time, and MB transferred logged to CSV |
| G6 | A reliability layer scoring correctness, grounding, hallucination, safety, confidence | Every answer carries a reliability score with component breakdown |
| G7 | A four-way comparison table from real experiments | Single / Multi-Agent / Federated / FedSLM measured on identical test sets |
| G8 | A Streamlit dashboard demonstrating the system live | Ask a question, watch the agent chain, see the verdict, browse training curves |

### 2.2 Non-goals

Explicitly out of scope. Listing these protects the timeline.

- Any clinical validation, IRB process, or real patient data.
- Differential privacy, secure aggregation, or homomorphic encryption. *(We will note in the report that FedAvg alone is not a formal privacy guarantee — this is honest scoping, not an oversight.)*
- More than four agent components. No planner, no memory, no tool-use agent, no LangGraph/CrewAI/AutoGen.
- Real network deployment across machines. Federation is **simulated** on one machine.
- Training a model from scratch, or full-parameter fine-tuning.
- Multilingual support. English only.
- Mobile app, authentication, multi-user accounts, or a production database.

### 2.3 Explicit non-requirement: fabricated results

Every number in the final report must be traceable to a file in `results/` produced by a script in `experiments/`. If an experiment does not fit the compute budget, we report it as **not run** rather than estimating it. This is a hard rule.

---

## 3. Users

| User | What they need from FedSLM |
|---|---|
| **The project team (primary)** | Reproducible experiment scripts, logged metrics, fast iteration on a 16 GB laptop |
| **FYP supervisor / evaluation panel** | A live demo that makes the architecture legible in 5 minutes, plus a defensible results table |
| **A future researcher reading the repo** | Clear setup, honest limitations, scripts that run unchanged |

There is no clinician user. A clinician-facing product would require validation work we are explicitly not doing.

---

## 4. System architecture

### 4.1 Training-time (federated)

```
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    HOSPITAL A       │   │    HOSPITAL B       │   │    HOSPITAL C       │
│  general medicine   │   │  pharmacology       │   │  surgery/pediatrics │
│                     │   │                     │   │                     │
│  local shard        │   │  local shard        │   │  local shard        │
│  (never leaves)     │   │  (never leaves)     │   │  (never leaves)     │
│        ↓            │   │        ↓            │   │        ↓            │
│  Qwen2.5-0.5B       │   │  Qwen2.5-0.5B       │   │  Qwen2.5-0.5B       │
│  + LoRA adapter     │   │  + LoRA adapter     │   │  + LoRA adapter     │
│        ↓            │   │        ↓            │   │        ↓            │
│  local training     │   │  local training     │   │  local training     │
└──────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘
           │                         │                         │
           │  LoRA weights only (~8 MB) ── raw data NEVER crosses this line
           │                         │                         │
           └────────────┬────────────┴────────────┬────────────┘
                        ↓                         ↓
              ┌──────────────────────────────────────────┐
              │        FLOWER FEDERATED SERVER           │
              │   FedAvg  (weighted by client samples)   │
              │   FedProx (secondary, if compute allows) │
              └────────────────────┬─────────────────────┘
                                   ↓
                       ┌───────────────────────┐
                       │   GLOBAL LoRA ADAPTER │  → broadcast back for round N+1
                       └───────────────────────┘
```

**The key design decision:** we federate the **LoRA adapter only**, not the base model. The base model is identical and frozen on every client, so it never needs transmitting. This turns communication cost from ~1 GB per client per round into ~8 MB — a 100×+ reduction, and one of the results we will measure and report.

### 4.2 Inference-time (multi-agent)

```
                        USER QUERY
                            │
                            ▼
                  ┌───────────────────┐
                  │   ORCHESTRATOR    │  plain Python — sequential, no framework
                  │  (state + trace)  │
                  └─────────┬─────────┘
                            │
     ┌──────────────────────┼──────────────────────┐
     │                      │                      │
     ▼                      ▼                      ▼
┌──────────┐  draft   ┌──────────┐  passages ┌──────────┐  flags
│ CLINICAL │ ───────► │ EVIDENCE │ ────────► │  SAFETY  │ ──────►
│  AGENT   │          │  AGENT   │           │  AGENT   │
│          │          │          │           │          │
│ global   │          │ FAISS +  │           │ rules +  │
│ SLM      │          │ MiniLM   │           │ SLM      │
│ generates│          │ retrieval│           │ check    │
│ a draft  │          │ over     │           │ for      │
│ answer   │          │ PubMedQA │           │ harm     │
└──────────┘          └──────────┘           └──────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │     VERIFIER      │
                  │                   │
                  │ grounding score   │
                  │ safety flags      │
                  │ answer confidence │
                  │        ↓          │
                  │  PASS / REVIEW /  │
                  │       FAIL        │
                  └─────────┬─────────┘
                            │
                            ▼
              FINAL RESPONSE + verdict + citations + trace
```

Four components. No more. The orchestrator is a `for` loop over a list of agents that carries a shared `AgentState` dataclass — it is plumbing, not a fifth agent.

---

## 5. Data requirements

### 5.1 The three datasets

All public, all downloadable from Hugging Face Hub, all free.

#### Dataset 1 — MedDialog (English)
- **What it is:** doctor–patient conversation pairs.
- **Why we need it:** teaches the model *how a clinician talks* — the conversational register. Without it a 0.5B model answers like a textbook index.
- **Feeds:** Clinical Agent fine-tuning.
- **Scale we will use:** ~30k–50k dialogue turns after cleaning (the full set is far larger than our compute allows).
- **Risk:** licensing is research-use, and the HF mirrors vary in quality. **Action: inspect and confirm license before use; if unavailable, substitute HealthCareMagic-100k from the ChatDoctor release.**

#### Dataset 2 — PubMedQA
- **What it is:** biomedical research questions with an abstract as context and a yes/no/maybe label. `pqa_labeled` = 1k expert-annotated, `pqa_artificial` = 211k generated.
- **Why we need it:** it is the **only** one of the three that ships a *context passage* with each question. That makes it the ground truth for evidence grounding — we can measure whether an answer is actually supported.
- **Feeds:** Evidence Agent retrieval corpus (FAISS index over the contexts) **and** the grounding metric.
- **Scale:** all 1k labeled for test; a 20k–50k slice of artificial for training and the retrieval index.
- **License:** MIT. Clean.

#### Dataset 3 — recommendation: **MedMCQA**

We will inspect all three candidates first, but the recommendation and its reasoning:

| Candidate | Size | License | Verdict |
|---|---|---|---|
| **MedMCQA** ✅ | 194k questions, 4 options, 21 subject labels | MIT | **Recommended** |
| MedQA (USMLE) | 12.7k | MIT | Rejected — too small and too hard |
| MedQuAD | 47k | Contested — NIH subsets removed | Rejected — licensing risk |

**Why MedMCQA wins, in three points:**

1. **It gives objective metrics for free.** Four options, one correct. Accuracy, precision, recall, and F1 fall straight out — no LLM-as-judge, no human annotation, no ambiguity. For a project that must not fabricate results, a cleanly scorable dataset is worth more than a more "interesting" one.
2. **Its subject labels are a natural non-IID key.** 21 medical subjects (pharmacology, anatomy, surgery, pediatrics…) means we can build realistic hospital specialisation instead of an artificial random skew. Hospital B genuinely becomes the pharmacology hospital.
3. **A 0.5B model can move on it.** MedQA is USMLE-level; a 0.5B model scores near the 25% chance floor, so *every system looks identical* and the whole comparison collapses. MedMCQA is easier, so improvements from federation and multi-agent verification are actually visible above noise. This is the deciding factor.

**Feeds:** the headline Accuracy / F1 comparison across all four systems, and the non-IID partitioning scheme.

### 5.2 Simulated hospitals and non-IID partitioning

Three hospitals, each with a different data distribution — because in reality a children's hospital and a cardiac centre do not see the same cases.

| Hospital | Specialisation skew | Rough composition |
|---|---|---|
| **A** | General medicine / primary care | MedDialog-heavy, general MedMCQA subjects |
| **B** | Pharmacology & therapeutics | PubMedQA-heavy, pharmacology/biochemistry MedMCQA |
| **C** | Surgery & pediatrics | Surgery/anatomy/pediatrics MedMCQA, some dialogue |

**Partitioning method:** Dirichlet distribution over MedMCQA subject labels with concentration parameter **α = 0.3** (strong skew). We also generate an **IID control split** (α = 100, effectively uniform) so we can run the IID vs non-IID comparison.

Quantity skew is also applied — hospitals should not have equal data volumes, because real ones do not.

**Requirement:** the partitioning script must be seeded and deterministic. Re-running it must reproduce identical shards.

### 5.3 Data pipeline requirements

Five scripts, run in order:

| # | Script | Output |
|---|---|---|
| 1 | `download.py` | Raw datasets cached in `data/raw/` |
| 2 | `clean.py` | Deduplicated, length-filtered, PII-scrubbed records |
| 3 | `preprocess.py` | Unified instruction-format JSONL in `data/processed/` |
| 4 | `partition.py` | `data/partitions/hospital_{a,b,c}/` for both IID and non-IID |
| 5 | `stats.py` | `results/data_stats.json` + charts — per-hospital size, subject histogram, label balance, token-length distribution |

**Hard constraint:** a single shared held-out test set, built before partitioning and never given to any client. All four systems are evaluated on exactly this set.

---

## 6. Model requirements

### 6.1 Primary SLM: **Qwen2.5-0.5B-Instruct**

| Criterion | Assessment |
|---|---|
| Size | 0.5B params — ~1 GB in fp16, fits free-tier Colab T4 three times over |
| License | **Apache 2.0** — fully permissive, no gatekeeping, no request form |
| Instruction following | Already instruction-tuned; usable zero-shot on day one, which unblocks the agent pipeline before fine-tuning finishes |
| Healthcare suitability | No medical pretraining, but strong general reasoning for its size — and a domain gap is *desirable* here, since it leaves headroom for fine-tuning to show measurable gains |
| LoRA/quantization support | First-class in PEFT and bitsandbytes |
| Federated fit | Small enough that 3 clients can train sequentially within one Colab session |

**Scale-up path:** if training proves faster than expected, repeat the headline experiment with **Qwen2.5-1.5B-Instruct**. Same code, one config change.

**Alternatives considered and why they lost:**

- **TinyLlama-1.1B** — 2× the parameters for weaker benchmark performance; older architecture.
- **SmolLM2-360M** — attractively small, but too weak on multiple-choice medical QA; risks the near-chance-floor problem that disqualified MedQA.
- **Phi-3-mini (3.8B)** — the best quality of the group, but 3 federated clients × multiple rounds does not fit a free T4 session. Compute, not quality, rules it out.

### 6.2 Training configuration

| Setting | Value | Reason |
|---|---|---|
| Method | **QLoRA** — 4-bit NF4 base + LoRA adapters | Fits free-tier VRAM; adapters are what we federate |
| LoRA rank / alpha | r=16, α=32, dropout=0.05 | Standard, well-tested starting point |
| Target modules | q/k/v/o + gate/up/down projections | Full attention + MLP coverage |
| Trainable params | ~1–2% of total | This is the number that makes communication cost small |
| Precision | bf16 compute | T4 supports it; halves memory vs fp32 |
| Max sequence length | 512 tokens | Covers PubMedQA contexts; keeps memory bounded |

---

## 7. Multi-agent requirements

### 7.1 Clinical Agent

- **Input:** user query.
- **Does:** classifies the query type (multiple-choice / yes-no-maybe / open dialogue), then generates a preliminary answer using the fine-tuned SLM.
- **Output:** `draft_answer`, `query_type`, `token_logprobs` (used later as a confidence signal).
- **Requirement:** must emit token-level confidence, not just text. The verifier needs it.

### 7.2 Evidence Agent

- **Input:** query + draft answer.
- **Does:** embeds the query with `all-MiniLM-L6-v2`, retrieves top-k=5 passages from a FAISS index built over PubMedQA contexts, and scores how much of the draft answer those passages actually support.
- **Output:** `retrieved_passages[]` with similarity scores, `grounding_score` ∈ [0,1], `unsupported_claims[]`.
- **Requirement:** runs entirely on CPU. This agent must work on the laptop with no GPU.

### 7.3 Safety Agent

- **Input:** query + draft answer + retrieved evidence.
- **Does:** a two-layer check —
  - **Layer 1 (rules, deterministic):** regex/keyword screens for specific dosage instructions, definitive diagnosis language, emergency red-flag symptoms without an escalation instruction, and contraindication mentions.
  - **Layer 2 (model):** the SLM is prompted to rate harm potential on a fixed rubric.
- **Output:** `safety_flags[]`, `risk_level` ∈ {low, medium, high}, `required_disclaimers[]`.
- **Requirement:** the rule layer must be able to override the model layer upward. A model saying "safe" cannot clear a hard red flag.

### 7.4 Verifier

- **Input:** everything above.
- **Does:** combines grounding score, safety risk level, and answer confidence into a single reliability score, then applies thresholds.
- **Output:** one of three verdicts, plus the reason:

| Verdict | Condition | User-facing behaviour |
|---|---|---|
| **PASS** | Well grounded, low risk, confident | Answer shown with citations |
| **REVIEW** | Partially grounded *or* medium risk *or* low confidence | Answer shown with a prominent caveat and the specific unsupported claims marked |
| **FAIL** | Poorly grounded *or* high risk | Answer withheld; the reason and an escalation message are shown instead |

- **Requirement:** thresholds live in one config file and are **calibrated on a validation split**, not hand-picked to flatter the results. The calibration procedure goes in the report.

---

## 8. Federated learning requirements

| # | Requirement |
|---|---|
| FL1 | Flower (`flwr`) in simulation mode; 3 `NumPyClient` instances, one per hospital |
| FL2 | Only LoRA adapter tensors are serialised into the parameter exchange — asserted in a unit test |
| FL3 | **FedAvg** weighted by client sample count (primary strategy) |
| FL4 | **FedProx** with a proximal term µ (secondary; run if compute allows — non-IID is exactly where it should help) |
| FL5 | ≥5 rounds, target 10; each round = local epochs on every client, then aggregation |
| FL6 | Per-round logging to CSV: round, per-client local loss, global eval loss, global accuracy, wall-clock seconds, MB transmitted |
| FL7 | A client must never be able to read another client's shard — enforced by directory structure and asserted in a test |
| FL8 | Checkpoint the global adapter every round so training can resume after a Colab session drop |

**Communication cost formula** (reported, not estimated):

```
MB_per_round = adapter_size_MB × num_clients × 2      (up + down)
MB_total     = MB_per_round × num_rounds
```

We measure `adapter_size_MB` from the actual serialised tensors.

---

## 9. Reliability and evaluation requirements

### 9.1 The four systems under comparison

| # | System | Federated? | Multi-agent? | Verifier? | What it isolates |
|---|---|---|---|---|---|
| 1 | **Single SLM** | ✗ | ✗ | ✗ | The floor |
| 2 | **Multi-Agent SLM** | ✗ | ✓ | ✓ | Gain from agents alone |
| 3 | **Federated SLM** | ✓ | ✗ | ✗ | Gain from federation alone |
| 4 | **FedSLM** | ✓ | ✓ | ✓ | Combined — does it compound? |

This is a clean 2×2 ablation. Systems 2 and 3 exist specifically so we can attribute any gain in system 4 to the right cause.

**Fairness rules:**
- Identical held-out test set for all four.
- System 1 and 3 differ *only* in how the adapter was trained (centralised on the pooled shards vs federated). Same total data, same epochs.
- Same decoding parameters, same seed, same prompt template everywhere.

### 9.2 Metrics

| Metric | Measured on | Method |
|---|---|---|
| Accuracy | MedMCQA, PubMedQA | Exact match on option letter / label |
| Precision, Recall, Macro-F1 | MedMCQA, PubMedQA | scikit-learn, macro-averaged |
| Evidence grounding | All | Fraction of answer claims entailed by retrieved passages |
| Hallucination rate | All | Fraction of claims with no supporting passage |
| Safety pass rate | Curated unsafe-query probe set | Fraction correctly flagged |
| Reliability score | All | Weighted composite of the four above |
| Latency (p50 / p95) | All | End-to-end wall-clock per query |
| Communication cost | Systems 3, 4 | MB transferred, measured |

### 9.3 Experiments to run

| ID | Experiment | Priority |
|---|---|---|
| E1 | Four-system comparison on the shared test set | **Must** |
| E2 | Federated round-by-round convergence curves | **Must** |
| E3 | IID vs non-IID under FedAvg | **Must** |
| E4 | Per-agent ablation — drop each agent, measure the delta | Should |
| E5 | FedAvg vs FedProx under non-IID | Should |
| E6 | Verifier threshold sensitivity | Should |
| E7 | Qwen2.5-1.5B repeat of E1 | Nice-to-have |

E1–E3 are the report's backbone. E4–E6 strengthen it. E7 only if time and Colab quota allow.

---

## 10. Interface requirements

A Streamlit dashboard, five pages. Full visual specification in `03_UI_LAYOUT.md`.

| Page | Job |
|---|---|
| **Ask** | Live demo — submit a query, watch the four agents run, see the verdict and citations |
| **Hospitals & Data** | Prove the non-IID split is real; per-hospital distribution charts |
| **Federated Training** | Round-by-round curves, communication cost, the data-boundary visual |
| **Results** | The four-system comparison table and charts |
| **Runs** | Browse saved traces and raw experiment artifacts |

Non-functional: runs locally via `streamlit run`, no auth, no database — reads from `results/` on disk.

---

## 11. Build order

Nine phases, strictly sequential. Each phase ends with something that **runs**.

| # | Phase | Deliverable | Est. |
|---|---|---|---|
| 1 | Project setup | Python 3.11 env, repo skeleton, config system, Colab notebook template | 1 day |
| 2 | Dataset preparation | 3 datasets downloaded, inspected, cleaned, unified. Third dataset confirmed. | 3 days |
| 3 | Partitioning | IID + non-IID shards, statistics report with charts | 2 days |
| 4 | Single SLM baseline | Fine-tuned adapter + first real accuracy number | 4 days |
| 5 | Multi-agent system | Four components + orchestrator, running on the baseline model | 5 days |
| 6 | Federated learning | Flower simulation, FedAvg, global adapter, per-round logs | 6 days |
| 7 | Verifier + reliability | Calibrated thresholds, reliability scoring, PASS/REVIEW/FAIL | 3 days |
| 8 | Experiments | E1–E3 mandatory, E4–E6 if time | 5 days |
| 9 | Final evaluation + UI | Streamlit dashboard, results write-up | 4 days |

**~33 working days.** Phases 4 and 6 are the risk concentration — both depend on Colab availability.

Per the agreed working method, each phase follows: explain what → explain why → create files → implement → run → fix → document how to run → **then** move on. No phase is skipped ahead of.

---

## 12. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Colab free tier disconnects mid-federated-run | **High** | High | Checkpoint the global adapter every round; resume from round N. Fall back to Kaggle (30h/week, more stable sessions). |
| MedDialog licensing or mirror quality is unusable | Medium | Medium | Substitute HealthCareMagic-100k; decision made in Phase 2 before anything depends on it. |
| 0.5B model too weak for gains to clear noise | Medium | **High** | MedMCQA chosen partly to avoid this. If it still happens: report confidence intervals, escalate to Qwen2.5-1.5B. |
| Multi-agent latency makes the demo feel broken | Medium | Low | Cache retrieval; stream agent-by-agent so the UI fills progressively instead of hanging. |
| Federated result is *worse* than centralised | Medium | Low | This is a legitimate, publishable finding under non-IID skew. Report it and analyse why. Do not hide it. |
| Python 3.14 on the dev laptop blocks PyTorch | **Confirmed** | High | Install Python 3.11 in a separate venv. Detailed in the tech stack doc. |
| Scope creep into more agents / frameworks | Medium | High | Four components is a hard cap, stated in the non-goals. |

---

## 13. Definition of done

The project is complete when all of the following are true:

- [ ] `python -m scripts.setup_check` passes on a clean Python 3.11 environment
- [ ] All three datasets download, clean, and partition from scratch with one command
- [ ] `results/data_stats.json` and its charts exist and show a genuine non-IID skew
- [ ] A fine-tuned single-SLM checkpoint exists with a recorded test accuracy
- [ ] A query through the multi-agent pipeline returns an answer, a verdict, citations, and a full trace
- [ ] Federated training completes ≥5 rounds with a per-round CSV log
- [ ] `results/comparison.csv` holds the four-system table, every cell traceable to a run artifact
- [ ] IID vs non-IID results are recorded
- [ ] The Streamlit dashboard runs and demonstrates all of the above
- [ ] `README.md` documents every command needed to reproduce every number
- [ ] No fabricated results anywhere. Unrun experiments are marked "not run."

---

## 14. Open decisions

Resolved by default unless you say otherwise:

| # | Decision | Default taken | Flip it by saying |
|---|---|---|---|
| 1 | Training compute | Google Colab free (T4) | "use Kaggle" / "local CPU only" |
| 2 | UI framework | Streamlit | "use FastAPI + React" / "use Gradio" |
| 3 | Third dataset | MedMCQA | "use MedQA" / "use MedQuAD" |
| 4 | Primary SLM | Qwen2.5-0.5B-Instruct | name another |
| 5 | Experiment tracking | CSV + JSON on disk (no account needed) | "use Weights & Biases" |

---

**Next document:** `02_TECH_STACK.md` — every library, every version, and exactly what you must install.
