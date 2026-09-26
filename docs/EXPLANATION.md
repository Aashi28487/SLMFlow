# SLMFlow — Plain-Language Interview Guide

> This file explains what the project is and how it works so you can confidently
> describe it to an interviewer without using confusing jargon.

---

## 1. What is this project in one sentence?

**SLMFlow is a framework that intelligently decides *which small AI model* to run
for a given task, and *when* to load or swap it, so that a limited-GPU laptop can
handle complex AI pipelines efficiently.**

---

## 2. The Problem It Solves

Modern AI assistants use Large Language Models (LLMs), which are huge and need a
lot of memory. Running them on a laptop with only 6 GB of GPU memory is hard.

Instead of using one big model, SLMFlow uses **several small models (SLMs)**
together. The challenge is:

- Each model takes time to load into GPU memory (called a **cold load**).
- If you swap models too often, you waste time on loading instead of working.
- Different models are better at different tasks (classification vs extraction vs QA).

SLMFlow solves this by:
1. **Tracking how much GPU/RAM memory is available** at all times.
2. **Picking the right model for each task** using a learned router.
3. **Scheduling which models to load/unload** to avoid unnecessary swaps.

---

## 3. Key Components (What I Built / Fixed)

### A. The Resource Manager (C1 — Core Contribution)
- Tracks two memory tiers: **GPU (VRAM)** and **CPU RAM**.
- Enforces a two-tier budget so models don't exceed available memory.
- Implements **swap-aware scheduling** — if two consecutive tasks can use the
  same model, it avoids unloading and reloading it (saves ~30–40 seconds per swap).

### B. The Model Registry
- A central store of **model cards** — data sheets describing each AI model:
  cold load time, warm load time, tokens per second, accuracy per task type,
  memory footprint.
- Model cards are measured from real hardware and saved to
  benchmarks/results/model_cards.json.

### C. The Ollama Backend
- Ollama is an open-source tool that runs AI models locally.
- I implemented an **Ollama backend** that sends HTTP requests to Ollama's local
  API (http://localhost:11434) to generate text, measure load times, and track
  memory usage.
- This means SLMFlow can use any Ollama-compatible model without changing code.

### D. The Router (C2 — Learned Routing)
- A lightweight **logistic regression classifier** that, given a task description,
  predicts which model is most likely to answer it correctly.
- It is trained using real inference results from both models on the benchmark set.
- It beats a random router and a naive ''always use the cheapest'' baseline.

### E. The Evaluation Harness
- Runs all configurations across three task families with three random seeds.
- Writes results to docs/RESULTS.md.
- Runs statistical significance tests (McNemar, paired bootstrap).

---

## 4. The Two Models I Used

| Model | Size | VRAM | Cold Load | Tokens/s | Accuracy |
|---|---|---|---|---|---|
| qwen2.5:0.5b | 0.5B params | 0.48 GB | 38.7 s | 1.32 | 80% |
| llama3.2:3b  | 3B params   | 2.55 GB | 16.2 s | 1.33 | 20% |

Interesting finding: The bigger model (llama3.2:3b) loaded faster (16s vs 38s)
because it is more optimised for the Ollama runtime. But it scored much lower
on the classification task. This is a real empirical finding.

---

## 5. What Regret Means

In scheduling research, regret = how much worse your scheduler performed
compared to a perfect oracle that knew the future.

- Greedy scheduler: decides one task at a time, trying to minimise swaps.
- Oracle: knows all tasks in advance and picks the perfect load order.
- Naive LRU: evicts the least-recently-used model (standard cache policy).

Lower regret = closer to optimal.

---

## 6. What Router Means

The router decides, for each incoming question, which model to send it to.
SLMFlow's router is trained on real data: it runs every model on every task
item and records which model got the right answer. Then it trains a classifier
to predict which model to pick next time without actually running both models.

---

## 7. The Three Task Families

| Family | What it tests |
|---|---|
| repurposing_label | Classify a drug-indication pair as approved / investigational / hypothesis |
| grounded_qa | Answer a question about a drug, grounded in retrieved documents |
| trial_lookup | Extract structured data from a clinical trial description |

---

## 8. Things That Were Missing and I Fixed

| Missing piece | What I did |
|---|---|
| OllamaBackend class | Implemented it using Python urllib to call Ollama REST API |
| get_backend() factory | Created it so the system can look up backends by name |
| load_cards() / save_cards() | Implemented JSON persistence for model cards |
| ModelRegistry.cheapest() | Added method to find the cheapest model by VRAM |
| ModelRegistry.digests() | Added method to return model digests for provenance |
| Backend registration in harness | Added auto-registration of Ollama backend on startup |

---

## 9. Key Insight for the Interviewer

The system treats model loading as a scheduling problem, not just an inference
problem. Most frameworks assume the model is already in memory. SLMFlow explicitly
accounts for the cost of loading a model, and makes routing decisions that minimize
that cost — which matters a lot on memory-constrained hardware like a 6 GB GPU laptop.

---

Last updated: September 2026 — measured on Intel i5 HX + RTX 4050 Laptop GPU (6 GB VRAM) + 17 GB RAM
