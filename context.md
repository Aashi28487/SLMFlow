# Project Context — SLM Orchestration Framework

**Working title:** SLMFlow — a multi-model orchestration framework/library for reliable Small Language Models
**Status:** Pre-implementation context document (direction, comparison, research goals)
**Date:** 23 August 2026

---

## 1. Project Idea

We are building a **general-purpose orchestration framework/library for Small Language Models (SLMs)**.

In spirit it is similar to LangChain/LangGraph, but those frameworks were designed around large cloud LLMs reached through an API. Ours is designed around **small models running on the user's own hardware** (roughly 0.5B–4B parameters, quantised).

Core requirements:

* Multiple SLMs cooperate through orchestration, not through one big model.
* Runs locally on a normal PC; some models should be small enough for mobile/edge.
* Works under limited hardware — no assumption of an expensive NVIDIA GPU or a cloud API key.
* Orchestration must be **reliable**, not just wired together.

The framework is the deliverable. Pharmaceutical/healthcare is the **test environment**, not the product.

---

## 2. Problem Statement

Cloud LLMs are powerful, but some organisations cannot send their data to them. A pharmaceutical company working with private drug research, internal knowledge bases, confidential documents, and unpublished scientific data may be unable — legally or by policy — to send that text to ChatGPT or any external API.

SLMs are a practical alternative: small enough to run on-premise, cheap enough to run continuously, and good enough for narrow, repeated tasks.

But **having several SLMs is not a system**. To make them usable you still need to:

| Need | Why it is hard with SLMs |
|---|---|
| Select the right model | Each small model is good at only a narrow slice |
| Route each sub-task | Wrong routing wastes the specialisation entirely |
| Coordinate models | Small models follow multi-step instructions poorly |
| Combine outputs | Models disagree, and the confident one is not always right |
| Handle failure | Malformed output is common at this size |
| Judge confidence | A single SLM hallucinates more than a large LLM |
| Fit the hardware | Models cannot all be resident in memory at once |

Today a developer does all of this by hand. That manual layer is what we want to turn into a framework.

---

## 3. Existing Work

### 3.1 SLM-Mux (primary reference)

**SLM-MUX: Orchestrating Small Language Models for Reasoning** — Wang, Wan et al. (Harvard, Georgia Tech, Stanford).

**What it is.** A training-free architecture for combining several SLMs on reasoning questions.

**Problem it solves.** The paper shows that orchestration methods built for frontier LLMs — Mixture-of-Agents, LLM-Debate, Multi-Agent Verification — *help* GPT-4o-class models but *hurt* SLMs, dropping accuracy by up to ~5.5%. Small models fall into **groupthink**: instead of correcting each other, they reinforce each other's errors (the authors attribute 59.5% of failures to this).

**Main idea.** Do not let the models talk to each other at all.

1. **Independent generation** — every SLM answers the same question *k* times at temperature > 0.
2. **Confidence estimation** — each model's confidence = how often it repeats its own most common answer (self-consistency). The answer from the most confident model wins; ties are broken by that model's accuracy on a validation set.

Two optimisations on top: a **model selection search** that picks complementary models by maximising `UnionAccuracy − λ·ContradictionPenalty` on a validation set, and **test-time scaling** (more models, or more samples per model).

**Results.** Up to +13.4% (MATH), +8.8% (GPQA), +7.0% (GSM8K) over existing orchestration methods; two SLMs beat Qwen-2.5 72B on GPQA and GSM8K and match it on MATH.

**Limitations stated by the authors:**
* Static — the same fixed model group is used for every query, with no per-question adaptation.
* The model selection search is exhaustive, so it is slow and costly with a large model pool.
* The tie-break rule (validation accuracy) is fixed and non-adaptive.
* Confidence is *only* self-consistency — a model can be consistently and confidently wrong.
* Majority voting restricts it to discrete answer spaces; open-ended generation is only sketched (via judge/embedding similarity).

**Not addressed in the paper — to investigate, not to assume as criticism:** local/edge deployment, memory budgets and model residency, model-swap cost, and end-to-end latency on consumer hardware. The paper reports token usage, not wall-clock or RAM on an integrated-GPU machine. We should verify this ourselves rather than claim it as a weakness.

**Relevant to us.** Confidence-based selection instead of debate; complementarity as the criterion for choosing models; contradiction penalty as a reliability signal. We can adopt SLM-Mux as **one strategy inside our framework** and as a **baseline** to compare against.

### 3.2 Little Coder video

A YouTube walkthrough by Little Coder covering this area. **Unverified — to be watched and cross-checked against the paper.** Treat as secondary/explanatory, not as a citable source, until reviewed.

### 3.3 Comparison

| Aspect | LangChain | LangGraph | SLM-Mux | Our framework |
|---|---|---|---|---|
| Target models | Any LLM, cloud-first | Any LLM, cloud-first | SLMs (7B–24B, served) | Small local SLMs (~0.5B–4B, quantised) |
| Local execution | Supported via integrations | Supported via integrations | Possible, not the focus | First-class design goal |
| Mobile/edge | Not a goal | Not a goal | Not addressed | Explicit goal (feasibility to investigate) |
| Multi-model orchestration | Chains/tools | Graphs, state machines | Fixed parallel ensemble | Graph + ensemble + routing |
| Routing | Manual / prompt-based | Manual, developer-defined | None per query (static set) | Per-task routing on capability + resources |
| Reliability | Left to developer | Left to developer | Self-consistency + tie-break | Consistency + verification + abstention + fallback |
| Hardware requirements | Not modelled | Not modelled | Not modelled | Modelled as a scheduling constraint |
| Privacy | Depends on provider | Depends on provider | Depends on serving | Local-by-default |
| Agent support | Yes | Yes, strong | No agent abstraction | Yes, lightweight SLM-sized agents |
| Benchmarking | External | External | Accuracy + token usage | Built-in quality **and** cost telemetry |

**Honest note:** LangChain/LangGraph *can* run local models. The difference is not capability but **design centre** — they do not treat RAM, model residency, swap cost, or latency as things the orchestrator must reason about. That gap is our opening.

---

## 4. What We Want to Build

A **general-purpose SLM orchestration library**, organised into layers rather than a long feature list:

| Layer | Responsibility | Research status |
|---|---|---|
| **Model registry** | Declare available SLMs with a capability profile (task types, size, quantisation, measured speed) | Mostly engineering |
| **Task analysis** | Turn a user task into typed sub-tasks | Needs research — can an SLM decompose reliably? |
| **Router** | Pick a model per sub-task from capability + live resource state | **Core contribution — needs research** |
| **Agent layer** | Small, single-purpose agents (extract, summarise, verify) with schema-validated I/O | Engineering + prompt design |
| **Orchestration engine** | Sequential / parallel / conditional / ensemble execution over shared context | Engineering |
| **Reliability layer** | Confidence scoring, cross-checking, verification, abstention, fallback models | **Core contribution — needs research** |
| **Resource manager** | Memory budget, model load/unload, swap-cost accounting | **Core contribution — needs research** |
| **Local/edge runtime** | Quantised CPU inference; mobile feasibility | To investigate — do not assume it works |
| **Telemetry & benchmarking** | Record quality and cost on every run | Engineering |

Not every item from the brainstorm becomes its own feature — *task decomposition*, *model selection*, and *fallback* are behaviours **inside** the routing and reliability layers, not separate modules.

---

## 5. Execution Flow (conceptual)

```text
User Task
   |
Task Analysis            -> split into typed sub-tasks
   |
Model / Agent Selection  -> router picks an SLM per sub-task
   |                        (capability profile + live memory/latency budget)
SLM Execution            -> agents run; orchestration engine controls the pattern
   |                        (sequential / parallel / ensemble)
Validation & Evaluation  -> confidence, cross-check, grounding; abstain if weak
   |                        on failure: fallback model or re-route
Final Response           -> answer + confidence + trace
```

Where each concept lives:

* **Agents** — the units of work at "SLM Execution".
* **Routing** — decides *which* model, using capability and resource state.
* **Orchestration** — decides *in what order and shape* agents run.
* **Evaluation** — after execution; may reject the answer.
* **Fallback** — triggered by evaluation failure or by a resource failure.
* **Reliability** — the combination of evaluation + fallback + abstention.
* **Resource awareness** — a constraint that touches routing, orchestration, and fallback simultaneously.

---

## 6. Hardware Constraint

The baseline development and evaluation machine is:

* Intel i5 **HX-series** CPU (high-power laptop class)
* Discrete GPU with **~6 GB VRAM** — CUDA offload is available
* ~16 GB system RAM, shared between OS and models
* 512 GB SSD, shared with the corpus in `data/`
* No GPU beyond the above, no cloud LLM API

> **Correction, 2026-09-10.** An earlier version of this section described the machine as having
> integrated Intel Iris graphics with *no CUDA and no usable VRAM*. **That was wrong.** The actual
> evaluation machine has a discrete GPU with ~6 GB VRAM. Exact CPU and GPU model numbers are
> `not yet probed` — Phase 0 must record them in `docs/ENVIRONMENT.md` by running code, not by
> assuming. This correction **strengthens** the research problem; see the two-tier note below.

Consequences to design around: only one or two models resident at a time, so **loading and unloading is part of the cost**; quantised inference (GGUF / int4-class) is the default; and latency will be seconds, not milliseconds.

**The memory budget is two-tiered, and that is an opportunity rather than a complication.** ~6 GB of
fast VRAM sits above ~16 GB of slower system RAM, which sits above the SSD. A model can be
GPU-resident, RAM-resident, or on disk only, and the cost of promoting it differs at every level. The
residency problem is therefore a genuine **two-level paging problem with variable page sizes and
variable miss costs** — a sharper instance of the constraint than a CPU-only machine would have
produced, not a softer one.

We do **not** assume everything runs well here. Part of the research is to establish where the ceiling is: which model sizes are usable, how long a swap costs, and at what point orchestration overhead stops being worth it. Any training that is needed happens elsewhere (e.g. a free Colab tier); the framework itself targets **inference on this machine**.

---

## 7. Pharmaceutical / Healthcare Demonstration

A single scenario used to show the framework works on a sensitive workload:

```text
Private pharmaceutical documents
   |
Local SLM agents  (retrieval -> extraction -> summarisation)
   |
Cross-check / verification agent  (claims vs. source evidence)
   |
Final answer with citations, or abstention
```

What it is meant to demonstrate: **privacy** (nothing leaves the machine), **local inference**, **multi-agent collaboration**, **reliability** (verified, cited, or refused), **model specialisation**, and **resource efficiency**.

This stays a demonstration. It is not a clinical tool, not a diagnosis system, and not a healthcare product.

---

## 8. Benchmarking Dimensions

No numeric targets yet — these are the axes we must be able to measure.

**Quality:** task accuracy; answer/summary quality; groundedness (claims supported by cited evidence); hallucination or unsupported-claim rate; abstention correctness (did it refuse when it should have?).

**Cost:** end-to-end latency; peak RAM; CPU utilisation; model load/swap count and time; tokens generated; number of model calls. Energy only if a practical measurement method exists — otherwise skip it rather than fake it.

**System behaviour:** routing accuracy (did the router pick the model that would have been best?); routing overhead; failure and recovery rate; output-schema validity rate.

**Comparisons:** vs. each single SLM alone; vs. one larger model of comparable total memory footprint; vs. SLM-Mux as an orchestration baseline; vs. a LangGraph-style static pipeline; and the **quality-vs-cost break-even point** — where adding models stops paying for itself.

---

## 9. Expected Research Contribution

> A lightweight, hardware-aware, reliable orchestration framework for cooperating SLMs that runs locally or on edge/mobile hardware.

| Existing (we adopt / build on) | Ours (proposed contribution) |
|---|---|
| Multi-agent orchestration patterns (LangChain / LangGraph / AutoGen) | Orchestration treated as scheduling under a **fixed memory budget** |
| Confidence-based selection, complementary model choice (SLM-Mux) | **Per-query, resource-aware routing** instead of one static model set |
| Verification / critic agents | Verification + **abstention** as a built-in framework guarantee, not user code |
| Accuracy benchmarks | **Joint quality-cost telemetry**, and the empirical break-even point |
| Local inference runtimes (llama.cpp / Ollama class) | A reusable **library** on top, not a one-off application |

The honest framing: the individual ingredients exist. The contribution is combining them under a **hardware constraint that existing frameworks ignore**, plus the empirical answer to "when does orchestrating small models actually beat one bigger model on a laptop?"

---

## 10. Project Structure

```text
SLM Orchestration Framework
|
├── Model Layer          — registry, capability profiles, loaders, quantised runtime
├── Routing Layer        — task analysis, model selection, resource-aware routing
├── Agent Layer          — small typed agents, schema-validated I/O
├── Orchestration Layer  — execution graph, shared context, patterns
├── Reliability Layer    — confidence, verification, abstention, fallback
├── Resource Manager     — memory budget, load/unload, swap accounting  (cross-cutting)
├── Telemetry            — quality + cost recorded per run
├── Benchmarking         — baselines, ablations, break-even analysis
└── Pharma Demonstration — sample workflow on private documents
```

Resource Manager is listed as a layer but behaves as a **cross-cutting service** — routing, orchestration, and fallback all query it.

---

## Answers to the Ten Questions

1. **What are we building?** A Python library that orchestrates several small local language models into one reliable system.
2. **Why?** Because wiring SLMs together by hand is manual, fragile, and ignores hardware limits.
3. **Why SLMs?** Privacy (data stays on-premise), cost, and no dependence on a cloud API.
4. **Why orchestration?** One small model is not reliable enough alone; several specialised ones, checked against each other, can be.
5. **Different from LangChain/LangGraph?** They are model-agnostic and cloud-centred; we treat memory, model residency, and swap cost as first-class scheduling constraints.
6. **Different from SLM-Mux?** SLM-Mux is a fixed, static ensemble for discrete-answer reasoning. We are a general framework with per-query routing, agents, workflows, and resource awareness — and we use SLM-Mux as a strategy and a baseline inside it.
7. **Research gaps?** Resource-aware routing for local multi-model systems; reliability/abstention as a framework guarantee; joint quality-cost evaluation; the multi-SLM vs. single-model break-even point.
8. **Pharma use case?** It shows the framework handling confidential documents entirely locally, with verified and cited output.
9. **Benchmarking?** Quality and cost measured together, against single SLMs, a larger single model, and SLM-Mux.
10. **Research + product?** The measurements answer an open question (paper); the code is a reusable library others can install and run on ordinary hardware (product).

---

# Claude Suggestions

Additions worth considering. **Nothing here is decided** — these are proposals, and several are ways to make the project defensible rather than extra features.

### A. Scope conflict to resolve first

`docs/01_PRD.md` and `src/fedslm/` describe a **federated learning** direction (FedSLM — LoRA adapters, simulated hospitals, Flower). This document describes an **orchestration framework** direction. These are two different research contributions and both are large. **Decide which is the thesis** and demote the other to future work; carrying both will weaken the evaluation of each. The existing `data/repurposing/` pipeline (PubMed / openFDA / ClinicalTrials) is reusable either way.

### B. Harness and evaluation concepts not yet named

* **Evaluation harness** — a runner that executes a fixed task set against any configuration and emits one results table. Build this *early*; it is what makes every later claim reproducible.
* **Ablations** — same harness, one component removed at a time (no router / no verifier / no fallback). This is what turns "our system works" into "this component is why it works".
* **Determinism and seeding** — fixed seeds and pinned model versions, or the numbers will not reproduce.
* **Trace logging** — record every sub-task, model choice, and reason. Needed for debugging *and* for the paper's qualitative examples.
* **Golden / regression set** — a small frozen set of tasks with known-good outputs, to catch regressions as the router changes.

### C. Metrics that would strengthen the paper

* **Cost-normalised accuracy** — accuracy per second, and accuracy per GB of peak RAM. This is the metric that makes the "small beats big on a laptop" argument concrete.
* **Pareto frontier** — plot quality against latency/memory for every configuration, rather than reporting one winning number.
* **Router regret** — accuracy of our router versus an oracle router that always picks the best model in hindsight. A clean, publishable number.
* **Selective accuracy / risk-coverage curve** — accuracy on the questions the system chose to answer, plotted against how many it answered. The standard way to show abstention has value.
* **Calibration (ECE)** — does a stated confidence of 0.8 actually mean 80% correct? SLM-Mux's admitted weakness is exactly here, so measuring it is a natural contribution.
* **Contradiction rate** — SLM-Mux's own penalty term, reused as a live runtime signal instead of only a search objective.
* **Cold-start vs. warm latency** — report both; model load time will dominate on this hardware and hiding it would be dishonest.
* **Statistical treatment** — multiple seeds, mean ± std, and a significance test. Single-run deltas of 1–2% mean nothing.

### D. Technical ideas worth investigating

* **Cascade / escalation** — try the cheapest model first, escalate only on low confidence (FrugalGPT's idea, applied to local memory cost instead of API price). Likely the biggest practical win.
* **Selective verification** — verify only claims below a confidence threshold, not everything, to keep the verifier's cost down.
* **Keep the verifier tiny** — a 0.5B verifier checking a 3B generator is far cheaper than symmetric verification and may capture most of the benefit.
* **Model-swap-aware scheduling** — batch all sub-tasks assigned to the same model together so it loads once. Simple, and directly measurable as a cost reduction.
* **Structured output enforcement** — grammar-constrained decoding (GBNF / JSON schema) instead of hoping small models emit valid JSON. This alone removes a large class of failures.
* **Capability profiling as a first-class step** — a one-off calibration run measuring each model's accuracy per task type and its speed on *this* machine, feeding the router. This is what makes the routing hardware-aware in a real sense.

### E. Risks to name explicitly in the paper

Orchestration overhead may exceed its benefit on CPU (a legitimate finding — report it either way); self-consistency sampling multiplies cost by *k*, which is expensive without a GPU; mobile/edge may prove infeasible within the timeframe and should then be scoped down to "demonstrated on one small model" rather than dropped silently; and LLM-as-judge evaluation using a *small* judge is unreliable, so prefer objective metrics or a documented human spot-check.

### F. Sources to check before writing related work

Belcak et al. 2025 (NVIDIA, *Small Language Models are the Future of Agentic AI*) is the strongest motivation citation. RouteLLM and FrugalGPT are the closest routing baselines. MAST (Cemri et al. 2025) is the multi-agent failure taxonomy that justifies the verification layer. Check whether SLM-Mux has released code — reproducing it as a baseline is much cheaper than reimplementing from the paper.
