# Research survey and novelty check (Phase 1)

> **This phase's Definition of Done is NOT met.** The brief requires a web
> search to establish novelty. The implementation environment had no search
> access, so what follows is written from prior knowledge as of the assistant's
> training cutoff and **must be verified against the current literature before
> any claim here is defended.** Specific verification tasks are listed at the
> end.
>
> Ground Rule 1 applies to citations as much as to numbers: nothing below
> should be cited in the thesis until the paper has been located and read.
> Where a detail is uncertain it is marked *[verify]*.

---

## 1. The landscape, by design centre

The useful axis is not capability but **design centre** — what the framework
assumes is expensive.

| System | Design centre | Treats a model as | Optimises | Models residency? |
|---|---|---|---|---|
| LangChain | composition of LLM calls | a remote endpoint | developer ergonomics | no |
| LangGraph | stateful agent graphs | a remote endpoint | control flow, cycles | no |
| AutoGen | multi-agent conversation | a remote endpoint | agent dialogue | no |
| CrewAI | role-based agent teams | a remote endpoint | task delegation | no |
| DSPy | prompt/weight optimisation | a differentiable module | prompt quality | no |
| RouteLLM | query → model routing | a priced API call | **price per call** | no (switching is free) |
| FrugalGPT | LLM cascades | a priced API call | **price per call** | no (switching is free) |
| llama.cpp / Ollama / MLC | local inference *engines* | a file to load | throughput, memory | yes, but no orchestration |
| SLM-MUX | fixed SLM ensemble | a sampling process | accuracy per token | no |
| **SLMFlow** | **orchestration under a hard memory ceiling** | **a resident object with load cost** | **quality per second and per GB** | **yes, as the central concern** |

The pattern: the orchestration frameworks do not model memory; the inference
engines model memory but do not orchestrate. Nothing occupies the intersection.

**Honest qualification.** LangChain and LangGraph *can* run local models, and
both have community integrations for llama.cpp and Ollama. The difference is not
capability but that RAM, residency, swap cost and latency are not things their
execution model reasons about. `context.md` §4 already states this correctly and
it should stay stated that way — overclaiming here is the easiest way to lose a
reviewer.

---

## 2. Per-claim novelty argument

### C1 — the Memory-Bounded Orchestration Problem

**The claim.** Formalising task-to-model assignment plus ordering plus a
residency plan under a hard per-tier memory ceiling, and solving it with a
scheduler measured against an offline oracle.

**Genuinely unaddressed?** *Probably yes, in this exact form* — but the
component problems are all classical, and the survey must say so:

- **Model residency is page replacement** with variable page sizes and variable
  miss costs. Belady's optimal, LRU, and the size-aware/cost-aware caching
  literature (GreedyDual-Size and successors) all apply *[verify]*. Weighted
  caching with variable fetch cost is a solved problem in the abstract.
- **Task-to-model assignment** is an assignment problem; with dependencies it is
  a scheduling problem on unrelated machines, which is NP-hard in general.
- **ML inference serving** has substantial work on model placement and
  multi-tenancy — Clipper, INFaaS, Clockwork, AlpaServe, and vLLM-adjacent
  serving work *[verify each]*. Several of these *do* reason about GPU memory
  and model loading. **This is the closest prior work and the biggest threat to
  the claim.**

**Where the remaining delta plausibly is.** The serving literature optimises
*throughput or latency SLOs for independent requests on server hardware*. It
does not optimise *end-to-end answer quality for a dependent DAG of sub-tasks*
where the assignment choice changes the *quality* of the output, not just its
speed, under a *single-slot* ceiling.

**The sharpened claim, and the one to defend:** MBOP couples a quality-bearing
assignment with cost-bearing residency, under a ceiling tight enough that
single-slot is the normal case rather than the degenerate one. If a paper is
found that already does this, say so plainly, cite it, and narrow to whatever
remains — per the brief, do not defend a claim the literature has answered.

**Verification tasks.** Search: "memory-aware LLM orchestration", "model
swapping scheduler LLM", "AlpaServe model placement", "Clockwork model serving
predictability", "INFaaS model-less inference", "cost-aware caching variable
size", "LLM agent scheduling memory constraint". Read AlpaServe and Clockwork
properly — they are the nearest neighbours.

### C2 — portable capability profiles across hardware tiers

**The claim.** A device card × a model card is the routing table, and the
research question is whether a routing policy calibrated on one device transfers
to a weaker device after re-calibration, or must be relearned.

**Genuinely unaddressed?** *The framing looks novel; the components do not.*
Hardware-aware NAS and on-device benchmarking (MLPerf Tiny, and per-device
latency tables in NAS work) both build device-specific cost models *[verify]*.
What is unusual is asking about **policy transfer** rather than cost-model
transfer: whether the *decision rule* survives a hardware change.

**Risk.** This is the weakest of the four claims because it needs a second real
device to say anything. Without a phone or SBC it is a tier model and a
`DeviceCard` class, which is engineering, not a result.

**Verification tasks.** Search: "hardware-aware routing transfer", "on-device
LLM benchmark suite", "cross-device policy transfer inference".

### C3 — abstention as a declarative contract

**The claim.** `Pipeline.from_config("qa.yaml", guarantee=Precision(0.90))`
auto-tunes confidence thresholds on a calibration split to meet a declared
precision target, then reports achieved coverage.

**Genuinely unaddressed?** *The mechanism is established; exposing it as an
orchestration-framework API is what appears new.*

- **Selective prediction / reject option** is decades old (Chow 1970) *[verify]*.
- **Conformal prediction** and **Learn-then-Test** (Angelopoulos et al.) give
  exactly the distribution-free guarantee used here *[verify the LTT reference
  and whether the fixed-sequence variant is stated as used]*.
- **Selective prediction for LLMs** is an active area; there is work on
  calibration and selective QA *[verify]*.

**Where the delta is.** Not the statistics — those are borrowed and should be
cited as borrowed. The delta is that no orchestration framework *exposes a
reliability target as a declarative configuration value* and tunes itself to
meet it. LangChain has no such concept; nor does LangGraph or AutoGen. The
contribution is API-level plus the measurement (risk-coverage, ECE per signal,
coverage at target precision).

**This is likely the most defensible claim**, precisely because it is modest and
specific.

**Verification tasks.** Search: "Learn-then-Test risk control", "conformal
selective prediction LLM", "declarative reliability API LLM", "selective
question answering calibration".

### C4 — cost-adaptive self-consistency

**The claim.** Sequential sampling with a posterior-based stopping rule, plus
cascade escalation, cutting the `k × |M|` cost of SLM-MUX at matched accuracy.

**Genuinely unaddressed?** *Likely already addressed, at least in part.* Adaptive
and early-stopping self-consistency has been studied — there is work on
adaptive-consistency with a stopping criterion, and on early-stopping
self-consistency *[verify: "Adaptive-Consistency", Aggarwal et al. is the one to
find]*. Cascades are well covered by FrugalGPT.

**The honest position.** Treat C4 as an **engineering optimisation with a
measured result on this hardware**, not a novel method. Cite the prior adaptive
consistency work, and claim only: (a) the specific Beta-Binomial stopping rule,
(b) its interaction with *model residency*, which the cloud versions have no
reason to consider, and (c) the measured cost reduction on CPU/6 GB-VRAM
hardware where sample cost is dominated by different terms than in the cloud.

This is still worth doing: the brief calls it "the optimisation most likely to
produce a clean, quotable result", and a measured number on real constrained
hardware has value even when the method is known.

**Verification tasks.** Search: "adaptive consistency LLM sampling", "early
stopping self-consistency", "FrugalGPT cascade", "dynamic sample allocation
reasoning".

---

## 3. SLM-MUX — the primary baseline

Read `slm-mux.pdf` closely and rewrite this section **in your own words** from
the paper itself. What the implementation in `strategies/slm_mux.py` assumes:

- independent generation from each model at `T > 0`, `k` samples each;
- self-consistency (modal-answer share) as the confidence signal;
- validation accuracy as the tie-break between equally consistent models;
- model-set selection by maximising `UnionAcc − λ · Contradiction`.

**Its self-admitted weakness** is that confidence is self-consistency alone.
That is the opening C3 attacks directly, and it should be quoted from the paper
rather than paraphrased.

**What the paper does not address**, per `context.md` §3 — and this must be
framed as *to investigate*, not asserted as criticism: local/edge deployment,
memory budgets, model residency, swap cost, and wall-clock latency on consumer
hardware. The paper reports token usage. Verify this by reading it rather than
assuming.

**Reproduction expectation.** Phase 6's DoD is the *qualitative* pattern, not
the numbers: different models, different hardware. Log both side by side and
state explicitly if the pattern does not reproduce.

---

## 4. Also to survey properly

| Work | Why it matters here |
|---|---|
| Belcak et al. 2025, *Small Language Models are the Future of Agentic AI* | the framing this project's premise rests on; verify the argument and the citation |
| MAST (Cemri et al. 2025), multi-agent failure taxonomy | gives a vocabulary for the failure classes Phase 10 removes |
| llama.cpp GBNF grammars | the mechanism behind the schema guarantee |
| MLC-LLM | the plausible T0 backend if llama.cpp ARM proves awkward |
| Speculative decoding | adjacent small-model-helps-big-model idea; worth distinguishing from ours |

---

## 5. Summary of the claim after this survey

Stated as it should be defended, weakest hedge first:

1. **C1 is the contribution.** Frame it explicitly as paging fused with
   assignment, cite the caching and inference-serving literature as the
   inheritance, and defend the delta: quality-bearing assignment over a
   dependent DAG under a single-slot ceiling. **AlpaServe and Clockwork must be
   read before this is defended.**
2. **C3 is the most defensible.** Modest, specific, and genuinely absent from
   every orchestration framework. Cite conformal prediction / LTT as borrowed.
3. **C4 is an optimisation, not a method.** Cite prior adaptive-consistency
   work, claim the measurement on constrained hardware.
4. **C2 is aspirational until a second device exists.** Scope it down in
   writing if the device does not materialise, per `BUILD_BRIEF.md` §2.

**Do not defend a claim the literature has already answered.** After the
verification searches above, rewrite this file with real citations and adjust
the claims accordingly — including deleting one if it does not survive.
