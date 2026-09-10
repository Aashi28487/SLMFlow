# SLMFlow — Key Challenges and Mitigations

Slide-ready risk table. Rows map to the contribution claims in `BUILD_BRIEF.md` §0
(**C1** memory-bounded orchestration, **C2** portable profiles, **C3** abstention
contract, **C4** cost-adaptive self-consistency) and to the hardware tiers in §2.

Every row has a defined fallback, so no single failure sinks the project.

| # | Challenge | Why it's hard | How we're tackling it | If it fails / future |
|---|---|---|---|---|
| 1 | **Model swapping dominates cost** | On a laptop or phone, loading a model costs seconds — often more than inference itself. LangChain, LangGraph and RouteLLM all treat the model as a free remote endpoint, so **no scheduler for this exists** | Formalise it as the Memory-Bounded Orchestration Problem (paging + assignment); ship a greedy swap-aware scheduler, plus an offline oracle to measure **scheduler regret** | Report honestly as a negative result ("loads amortise; gain is small") and shift weight to routing quality |
| 2 | **Hard RAM ceiling — 1.5–3 GB on phone tier** | Only one model fits at a time; no daemon, no spare RAM. Any heavy dependency in the import path kills the phone claim outright | Pure-Python core with zero mandatory native deps; single-slot residency as the *default*, not the fallback; memory accounting built in from commit 1 | Scope down in writing to "core imports and runs one 0.5B model end-to-end on device, measured" — never quietly overclaim |
| 3 | **Small-model confidence is unreliable** | SLM-MUX itself admits confidence is self-consistency alone. A confidently wrong drug-repurposing answer is worse than no answer | Multi-signal confidence (log-prob margin, verifier agreement, retrieval support, cross-model contradiction); auto-tune thresholds to a **user-declared precision target**; measure ECE and risk–coverage | Fall back to empirical thresholds on a calibration split, reporting achieved coverage rather than a formal guarantee |
| 4 | **Reliability costs k × \|M\| generations** | Self-consistency multiplies cost by samples × models. On CPU that is minutes per query — deployment-fatal | Sequential self-consistency with early stopping; cascade (cheapest model first, escalate only when uncertain); measure cost cut at matched accuracy | Add swap-aware batching, KV-cache reuse across sub-tasks, per-task quantisation choice |
| 5 | **Novelty and measurement validity** | Routing and cascades are a crowded field (RouteLLM, FrugalGPT); one machine and small models produce noisy numbers that won't survive review | Per-claim novelty audit before building; >=3 seeds with mean +/- std, paired bootstrap / McNemar on every claimed difference, pinned model digests, one-command reproduction | Sharpen the claim to the remaining delta rather than defending a question the literature already answered |

---

## Slide framing

**Suggested title:** *Orchestrating small models is a scheduling problem under a hard
memory budget — and the cost every existing framework ignores is the one that dominates
on real hardware.*

**Speaker notes (not on the slide):**

- Rows 1 and 3 are the publishable contributions. Rows 2, 4 and 5 are the credibility
  that lets them stand.
- Every row has a stated fallback — that is the answer to "what is your risk plan?"
- Per Ground Rule 9, a negative result on row 1 is a valid and expected finding, not a
  project failure.

**Caveat:** this table contains no measured numbers. Cost and accuracy figures are
`not yet measured` until Phases 2, 7, 10 and 11 have been run on this machine.
