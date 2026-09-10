# Comparison with existing frameworks

Phase 20 deliverable, drafted now so the comparison is structural rather than
score-based. Measured rows are filled by Phase 12.

## The axis that matters: what each system assumes is expensive

| | LangChain | LangGraph | AutoGen | RouteLLM / FrugalGPT | SLM-MUX | **SLMFlow** |
|---|---|---|---|---|---|---|
| Design centre | composing LLM calls | stateful agent graphs | agent conversation | cost-aware routing | fixed SLM ensemble | **orchestration under a hard memory ceiling** |
| A model is | a remote endpoint | a remote endpoint | a remote endpoint | a priced API call | a sampling process | **a resident object with a load cost** |
| Optimises | ergonomics | control flow | dialogue | **price per call** | accuracy per token | **quality per second and per GB** |
| Models memory residency | no | no | no | no | no | **yes, per tier** |
| Cost of switching models | zero, implicitly | zero | zero | **zero, by assumption** | n/a | **the dominant cost, measured** |
| Refuses to over-commit memory | n/a | n/a | n/a | n/a | n/a | **yes; `ResourceExhausted` names the tier** |
| Reliability target as configuration | no | no | no | no | no | **yes, `guarantee=Precision(0.9)`** |
| Abstention as a first-class outcome | no | no | no | no | no | **yes, on `Result`** |
| Schema guaranteed, not hoped for | validation | validation | validation | n/a | n/a | **GBNF grammar, else validation** |
| Core runs with zero native dependencies | no | no | no | no | no | **yes** |
| Offline-only by construction | no | no | no | no | yes | **yes, Ground Rule 6** |

**Be fair about this table.** LangChain and LangGraph *can* run local models via
their llama.cpp and Ollama integrations, and they are far more capable than
SLMFlow at nearly everything else — tool use, retrieval abstractions, streaming,
ecosystem, documentation. The claim is narrow and must stay narrow: they do not
treat RAM, residency, swap cost or latency as things the *orchestrator* reasons
about. That gap is the opening. Overstating it is the fastest way to lose a
reviewer, and `context.md` §4 already words it correctly.

## What SLMFlow does not have

Stated plainly, because a comparison listing only our wins is dishonest:

- no tool-use or function-calling abstraction
- no streaming UI, no callback ecosystem, no tracing UI
- no vector-store abstraction beyond one BM25 and one FAISS retriever
- no hosted-model support at all, by design — which makes it unusable for the
  majority of real applications
- no agent-conversation pattern, which is AutoGen's core competence
- a single-file YAML config, not a programmable graph API
- no community, no plugins, no stability guarantees

## Measured comparison (Phase 12, not yet run)

| system | accuracy | s/item | peak VRAM GB | peak RAM GB | acc/s | acc/GB |
|---|---|---|---|---|---|---|
| single small model | not yet measured | | | | | |
| single 7-8B baseline | not yet measured | | | | | |
| SLM-MUX, re-implemented | not yet measured | | | | | |
| Agent-Forest majority vote | not yet measured | | | | | |
| SLMFlow adaptive + greedy scheduler | not yet measured | | | | | |

A direct LangChain or LangGraph row is deliberately **absent**. Running the same
task through them would compare our prompts and our models against our prompts
and our models, differing only in plumbing — a benchmark of the glue, dressed up
as a benchmark of the idea. The honest comparison is structural (the table
above) plus our own ablations, which isolate what each component actually
contributes.
