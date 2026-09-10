# SLMFlow — Activity Diagram Prompt

Prompt for draw.io (Arrange > Insert > Advanced > Mermaid) or ChatGPT free tier.
Describes the **runtime execution flow** of the framework, not the build phases.
Decision points map to the graded claims in `BUILD_BRIEF.md` §0:
residency/swap = **C1**, confidence/abstain = **C3**, early-stopping self-consistency = **C4**.

---

```
Create a UML activity diagram for "SLMFlow", a memory-bounded orchestration
framework that runs small language models locally on CPU under a hard RAM
budget. Output Mermaid `flowchart TD` code I can paste into draw.io via
Extras > Edit Diagram, and also render it as an image.

Style: pink rounded START, yellow oval END, light-blue rectangles for
actions, light-purple diamonds for decisions, every branch labelled Yes/No,
strict top-to-bottom flow with loop-back arrows routed down the side.

Flow:
START -> Load YAML pipeline config -> "Config valid?" No -> Report schema
error -> END. Yes -> Probe device (RAM, cores, tier) -> Load model registry
(measured model cards) -> "Any model fits memory budget B?" No -> Raise
ResourceExhausted -> END. Yes -> Decompose task into sub-task DAG -> Route
sub-task using capability profile + live memory state -> "Model already
resident?" No -> Evict LRU model and load (swap cost) -> rejoin. Yes ->
Generate output with JSON-schema/GBNF constraint -> "Output schema valid?"
No -> Fallback to alternate model -> rejoin. Yes -> Score confidence via
early-stopping self-consistency -> "Confidence >= target?" No -> Escalate
to larger model or Abstain -> rejoin -> "More sub-tasks in DAG?" Yes ->
back to Route sub-task. No -> Write JSONL trace (quality + cost, cold and
warm) -> Return answer with confidence score -> END.

Keep every label under six words. Fit on one page.
```

---

## Usage notes

- **draw.io**: older builds reject Mermaid in Extras > Edit Diagram — use
  **Arrange > Insert > Advanced > Mermaid** instead, then **Layout > Vertical Flow**
  to tidy the loop-back arrows.
- **ChatGPT free tier**: ~215 words, comfortably within a single turn.
