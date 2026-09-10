# SLMFlow — Sequence Diagram Prompt

Prompt for draw.io (Arrange > Insert > Advanced > Mermaid) or ChatGPT free tier.
Shows one `Pipeline.run()` call across the component seams named in
`BUILD_BRIEF.md` §4. The `alt` fragments carry the graded claims:
residency/swap = **C1**, confidence and abstention = **C3**,
early-stopping self-consistency = **C4**.

---

```
Create a UML sequence diagram for "SLMFlow", a memory-bounded orchestration
framework that runs small language models locally on CPU under a hard RAM
budget. Output Mermaid `sequenceDiagram` code I can paste into draw.io via
Arrange > Insert > Advanced > Mermaid, and also render it as an image.

Participants, left to right: User, Pipeline, ConfigLoader, DeviceProbe,
Registry, Router, ResourceManager, Backend, Reliability, Telemetry.

Style: activation bars on every call, dashed arrows for returns, and
labelled loop and alt fragments boxed around the repeated parts.

Messages:
User -> Pipeline: run(task, Precision(0.90))
Pipeline -> ConfigLoader: load YAML, returns validated spec
Pipeline -> DeviceProbe: probe(), returns DeviceCard (RAM, cores, tier)
Pipeline -> Registry: models_within(budget B), returns ModelCards
loop for each sub-task in DAG:
  Pipeline -> Router: select(sub-task, live memory state)
  Pipeline -> ResourceManager: resident(model)
  alt model not resident: ResourceManager -> Backend: evict LRU, load model
  else resident: cache hit, zero swap cost
  Pipeline -> Backend: generate(prompt, JSON schema or GBNF)
  Pipeline -> Reliability: score(samples), returns confidence
  alt confidence below threshold: Pipeline -> Router: escalate or abstain
  else: accept answer
  Pipeline -> Telemetry: record(latency, peak bytes, tokens/sec)
Pipeline --> User: answer, confidence, JSONL trace

Keep message labels under seven words. Fit on one page.
```

---

## Usage notes

- **draw.io**: if **Arrange > Insert > Advanced > Mermaid** is missing on your build,
  paste the Mermaid into **Extras > Edit Diagram** instead. Mermaid renders sequence
  diagrams as a static image — ungroup once inserted if you need to edit lifelines.
- **ChatGPT free tier**: ~220 words, fits a single turn.
- Pairs with [ACTIVITY_DIAGRAM_PROMPT.md](ACTIVITY_DIAGRAM_PROMPT.md) — same runtime
  flow, viewed by component rather than by control flow.
