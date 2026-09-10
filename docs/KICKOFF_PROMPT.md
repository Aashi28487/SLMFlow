# SLMFlow — Kickoff Prompt for Claude Code

Paste the block below as the **first message** of a fresh Claude Code session in
`C:\FYP\implementation`. It does not restate the brief — it points at it and fixes
the two things the brief cannot enforce on its own: **the hardware truth** and
**the execution order**.

---

```
Read BUILD_BRIEF.md in full before writing any code, then read context.md,
slm-mux.pdf and Topic_1 SLM.pptx.pdf. BUILD_BRIEF.md is authoritative; where it
and context.md disagree, follow the brief and log the conflict in
docs/DECISIONS.md.

Hardware you are building for (BUILD_BRIEF.md §2, corrected 2026-09-10):
Intel i5 HX-series, discrete GPU with ~6 GB VRAM and working CUDA, 16 GB system
RAM, 512 GB SSD. Earlier drafts said "Iris iGPU, no CUDA, no usable VRAM" — that
was wrong, ignore it. Probe the real CPU and GPU model numbers in Phase 0 and
write them into docs/ENVIRONMENT.md by running code, not by assuming.

Two consequences you must design for from the first commit:
1. The memory budget is a vector, not a scalar: ~5 GB usable VRAM over ~8-10 GB
   usable system RAM over the SSD. Track and report peak_vram_bytes and
   peak_ram_bytes separately. Treat GPU/CPU as the fast/slow tiers of one cache
   hierarchy — that framing is the C1 contribution, so write it up in
   ARCHITECTURE.md.
2. CUDA stays inside the llama_cpp backend plugin behind a lazy import. Nothing
   native enters the `import slmflow` path. A DeviceCard must still construct on
   a machine with no GPU.

Do NOT download a set of candidate models and start benchmarking. Follow the
model acquisition ramp in §2.1 — one small model, measure and commit, then a
second, and only expand the sweep after the vertical slice is green. Cap total
GGUF weights at ~40 GB and keep a digest manifest; the SSD is shared with data/.

Follow the execution order in §6's "Phase order amendment", not the phase
numbering: Phase 0, Phase 1, Phase 4 architecture, Phase 2a model #1, Phase 5
partial (registry + resource manager + MockBackend, full test suite green with
no weights on disk), Phase 2a model #2, Phase 5 complete (YAML pipeline over two
real models with a real trace), Phase 7 heuristic routing only. Stop at that
checkpoint and show me the trace before starting the Phase 2b sweep.

Ground rules that I will check: never fabricate a number — write "not yet
measured"; git init and .gitignore before the first commit; commit after each
increment; never leave main with failing tests; log every judgment call in
docs/DECISIONS.md with a date. Start with Phase 0 now and work through to the
checkpoint without waiting for me between phases unless you hit a genuine
blocker under §7.
```

---

## Why the reordering (for your own notes / viva answer)

The brief's numbering puts the full model benchmark at Phase 2, before the architecture
exists. That is the dependency order — you cannot pick models without profiles — but
executing it literally means 20–50 GB of downloads and days of cold-load measurement
producing profiles for models the architecture may never use.

The amendment splits it: **two models is enough to prove routing, swapping and ensembling
are real.** The sweep's job is choosing the *best* models, which only matters once the
machinery works. What it costs is one likely revision of the `ModelCard` schema after the
sweep lands — cheap, versus benchmarking the wrong candidate set.

## What changed in the source docs, 2026-09-10

| File | Change |
|---|---|
| `context.md` §6 | Hardware bullets corrected (6 GB VRAM, CUDA available, 512 GB SSD); dated correction note; two-tier memory paragraph added |
| `BUILD_BRIEF.md` §2 | Tier table rewritten: T1 is the real machine, T2 marked largely unnecessary; hardware correction note with the vector-budget and on-machine-baseline consequences |
| `BUILD_BRIEF.md` §2 | Four design rules added: CUDA as plugin only, VRAM/RAM accounted separately, disk as a third budget, local QLoRA now borderline-feasible |
| `BUILD_BRIEF.md` §2.1 | **New** — staged model acquisition ramp with per-stage gates |
| `BUILD_BRIEF.md` §6 | **New** — phase order amendment: Phase 2a/2b split, execution order, stated trade-off |
