# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## 0.1.1 - unreleased (initial alpha)

First public release. Alpha: the API may change before 1.0, and no performance
advantage is claimed (see the README limitations). Version 0.1.0 was never
published: TestPyPI rejected its filenames as previously used by a deleted
file, so the first release is 0.1.1. The contents are those prepared as 0.1.0.

### Added
- Memory-bounded orchestration core: two-tier memory budget, resource manager
  with LRU and largest-first eviction, swap ledger.
- Schedulers (swap-aware greedy, offline oracle, naive load-on-demand) and
  regret measurement.
- Routers (heuristic, learned, random, static, oracle).
- Strategies: single, SLM-MUX, agent-forest, cascade, adaptive self-consistency.
- Reliability: confidence signals, `Precision(target)` calibration with a
  Clopper-Pearson bound, abstention policy, ECE, Brier, bootstrap, McNemar.
- Backends: Ollama, llama.cpp (`slmflow[llama]`) and a mock backend.
- `slmflow` command line: `probe`, `doctor`, `plan`, `run`, `inspect`, `models`.
- Zero required dependencies; optional extras `ollama`, `llama`, `yaml`, `rag`,
  `bench`, `dev`.

### Fixed (during release preparation)
- Benchmark scorers: JSON enum answers are scored by value; `grounded_qa` is
  scored by evidence support instead of template containment. Earlier all-zero
  accuracy numbers were invalid and are not to be cited.
- `llama_cpp` backend is tracked and registered; unknown-backend errors carry
  the `backend` attribute.
- Ollama backend: JSON mode for schema-bound steps; an unreachable daemon or an
  unpulled model now raises `BackendUnavailable` with the command to fix it
  instead of returning an `ERROR:` string that was scored as a wrong answer.
- `Pipeline.from_config(path)` without an explicit registry no longer fails with
  a `TypeError`; `ModelRegistry.from_file()` defaults to the standard cards path
  and reports a missing file with instructions.
- A missing or malformed config file is a `ConfigError` (one line on the CLI),
  not a traceback.
- `slmflow --mock` runs print a `MOCK BACKEND` notice on stderr.
- A flaky cold/warm load timing test is now deterministic.

### Packaging
- MIT `LICENSE`, real project URLs, four authors.
- Removed the unused `httpx` and `typer` dependencies (the `ollama` extra no
  longer installs anything; the `cli` extra was removed).
- Source distribution excludes datasets, benchmark outputs and scratch files.
- CI builds the wheel and sdist and runs `twine check`.
