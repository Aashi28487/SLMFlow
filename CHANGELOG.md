# Changelog

## Unreleased

- Fix benchmark scorers: JSON enum answers are scored by value; `grounded_qa`
  is scored by evidence support instead of template containment.
- Ollama backend sends `format: "json"` for schema-bound steps.
- `llama_cpp` backend is registered; unknown-backend errors carry `backend`.
- Add `benchmarks/run_all.py` for one-command runs on a fresh machine.
- Add LICENSE (MIT), project URLs, CI workflow.
