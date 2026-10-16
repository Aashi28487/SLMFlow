# Release procedure (0.1.1)

**Do not run any of this on the low-spec laptop.** Everything below is for the
stronger machine. Production PyPI upload requires explicit owner approval after
the final report has been reviewed.

## 0. Prerequisites on the stronger machine

- Python 3.10-3.13 and `git`; a fresh clone of the repository at the commit to release.
- Nothing else for the release gate: it needs no model, no Ollama and no network
  except to download build tools from PyPI.
- For the optional real-backend smoke test only: Ollama running and the models
  pulled (see `RUN.md`).
- Accounts: a PyPI account and a TestPyPI account (they are separate). Create an
  API token on each, scoped to the project once it exists. **Never paste a token
  into a file, commit, chat or log.**

## 1. Owner decisions needed before upload

| Decision | Current state |
|---|---|
| Authors / contact emails | Names only (Aashi, Sneha, Rutvik, Prakriti). Add `email = "..."` to `[project].authors` only with the owners' consent. |
| Licence | MIT, copyright "2026 Aashi, Sneha, Rutvik, Prakriti". Confirm with all four authors and the institution that MIT is permitted (e.g. no university IP claim). |
| Repository URL | `https://github.com/Aashi28487/SLMFlow` is the confirmed canonical repository for 0.1.1 and is already public (verified 2026-10-16). `pyproject.toml`, `README.md`, `RUN.md` and `tests/test_release.py` all reference it. |
| Package name | `slmflow` returned 404 on both pypi.org and test.pypi.org when checked on 2026-10-15, i.e. unregistered. Only the first upload actually reserves it, and PyPI also rejects names too similar to existing ones. |
| Version | `0.1.1` (alpha) in `src/slmflow/__init__.py`. A version can never be re-uploaded or replaced once published. `0.1.0` is skipped: TestPyPI rejected its filenames ("previously used by a file that has since been deleted"); never rename or re-upload those files. |

## 2. Validate (stronger machine)

```bash
git clone https://github.com/Aashi28487/SLMFlow && cd SLMFlow
python scripts/validate_release.py
```

This builds the sdist and wheel in a build venv, runs `twine check`, inspects the
archives, installs the wheel into a *separate* clean venv, checks that the core
pulls no third-party packages, runs mock API and CLI smoke tests, checks optional
dependency behaviour, then runs `pytest -m "not real_model"`. Artifacts land in
`release_check/dist/`. Use `--keep-going` to see all failures at once.

Optional, separately scheduled real-backend smoke test (not part of the gate):

```bash
python -m benchmarks.run_all --fresh --seeds 1 --limit 20
```

Interpret it honestly: accuracy must beat the majority-class baseline printed in
the results table, and `slm_mux`/`agent_forest` abstaining on everything is a
known problem with small models and no grammar decoding, not a pass.

## 3. TestPyPI

Credentials via environment variables only (token starts with `pypi-`):

```bash
# bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<testpypi token>        # typed into your shell, never saved
python -m twine upload --repository testpypi release_check/dist/*
```

```powershell
# PowerShell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = Read-Host "TestPyPI token"   # not echoed into a file
python -m twine upload --repository testpypi release_check/dist/*
```

Then install it into a brand-new environment and run the quickstart:

```bash
python -m venv tp
source tp/bin/activate                  # PowerShell: tp\Scripts\Activate.ps1
pip install --index-url https://test.pypi.org/simple/ slmflow==0.1.1
python -c "import slmflow; print(slmflow.__version__)"
slmflow --version
```

Create a small config file (this one-liner works the same in bash, PowerShell
and cmd; or save the three YAML lines below in `smoke.yaml` with any editor) and
run it against the mock backend:

```bash
python -c "open('smoke.yaml','w').write('name: t\nsteps:\n  - name: s\n    prompt: hi\n')"
slmflow run smoke.yaml --mock
```

```yaml
name: t
steps:
  - name: s
    prompt: hi
```

The run prints a `MOCK BACKEND` notice on stderr; the answer is scripted, not a
model output. Exit code 0 (or 3 if it abstained) means the install works.

Trusted publishing (GitHub Actions OIDC) is an alternative to tokens and avoids
storing any secret; set it up in the PyPI project settings if you prefer it.

## 4. Final report and approval

Fill the checklist: every `validate_release.py` step PASS, TestPyPI install and
quickstart PASS, README renders on the TestPyPI page, owner decisions in section 1
resolved, known limitations still stated in the README.

## 5. Production PyPI (only after explicit written approval)

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<pypi token>
python -m twine upload release_check/dist/*
```

Verify afterwards in a clean environment: `pip install slmflow==0.1.1`. Then tag
the release in git (`git tag v0.1.1`) only if the owners authorise pushing tags.
