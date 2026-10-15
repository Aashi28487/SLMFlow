"""Release validation: build, inspect, install and smoke-test SLMFlow.

Run this on the stronger machine, from the repository root. It is NOT meant for
the low-spec laptop. It never runs a model, starts a model server, runs the
benchmarks or uploads anything.

It DOES need network access: pip and the isolated build download pip, build,
twine, hatchling and (for the test step) the ``dev`` extra from the configured
package index (PyPI by default). It also installs packages into two throwaway
virtual environments.

Generated output goes under ``<repo>/release_check/`` only. ``--workdir`` must
be that directory or a subdirectory of it; any other path is refused, and an
existing non-empty directory is deleted only if this script created it (it
carries a marker file).

    python scripts/validate_release.py
    python scripts/validate_release.py --workdir release_check/run2 --keep-going

Steps (each prints PASS/FAIL and elapsed time; the exit code is 0 only if all pass):

 1. report Python/platform
 2. create a build venv, install build + twine
 3. build the sdist and wheel
 4. twine check and inspect archive contents (no data/, no results, LICENSE present)
 5. create a separate clean test venv and install the built wheel
 6. verify the core imports with no optional extras and the CLI starts
 7. mock-only public-API smoke test and CLI smoke test
 8. install the dev extra into the test venv and run the fast test suite
 9. verify optional-dependency behaviour (llama_cpp missing -> actionable error)
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKDIR_BASE = ROOT / "release_check"
WORKDIR_MARKER = ".slmflow_validate_release"
FORBIDDEN_IN_SDIST = ("data/", "benchmarks/results/", "docs/", ".pdf", ".docx", ".gguf")
RESULTS: list[tuple[str, bool, float, str]] = []


def venv_python(env_dir: Path) -> Path:
    """Path of the interpreter inside a venv, on any platform."""
    sub = "Scripts" if sys.platform == "win32" else "bin"
    return env_dir / sub / ("python.exe" if sys.platform == "win32" else "python")


def safe_workdir(arg: str) -> Path:
    """Resolve ``--workdir`` and refuse anything that is not under ``release_check/``.

    Raises:
        ValueError: If the path is outside the dedicated output directory, is not
            a directory, or already holds files this script did not create.
    """
    base = WORKDIR_BASE.resolve()
    candidate = (ROOT / arg).resolve()  # an absolute ``arg`` replaces ROOT; checked below
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError(
            f"refusing --workdir {arg!r} ({candidate}): it must be {base} or a "
            f"subdirectory of it"
        ) from None
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"refusing --workdir {candidate}: it exists and is not a directory")
        if any(candidate.iterdir()) and not (candidate / WORKDIR_MARKER).is_file():
            raise ValueError(
                f"refusing to delete {candidate}: it is not empty and was not created by "
                f"this script (no {WORKDIR_MARKER} marker). Remove it yourself or choose "
                f"another subdirectory of release_check/"
            )
    return candidate


def run(cmd: list[str], *, cwd: Path = ROOT, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    """Run a command, echoing it, capturing output."""
    print("   $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(
        [str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def step(name: str, fn, *, keep_going: bool) -> bool:  # noqa: ANN001
    """Run one step, record the outcome, stop on failure unless asked not to."""
    print(f"\n== {name}", flush=True)
    start = time.perf_counter()
    try:
        detail = fn() or ""
        ok = True
    except Exception as exc:  # noqa: BLE001 - report every failure, never swallow
        detail = f"{type(exc).__name__}: {exc}"
        ok = False
    elapsed = time.perf_counter() - start
    RESULTS.append((name, ok, elapsed, detail))
    print(f"   {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s) {detail}".rstrip(), flush=True)
    if not ok and not keep_going:
        report()
        raise SystemExit(1)
    return ok


def need_ok(proc: subprocess.CompletedProcess[str], what: str) -> str:
    """Raise with the tail of the output when a command failed."""
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-2500:]
        raise RuntimeError(f"{what} exited {proc.returncode}\n{tail}")
    return proc.stdout


def report() -> None:
    """Print the summary table."""
    print("\n" + "=" * 70 + "\nSUMMARY")
    for name, ok, elapsed, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {elapsed:7.1f}s  {name}")
    failed = [n for n, ok, _, _ in RESULTS if not ok]
    print("\nRESULT:", "ALL PASSED" if not failed else f"{len(failed)} FAILED: {failed}")


def main() -> int:
    """Run the validation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--workdir", default="release_check", help="where venvs and dist go")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failure")
    parser.add_argument("--skip-tests", action="store_true", help="skip the pytest step")
    args = parser.parse_args()

    try:
        work = safe_workdir(args.workdir)
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2; nothing has been deleted
    if work.exists():
        shutil.rmtree(work)  # safe: under release_check/ and created by this script
    work.mkdir(parents=True)
    (work / WORKDIR_MARKER).write_text("created by scripts/validate_release.py\n", encoding="utf-8")
    dist = work / "dist"
    build_env, test_env = work / "venv-build", work / "venv-test"
    keep = args.keep_going

    def info() -> str:
        return f"python {sys.version.split()[0]} on {platform.platform()}"

    def make_build_env() -> str:
        venv.EnvBuilder(with_pip=True, clear=True).create(build_env)
        py = venv_python(build_env)
        need_ok(run([py, "-m", "pip", "install", "--upgrade", "pip", "build", "twine"]), "pip install")
        return str(py)

    def build() -> str:
        need_ok(run([venv_python(build_env), "-m", "build", "--outdir", dist]), "python -m build")
        sdists, wheels = list(dist.glob("*.tar.gz")), list(dist.glob("*.whl"))
        if len(sdists) != 1 or len(wheels) != 1:
            raise RuntimeError(f"expected 1 sdist and 1 wheel, found {sdists} {wheels}")
        return f"{sdists[0].name}, {wheels[0].name}"

    def check_dists() -> str:
        need_ok(run([venv_python(build_env), "-m", "twine", "check", *dist.glob("*")]), "twine check")
        sdist = next(dist.glob("*.tar.gz"))
        wheel = next(dist.glob("*.whl"))
        with tarfile.open(sdist) as tar:
            names = tar.getnames()
        bad = [n for n in names if any(tok in n for tok in FORBIDDEN_IN_SDIST)]
        if bad:
            raise RuntimeError(f"sdist contains files that should be excluded: {bad[:10]}")
        if not any(n.endswith("/LICENSE") for n in names):
            raise RuntimeError("LICENSE missing from sdist")
        with zipfile.ZipFile(wheel) as zf:
            wnames = zf.namelist()
        if not any(n.endswith("py.typed") for n in wnames):
            raise RuntimeError("py.typed missing from wheel")
        if any(n.startswith(("benchmarks/", "tests/", "data/")) for n in wnames):
            raise RuntimeError("wheel contains non-package directories")
        if not any(n.endswith("LICENSE") and "dist-info" in n for n in wnames):
            raise RuntimeError("LICENSE missing from wheel metadata")
        size_mb = sdist.stat().st_size / 1e6
        return f"sdist {size_mb:.2f} MB, {len(names)} files; wheel {len(wnames)} files"

    def make_test_env() -> str:
        venv.EnvBuilder(with_pip=True, clear=True).create(test_env)
        py = venv_python(test_env)
        need_ok(run([py, "-m", "pip", "install", "--upgrade", "pip"]), "pip upgrade")
        need_ok(run([py, "-m", "pip", "install", next(dist.glob("*.whl"))]), "wheel install")
        return "wheel installed into a clean environment"

    def core_import() -> str:
        py = venv_python(test_env)
        out = need_ok(run([py, "-c", "import slmflow; print(slmflow.__version__)"], cwd=work), "import")
        listing = need_ok(run([py, "-m", "pip", "list", "--format=freeze"], cwd=work), "pip list")
        extras = [ln for ln in listing.splitlines() if ln.split("==")[0].lower() not in {"pip", "slmflow", "setuptools", "wheel"}]
        if extras:
            raise RuntimeError(f"core install pulled in third-party packages: {extras}")
        cli = need_ok(run([py, "-m", "slmflow.cli", "--version"], cwd=work), "cli --version")
        return f"version {out.strip()}; {cli.strip()}; no third-party packages"

    def api_and_cli_smoke() -> str:
        py = venv_python(test_env)
        snippet = textwrap.dedent(
            """
            from slmflow import MemoryBudget, ModelRegistry, Pipeline
            from slmflow.models.backends.mock import MockBackend, make_pool
            registry = ModelRegistry(make_pool(
                [("small", 0.5, 400_000_000), ("large", 3.0, 2_200_000_000)],
                accuracy={"small": {"classify": 0.62}, "large": {"classify": 0.81}}))
            registry.register_backend("mock", MockBackend(default_answer="investigational"))
            pipe = Pipeline.from_dict(
                {"name": "q", "steps": [{"name": "l", "task_type": "classify", "prompt": "Is it?"}]},
                registry=registry, budget=MemoryBudget.single_slot(2_500_000_000))
            r = pipe.run()
            assert r.answer == "investigational", r.answer
            print("mock quickstart ok")
            """
        )
        need_ok(run([py, "-c", snippet], cwd=work), "mock quickstart")
        cfg = work / "smoke.yaml"
        cfg.write_text("name: smoke\nsteps:\n  - name: s\n    prompt: hello\n", encoding="utf-8")
        proc = run([py, "-m", "slmflow.cli", "run", cfg, "--mock"], cwd=work)
        if proc.returncode not in (0, 3):
            raise RuntimeError(f"cli run --mock exited {proc.returncode}\n{proc.stderr[-1500:]}")
        if "MOCK BACKEND" not in proc.stderr:
            raise RuntimeError("mock run was not labelled as mock on stderr")
        bad = run([py, "-m", "slmflow.cli", "run", work / "absent.yaml", "--mock"], cwd=work)
        if bad.returncode != 2 or "Traceback" in bad.stderr:
            raise RuntimeError(f"missing config should exit 2 cleanly, got {bad.returncode}: {bad.stderr[-800:]}")
        return "API and CLI mock smoke tests ok"

    def optional_behaviour() -> str:
        py = venv_python(test_env)
        snippet = (
            "from slmflow.models.backends import get_backend\n"
            "from slmflow.errors import BackendUnavailable\n"
            "from slmflow.models.backends.mock import make_pool\n"
            "card = make_pool([('m', 0.5, 1)])[0]\n"
            "b = get_backend('llama_cpp')\n"
            "assert b.is_available() is False\n"
            "try:\n    b.load(card)\nexcept BackendUnavailable as e:\n    assert 'slmflow[llama]' in str(e); print('ok')\n"
            "else:\n    raise SystemExit('expected BackendUnavailable')\n"
        )
        need_ok(run([py, "-c", snippet], cwd=work), "optional dependency behaviour")
        return "missing llama_cpp -> BackendUnavailable with install hint"

    def tests() -> str:
        py = venv_python(test_env)
        need_ok(run([py, "-m", "pip", "install", "-e", f"{ROOT}[dev]"]), "pip install -e .[dev]")
        proc = run([py, "-m", "pytest", "-m", "not real_model", "-p", "no:cacheprovider"], cwd=ROOT)
        need_ok(proc, "pytest")
        return proc.stdout.strip().splitlines()[-1]

    ok = step("environment", info, keep_going=keep)
    step("create build environment + install build tools", make_build_env, keep_going=keep)
    step("build sdist and wheel", build, keep_going=keep)
    step("twine check + inspect archive contents", check_dists, keep_going=keep)
    step("install wheel into a clean test environment", make_test_env, keep_going=keep)
    step("core imports with no optional extras; CLI starts", core_import, keep_going=keep)
    step("mock public-API and CLI smoke tests", api_and_cli_smoke, keep_going=keep)
    step("optional-dependency behaviour", optional_behaviour, keep_going=keep)
    if not args.skip_tests:
        step("fast deterministic test suite (editable dev install)", tests, keep_going=keep)

    report()
    print(f"\nArtifacts: {dist}")
    return 0 if all(r[1] for r in RESULTS) and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
