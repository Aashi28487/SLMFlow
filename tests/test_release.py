"""Release-gate tests: deterministic, no network, no model weights, no Ollama.

These guard the things a first-time user of the published package hits:
the core imports with no extras installed, backends fail with actionable
errors, bad input produces a one-line CLI error rather than a traceback, and
the documented quickstart actually runs.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import slmflow
from slmflow.errors import BackendUnavailable, ConfigError
from slmflow.models.backends import available_backends, get_backend

ROOT = Path(__file__).resolve().parent.parent
OPTIONAL_MODULES = (
    "numpy", "yaml", "psutil", "httpx", "typer", "faiss",
    "sentence_transformers", "llama_cpp", "matplotlib",
)


def _no_network(*_args: object, **_kwargs: object) -> None:
    raise urllib.error.URLError(ConnectionRefusedError("no daemon (test)"))


# --------------------------------------------------------------------------- #
class TestCoreImport:
    def test_imports_with_every_optional_dependency_blocked(self) -> None:
        """``import slmflow`` must work with none of the optional extras installed."""
        code = (
            "import sys\n"
            f"for n in {OPTIONAL_MODULES!r}:\n"
            "    sys.modules[n] = None\n"
            "import slmflow\n"
            "from slmflow.cli import build_parser\n"
            "from slmflow.models.backends import get_backend\n"
            "get_backend('mock'); get_backend('ollama'); get_backend('llama_cpp')\n"
            "print('ok', slmflow.__version__)\n"
        )
        src_dir = str(Path(slmflow.__file__).resolve().parent.parent)
        import os

        env = dict(os.environ)
        env["PYTHONPATH"] = src_dir
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.startswith("ok")

    def test_public_names_resolve(self) -> None:
        for name in slmflow.__all__:
            assert hasattr(slmflow, name), name

    def test_version_is_a_string(self) -> None:
        assert isinstance(slmflow.__version__, str) and slmflow.__version__


# --------------------------------------------------------------------------- #
class TestBackendRegistry:
    def test_known_backends_construct_without_side_effects(self) -> None:
        assert get_backend("mock").name == "mock"
        assert get_backend("ollama").name == "ollama"
        assert get_backend("llama_cpp").name == "llama_cpp"

    def test_unknown_backend_raises_with_backend_attribute(self) -> None:
        with pytest.raises(BackendUnavailable) as info:
            get_backend("bogus")
        assert info.value.backend == "bogus"
        assert "mock" in str(info.value)

    def test_available_backends_reports_without_network(self, monkeypatch) -> None:
        monkeypatch.setattr(urllib.request, "urlopen", _no_network)
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        status = available_backends()
        assert status == {"ollama": False, "llama_cpp": False, "mock": True}

    def test_llama_cpp_missing_extra_gives_install_hint(self, monkeypatch, small_card) -> None:
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        backend = get_backend("llama_cpp")
        assert backend.is_available() is False
        with pytest.raises(BackendUnavailable, match=r"slmflow\[llama\]"):
            backend.load(small_card)


class TestOllamaErrors:
    def test_daemon_down_on_load_is_actionable(self, monkeypatch, small_card) -> None:
        monkeypatch.setattr(urllib.request, "urlopen", _no_network)
        with pytest.raises(BackendUnavailable, match="ollama serve"):
            get_backend("ollama").load(small_card)

    def test_daemon_down_on_generate_is_not_a_silent_wrong_answer(
        self, monkeypatch, small_card
    ) -> None:
        monkeypatch.setattr(urllib.request, "urlopen", _no_network)
        with pytest.raises(BackendUnavailable, match="ollama serve"):
            get_backend("ollama").generate(small_card, "hi")

    def test_unpulled_model_says_how_to_pull_it(self, monkeypatch, small_card) -> None:
        def not_found(*_a: object, **_k: object) -> None:
            raise urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)  # type: ignore[arg-type]

        monkeypatch.setattr(urllib.request, "urlopen", not_found)
        with pytest.raises(BackendUnavailable, match="ollama pull small"):
            get_backend("ollama").generate(small_card, "hi")

    def test_timeout_is_reported_not_raised_as_unavailable(self, monkeypatch, small_card) -> None:
        def slow(*_a: object, **_k: object) -> None:
            raise urllib.error.URLError(TimeoutError("timed out"))

        monkeypatch.setattr(urllib.request, "urlopen", slow)
        gen = get_backend("ollama").generate(small_card, "hi")
        assert gen.text.startswith("ERROR:")


# --------------------------------------------------------------------------- #
class TestConfigParsing:
    def test_valid_mapping(self) -> None:
        config = slmflow.parse_config({"name": "p", "steps": [{"name": "s", "prompt": "hi"}]})
        assert config.name == "p"
        assert len(config.steps) == 1

    def test_valid_yaml_string(self) -> None:
        config = slmflow.parse_config("name: y\nsteps:\n  - name: s\n    prompt: hi\n")
        assert config.name == "y"

    @pytest.mark.parametrize(
        "bad",
        [[], 3, {"name": "p"}, {"name": "p", "steps": []}, {"name": "p", "steps": "nope"}],
    )
    def test_malformed_config_raises_config_error(self, bad) -> None:
        with pytest.raises(ConfigError):
            slmflow.parse_config(bad)

    def test_missing_config_file_raises_config_error(self, tmp_path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            slmflow.load_config(tmp_path / "absent.yaml")

    def test_invalid_json_file_raises_config_error(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid JSON"):
            slmflow.load_config(path)

    def test_missing_default_cards_file_is_actionable(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigError, match="model cards"):
            slmflow.ModelRegistry.from_file()


# --------------------------------------------------------------------------- #
class TestQuickstart:
    def test_documented_mock_quickstart_runs(self) -> None:
        """The README / module-docstring quickstart, verbatim in behaviour."""
        from slmflow import MemoryBudget, ModelRegistry, Pipeline
        from slmflow.models.backends.mock import MockBackend, make_pool

        registry = ModelRegistry(make_pool(
            [("small", 0.5, 400_000_000), ("large", 3.0, 2_200_000_000)],
            accuracy={"small": {"classify": 0.62}, "large": {"classify": 0.81}},
        ))
        registry.register_backend("mock", MockBackend(default_answer="investigational"))
        pipeline = Pipeline.from_dict(
            {"name": "quickstart",
             "steps": [{"name": "label", "task_type": "classify", "prompt": "Is it?"}]},
            registry=registry,
            budget=MemoryBudget.single_slot(2_500_000_000),
        )
        result = pipeline.run()
        assert result.answer == "investigational"
        assert result.trace.total_swaps >= 1


# --------------------------------------------------------------------------- #
class TestCLI:
    def _config(self, tmp_path) -> Path:
        path = tmp_path / "p.yaml"
        path.write_text("name: t\nsteps:\n  - name: s\n    prompt: hello\n", encoding="utf-8")
        return path

    def test_version_flag(self, capsys) -> None:
        from slmflow.cli import main

        with pytest.raises(SystemExit) as info:
            main(["--version"])
        assert info.value.code == 0
        assert slmflow.__version__ in capsys.readouterr().out

    @pytest.mark.parametrize("argv", [[], ["bogus"], ["run"]])
    def test_bad_arguments_exit_2(self, argv) -> None:
        from slmflow.cli import main

        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == 2

    def test_missing_config_is_one_line_error_not_traceback(self, tmp_path, capsys) -> None:
        from slmflow.cli import main

        assert main(["run", str(tmp_path / "absent.yaml"), "--mock"]) == 2
        assert "not found" in capsys.readouterr().err

    def test_missing_cards_file_is_one_line_error(self, tmp_path, capsys) -> None:
        from slmflow.cli import main

        assert main(["models", "--cards", str(tmp_path / "absent.json")]) == 2
        assert "model cards file" in capsys.readouterr().err

    def test_inspect_missing_trace(self, tmp_path, capsys) -> None:
        from slmflow.cli import main

        assert main(["inspect", str(tmp_path / "absent.jsonl")]) == 1
        assert "no run records" in capsys.readouterr().err

    def test_malformed_input_flag(self, tmp_path) -> None:
        from slmflow.cli import main

        with pytest.raises(SystemExit) as info:
            main(["plan", str(self._config(tmp_path)), "--mock", "--input", "oops"])
        assert "KEY=VALUE" in str(info.value)

    def test_mock_runs_are_labelled_as_mock(self, tmp_path, capsys) -> None:
        from slmflow.cli import main

        main(["run", str(self._config(tmp_path)), "--mock"])
        assert "MOCK BACKEND" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
class TestRepositoryMetadata:
    def test_license_present(self) -> None:
        license_file = ROOT / "LICENSE"
        if not license_file.exists():
            pytest.skip("not running from a source tree")
        assert license_file.read_text(encoding="utf-8").startswith("MIT License")

    def test_no_placeholder_urls_in_pyproject(self) -> None:
        pyproject = ROOT / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("not running from a source tree")
        text = pyproject.read_text(encoding="utf-8")
        for placeholder in ("your-username", "example.com", "TODO", "CHANGEME"):
            assert placeholder not in text, placeholder
        assert "github.com/Aashi28487/SLMFlow" in text
