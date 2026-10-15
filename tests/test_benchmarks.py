"""Scorer and backend-registration regressions.

These pin the bugs that made every benchmark accuracy read 0.000: a JSON enum
answer compared against a bare label, and a groundedness metric that required
the model to reproduce a fixed answer template.
"""

from __future__ import annotations

import importlib.util

import pytest


def _benchmarks_available() -> bool:
    """True when the repository's ``benchmarks`` package can be found.

    The package is not part of the wheel or the source distribution, so these
    tests only make sense in a source checkout. Only "not found" is treated as
    absence: if the package exists but fails to import, that error propagates.
    """
    try:
        return importlib.util.find_spec("benchmarks.harness") is not None
    except ModuleNotFoundError:
        return False


if not _benchmarks_available():
    pytest.skip(
        "the 'benchmarks' package is not present (running outside a source checkout)",
        allow_module_level=True,
    )

from benchmarks.harness import GROUNDED_THRESHOLD, _score_item  # noqa: E402
from benchmarks.tasks import BenchmarkItem, TaskFamily  # noqa: E402
from slmflow.errors import BackendUnavailable  # noqa: E402
from slmflow.models.backends import get_backend  # noqa: E402
from slmflow.models.backends.mock import MockBackend  # noqa: E402
from slmflow.types import TaskType  # noqa: E402


def _family(metric: str) -> TaskFamily:
    return TaskFamily(
        name="t",
        task_type=TaskType.CLASSIFY,
        items=(),
        prompt_template="",
        output_schema=None,
        metric=metric,
    )


def _item(gold: str) -> BenchmarkItem:
    return BenchmarkItem(item_id="1", inputs={}, gold=gold, task_type=TaskType.CLASSIFY)


class TestExactMatch:
    def test_bare_label(self) -> None:
        assert _score_item(_family("exact_match"), _item("approved"), "approved")

    def test_json_enum_answer(self) -> None:
        assert _score_item(
            _family("exact_match"), _item("approved"), '{"label": "approved"}'
        )

    def test_wrong_label(self) -> None:
        assert not _score_item(
            _family("exact_match"), _item("approved"), '{"label": "hypothesis"}'
        )


class TestJsonExactMatch:
    def test_match_and_mismatch(self) -> None:
        item = _item('{"max_phase": 4, "n_trials": 3}')
        family = _family("json_exact_match")
        assert _score_item(family, item, '{"max_phase": 4, "n_trials": 3}')
        assert not _score_item(family, item, '{"max_phase": 3, "n_trials": 3}')
        assert not _score_item(family, item, "not json")


class TestGroundedness:
    def test_supported_answer_is_correct(self) -> None:
        item = _item("Yes - there is registered clinical evidence ...")
        assert _score_item(_family("groundedness"), item, "some answer", GROUNDED_THRESHOLD)

    def test_unsupported_answer_is_wrong(self) -> None:
        item = _item("Yes - there is registered clinical evidence ...")
        assert not _score_item(_family("groundedness"), item, "some answer", 0.1)

    def test_empty_answer_is_wrong(self) -> None:
        assert not _score_item(_family("groundedness"), _item("x"), "  ", 1.0)

    def test_missing_groundedness_is_wrong(self) -> None:
        assert not _score_item(_family("groundedness"), _item("x"), "answer", None)


class TestBackendRegistry:
    def test_mock(self) -> None:
        assert isinstance(get_backend("mock"), MockBackend)

    def test_unknown_backend_raises_cleanly(self) -> None:
        with pytest.raises(BackendUnavailable) as info:
            get_backend("does-not-exist")
        assert info.value.backend == "does-not-exist"


class TestRunAllSpecs:
    def test_default_backend(self) -> None:
        from benchmarks.run_all import parse_model_spec

        assert parse_model_spec("llama3.2:3b-instruct-q4_K_M:3.0") == (
            "llama3.2:3b-instruct-q4_K_M",
            3.0,
            "ollama",
        )

    def test_explicit_backend(self) -> None:
        from benchmarks.run_all import parse_model_spec

        assert parse_model_spec("m.gguf:7.0:llama_cpp") == ("m.gguf", 7.0, "llama_cpp")

    def test_missing_params_rejected(self) -> None:
        from benchmarks.run_all import parse_model_spec

        with pytest.raises(ValueError):
            parse_model_spec("justaname")

    def test_dry_run_runs_nothing(self) -> None:
        from benchmarks.run_all import main

        assert main(["--dry-run", "--skip-profile"]) == 0
