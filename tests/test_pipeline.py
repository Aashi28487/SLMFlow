"""End-to-end tests: config, routing, strategies, schema, pipeline, CLI.

Phase 5's Definition of Done requires a YAML-configured pipeline over >= 2
models running end to end and producing an answer plus a structured trace. The
mock-backed version of that test lives here; the real-model version is marked
``real_model`` and runs only on a machine with weights.
"""

from __future__ import annotations

import json

import pytest

from slmflow.agents.schema import (
    enum_schema,
    extract_json,
    object_schema,
    parse_and_validate,
    schema_to_gbnf,
    validate,
)
from slmflow.config.loader import parse_config
from slmflow.config.minyaml import parse_yaml
from slmflow.errors import ConfigError, SchemaViolation
from slmflow.models.backends.mock import MockBackend, ScriptedResponse, make_pool
from slmflow.models.registry import ModelRegistry, card_from_dict, card_to_dict
from slmflow.orchestration.pipeline import Pipeline, PipelineOptions
from slmflow.resource.budget import MemoryBudget
from slmflow.routing.features import FEATURE_NAMES, extract_features
from slmflow.routing.heuristic import HeuristicRouter, HeuristicWeights, StaticRouter
from slmflow.routing.learned import LearnedRouter, TrainingExample, train_router
from slmflow.strategies.slm_mux import select_model_set
from slmflow.types import ModelCard, TaskGraph, TaskType

GB = 1_000_000_000


# --------------------------------------------------------------------------- #
class TestMiniYaml:
    def test_nested_mappings_and_sequences(self):
        text = """
name: demo
budget:
  ram_gb: 8.0
steps:
  - name: first
    prompt: hello
  - name: second
    prompt: world
"""
        parsed = parse_yaml(text)
        assert parsed["name"] == "demo"
        assert parsed["budget"]["ram_gb"] == 8.0
        assert [s["name"] for s in parsed["steps"]] == ["first", "second"]

    def test_inline_collections(self):
        parsed = parse_yaml("models: [a, b, c]\nflags: {fast: true, k: 5}")
        assert parsed["models"] == ["a", "b", "c"]
        assert parsed["flags"] == {"fast": True, "k": 5}

    def test_block_scalar(self):
        parsed = parse_yaml("prompt: |\n  line one\n  line two\nk: 2")
        assert "line one" in parsed["prompt"]
        assert parsed["k"] == 2

    def test_comments_and_blanks_ignored(self):
        parsed = parse_yaml("# header\n\nname: x  # trailing\n")
        assert parsed == {"name": "x"}

    def test_scalars_are_typed(self):
        parsed = parse_yaml("a: 1\nb: 1.5\nc: true\nd: null\ne: text\nf: 'quoted'")
        assert parsed == {
            "a": 1, "b": 1.5, "c": True, "d": None, "e": "text", "f": "quoted"
        }

    def test_tabs_are_rejected_with_a_line_number(self):
        with pytest.raises(ConfigError, match="tab"):
            parse_yaml("a:\n\tb: 1")

    def test_anchors_refused_rather_than_misread(self):
        """Silently mishandling an anchor would corrupt an experiment."""
        with pytest.raises(ConfigError, match="anchors"):
            parse_yaml("base: &base\n  k: 1")

    def test_duplicate_key_rejected(self):
        with pytest.raises(ConfigError, match="duplicate"):
            parse_yaml("a: 1\na: 2")


class TestConfigValidation:
    def _minimal(self) -> dict:
        return {"name": "t", "steps": [{"name": "s", "prompt": "hi"}]}

    def test_minimal_config_parses(self):
        config = parse_config(self._minimal())
        assert config.name == "t"
        assert len(config.steps) == 1
        assert config.digest  # a stable identity for the results artifact

    def test_missing_steps_rejected(self):
        with pytest.raises(ConfigError, match="steps"):
            parse_config({"name": "t"})

    def test_empty_prompt_rejected(self):
        with pytest.raises(ConfigError, match="prompt"):
            parse_config({"name": "t", "steps": [{"name": "s", "prompt": "  "}]})

    def test_unknown_strategy_lists_valid_options(self):
        with pytest.raises(ConfigError) as excinfo:
            parse_config(
                {"name": "t", "steps": [{"name": "s", "prompt": "x", "strategy": "magic"}]}
            )
        assert "slm_mux" in str(excinfo.value)

    def test_sampling_strategy_at_zero_temperature_is_rejected(self):
        """A vacuous confidence signal is a silent experiment failure."""
        with pytest.raises(ConfigError, match="temperature"):
            parse_config(
                {
                    "name": "t",
                    "steps": [
                        {"name": "s", "prompt": "x", "strategy": "slm_mux", "temperature": 0.0}
                    ],
                }
            )

    def test_unknown_task_type_lists_valid_options(self):
        with pytest.raises(ConfigError) as excinfo:
            parse_config(
                {"name": "t", "steps": [{"name": "s", "prompt": "x", "task_type": "vibes"}]}
            )
        assert "classify" in str(excinfo.value)

    def test_budget_without_any_tier_is_rejected(self):
        with pytest.raises(ConfigError, match="neither"):
            parse_config({**self._minimal(), "budget": {"unrelated": 1}})

    def test_guarantee_shorthand_and_mapping(self):
        assert parse_config({**self._minimal(), "guarantee": 0.9}).guarantee.target == 0.9
        config = parse_config(
            {**self._minimal(), "guarantee": {"precision": 0.85, "delta": 0.05}}
        )
        assert config.guarantee.delta == 0.05

    def test_prompt_placeholder_missing_input_is_reported(self):
        config = parse_config(
            {"name": "t", "steps": [{"name": "s", "prompt": "About {drug}."}]}
        )
        with pytest.raises(ConfigError, match="drug"):
            config.graph({"other": "x"})

    def test_dependency_cycle_rejected(self):
        config = parse_config(
            {
                "name": "t",
                "steps": [
                    {"name": "a", "prompt": "x", "depends_on": ["b"]},
                    {"name": "b", "prompt": "y", "depends_on": ["a"]},
                ],
            }
        )
        with pytest.raises(ConfigError, match="cycle"):
            config.graph()

    def test_dangling_dependency_rejected(self):
        config = parse_config(
            {"name": "t", "steps": [{"name": "a", "prompt": "x", "depends_on": ["ghost"]}]}
        )
        with pytest.raises(ConfigError, match="ghost"):
            config.graph()


# --------------------------------------------------------------------------- #
class TestSchemaEnforcement:
    def test_enum_schema_validates(self):
        schema = enum_schema(["approved", "investigational"])
        assert validate({"label": "approved"}, schema)
        assert not validate({"label": "maybe"}, schema)

    def test_additional_properties_rejected(self):
        schema = enum_schema(["a"])
        result = validate({"label": "a", "extra": 1}, schema)
        assert not result.valid
        assert any("unexpected" in e for e in result.errors)

    def test_type_and_range_checks(self):
        schema = object_schema(
            {"max_phase": {"type": "integer", "minimum": 0, "maximum": 4}},
            required=["max_phase"],
        )
        assert validate({"max_phase": 3}, schema)
        assert not validate({"max_phase": 9}, schema)
        assert not validate({"max_phase": "3"}, schema)

    def test_bool_is_not_an_integer(self):
        schema = object_schema({"n": {"type": "integer"}}, required=["n"])
        assert not validate({"n": True}, schema)

    def test_extract_json_salvages_chatty_output(self):
        assert extract_json('Sure! ```json\n{"label": "approved"}\n```') == {
            "label": "approved"
        }
        assert extract_json("here it is: [1, 2, 3] ok") == [1, 2, 3]
        assert extract_json("no json here") is None

    def test_extract_json_respects_string_escapes(self):
        assert extract_json('{"t": "a}b"}') == {"t": "a}b"}

    def test_strict_parse_raises_with_the_raw_output(self):
        with pytest.raises(SchemaViolation) as excinfo:
            parse_and_validate("not json", enum_schema(["a"]), strict=True)
        assert excinfo.value.raw_output == "not json"
        assert "grammar" in str(excinfo.value).lower()

    def test_gbnf_compiles_an_enum(self):
        grammar = schema_to_gbnf(enum_schema(["yes", "no"]))
        assert "root ::=" in grammar
        assert '"\\"yes\\""' in grammar
        assert "ws ::=" in grammar

    def test_gbnf_compiles_nested_objects(self):
        grammar = schema_to_gbnf(
            object_schema(
                {
                    "max_phase": {"type": "integer"},
                    "trials": {"type": "array", "items": {"type": "string"}},
                },
                required=["max_phase", "trials"],
            )
        )
        assert "integer" in grammar
        assert "root-trials" in grammar

    def test_gbnf_refuses_an_uncompilable_schema_loudly(self):
        """Silently skipping the grammar would weaken the guarantee invisibly."""
        with pytest.raises(ValueError, match="Union types"):
            schema_to_gbnf({"type": ["string", "number"]})
        with pytest.raises(ValueError, match="no properties"):
            schema_to_gbnf({"type": "object", "properties": {}})


# --------------------------------------------------------------------------- #
class TestRouting:
    def test_feature_vector_length_matches_names(self, classify_task):
        features = extract_features(classify_task)
        assert len(features.vector()) == len(FEATURE_NAMES)
        assert features["is_classify"] == 1.0
        assert features["bias"] == 1.0

    def test_heuristic_prefers_higher_accuracy(self, classify_task, pool_of_three):
        chosen = HeuristicRouter().select(classify_task, pool_of_three)
        assert chosen.model_id == "c"

    def test_warm_bonus_can_flip_the_choice(self, classify_task, pool_of_three):
        """The resource-aware term: this is what price-based routers lack."""
        router = HeuristicRouter(HeuristicWeights(warm_bonus=0.5, size_penalty=0.0))
        chosen = router.select(classify_task, pool_of_three, resident=["a"])
        assert chosen.model_id == "a"

    def test_unprofiled_model_is_penalised(self, classify_task, unprofiled_card, small_card):
        chosen = HeuristicRouter().select(classify_task, [unprofiled_card, small_card])
        assert chosen.model_id == "small"

    def test_static_router_reports_a_missing_model(self, classify_task, pool_of_three):
        from slmflow.errors import ModelNotFound

        with pytest.raises(ModelNotFound):
            StaticRouter("nonexistent").select(classify_task, pool_of_three)

    def test_learned_router_trains_and_selects(self, classify_task, pool_of_three):
        features = tuple(extract_features(classify_task).vector())
        examples = [
            TrainingExample(features, "a", 0.0),
            TrainingExample(features, "c", 1.0),
            TrainingExample(features, "a", 0.0),
            TrainingExample(features, "c", 1.0),
            TrainingExample(features, "b", 0.0),
            TrainingExample(features, "b", 1.0),
        ]
        router = train_router(examples, epochs=200, holdout_fraction=0.0)
        chosen = router.select(classify_task, pool_of_three)
        assert chosen.model_id in {"a", "b", "c"}
        assert router.report is not None
        assert router.report.n_models == 3

    def test_learned_router_needs_two_models(self):
        with pytest.raises(ValueError, match="at least 2 models"):
            train_router([TrainingExample((1.0,) * len(FEATURE_NAMES), "only", 1.0)] * 5)

    def test_learned_router_needs_enough_data(self):
        with pytest.raises(ValueError, match="at least 4"):
            train_router([TrainingExample((1.0,) * len(FEATURE_NAMES), "a", 1.0)])

    def test_learned_router_roundtrips(self, tmp_path, classify_task, pool_of_three):
        features = tuple(extract_features(classify_task).vector())
        examples = [
            TrainingExample(features, "a", 0.0),
            TrainingExample(features, "c", 1.0),
        ] * 3
        router = train_router(examples, epochs=50, holdout_fraction=0.0)
        path = tmp_path / "router.json"
        router.save(path)
        reloaded = LearnedRouter.load(path)
        assert reloaded.predict_scores(classify_task).keys() == router.predict_scores(
            classify_task
        ).keys()

    def test_learned_router_rejects_a_stale_feature_set(self, tmp_path):
        path = tmp_path / "stale.json"
        path.write_text(
            json.dumps(
                {
                    "feature_names": ["bias"],
                    "means": [0.0],
                    "stds": [1.0],
                    "weights": {"a": [0.0]},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="different feature set"):
            LearnedRouter.load(path)


# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_card_roundtrip_preserves_unmeasured_as_null(self, unprofiled_card):
        payload = card_to_dict(unprofiled_card)
        assert payload["cold_load_s"] is None
        restored = card_from_dict(payload)
        assert restored.cold_load_s is None
        assert not restored.is_profiled

    def test_card_rejects_a_non_numeric_cost(self):
        with pytest.raises(ConfigError, match="must be a number"):
            card_from_dict({"model_id": "m", "backend": "mock", "cold_load_s": "fast"})

    def test_card_requires_identity_fields(self):
        with pytest.raises(ConfigError, match="backend"):
            card_from_dict({"model_id": "m"})

    def test_within_budget_filters_by_tier(self, small_card, large_card):
        registry = ModelRegistry([small_card, large_card])
        fits = registry.within_budget(MemoryBudget.single_slot(1 * GB))
        assert [c.model_id for c in fits] == ["small"]

    def test_cascade_order_puts_unprofiled_last(self, small_card, unprofiled_card):
        registry = ModelRegistry([unprofiled_card, small_card])
        order = [c.model_id for c in registry.cascade_order(TaskType.CLASSIFY)]
        assert order[-1] == "unmeasured"

    def test_missing_cards_file_gives_an_actionable_error(self, tmp_path):
        with pytest.raises(ConfigError, match="profile_models"):
            ModelRegistry.from_file(tmp_path / "absent.json")


# --------------------------------------------------------------------------- #
class TestSLMMuxSelection:
    def test_selection_prefers_complementary_low_contradiction_sets(self):
        correct = {
            "a": [True, True, False, False],
            "b": [False, False, True, True],
            "c": [True, True, False, False],
        }
        answers = {
            "a": ["x", "x", "y", "y"],
            "b": ["y", "y", "x", "x"],
            "c": ["x", "x", "y", "y"],
        }
        scores = select_model_set(correct, answers, lambda_contradiction=0.0, max_size=2)
        best = scores[0]
        # a+b is complementary: union accuracy 1.0.
        assert best.union_accuracy == pytest.approx(1.0)
        assert set(best.models) == {"a", "b"}

    def test_contradiction_penalty_changes_the_winner(self):
        correct = {"a": [True, True], "b": [False, True]}
        answers = {"a": ["x", "y"], "b": ["z", "y"]}
        no_penalty = select_model_set(correct, answers, lambda_contradiction=0.0)
        heavy = select_model_set(correct, answers, lambda_contradiction=5.0)
        assert no_penalty[0].score >= heavy[0].score

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(ValueError, match="aligned"):
            select_model_set({"a": [True], "b": [True, False]}, {"a": ["x"], "b": ["x", "y"]})

    def test_over_budget_sets_are_flagged_not_hidden(self):
        correct = {"a": [True], "b": [True]}
        answers = {"a": ["x"], "b": ["x"]}
        scores = select_model_set(
            correct,
            answers,
            footprints={"a": 2 * GB, "b": 3 * GB},
            budget_bytes=4 * GB,
        )
        pair = next(s for s in scores if len(s.models) == 2)
        assert pair.peak_bytes == 5 * GB  # visible, so the swap cost is not hidden


# --------------------------------------------------------------------------- #
class TestPipelineEndToEnd:
    def _registry(self) -> ModelRegistry:
        pool = make_pool(
            [("small", 0.5, 400_000_000), ("large", 3.0, 2_200_000_000)],
            accuracy={
                "small": {"classify": 0.60, "qa": 0.52},
                "large": {"classify": 0.81, "qa": 0.74},
            },
            load_s={"small": 0.05, "large": 0.2},
        )
        registry = ModelRegistry(pool)
        registry.register_backend(
            "mock",
            MockBackend(
                {
                    "small": {"*": ScriptedResponse({"investigational": 0.7, "approved": 0.3})},
                    "large": {"*": ScriptedResponse({"investigational": 1.0})},
                },
                default_answer="investigational",
            ),
        )
        return registry

    def test_yaml_pipeline_over_two_models_produces_answer_and_trace(self, tmp_path):
        """Phase 5 Definition of Done, on the mock backend."""
        config = tmp_path / "pipe.yaml"
        config.write_text(
            """
name: two_model_demo
router: heuristic
scheduler: greedy
seed: 7
steps:
  - name: label
    task_type: classify
    prompt: Classify the evidence for {drug}.
    strategy: adaptive
    k: 4
    temperature: 0.7
    max_output_tokens: 16
""",
            encoding="utf-8",
        )
        trace_file = tmp_path / "trace.jsonl"
        pipeline = Pipeline.from_config(
            config,
            registry=self._registry(),
            budget=MemoryBudget.single_slot(2_500_000_000),
            options=PipelineOptions(trace_path=trace_file, seed=7),
        )
        with pipeline:
            result = pipeline.run({"drug": "metformin"})

        assert result.answer == "investigational"
        assert result.confidence is not None
        trace = result.trace
        assert trace.steps
        assert trace.total_swaps >= 1
        assert trace.swap_seconds > 0  # the cost other frameworks hide
        assert trace.total_generations >= 1
        assert trace.models_used
        assert trace.config_digest

        # The structured trace must be on disk and machine-readable.
        lines = [json.loads(line) for line in trace_file.read_text().splitlines() if line]
        assert lines
        assert lines[-1]["derived"]["total_swaps"] == trace.total_swaps

    def test_multi_step_dag_runs_in_dependency_order(self):
        pipeline = Pipeline.from_dict(
            {
                "name": "chain",
                "steps": [
                    {"name": "first", "prompt": "Step one", "task_type": "classify"},
                    {
                        "name": "second",
                        "prompt": "Step two",
                        "task_type": "qa",
                        "depends_on": ["first"],
                    },
                ],
            },
            registry=self._registry(),
            budget=MemoryBudget.single_slot(2_500_000_000),
        )
        with pipeline:
            result = pipeline.run()
        assert [s.step for s in result.trace.steps] == ["first", "second"]

    def test_concurrency_is_bounded_by_memory_not_cpu(self):
        """Two concurrent steps need two resident models; usually impossible."""
        pipeline = Pipeline.from_dict(
            {"name": "c", "steps": [{"name": "s", "prompt": "x"}]},
            registry=self._registry(),
            budget=MemoryBudget.single_slot(500_000_000),  # only 'small' fits
        )
        assert pipeline.concurrency_limit() == 1

    def test_wider_budget_allows_more_concurrency(self):
        pipeline = Pipeline.from_dict(
            {"name": "c", "steps": [{"name": "s", "prompt": "x"}]},
            registry=self._registry(),
            budget=MemoryBudget.single_slot(8 * GB),
        )
        assert pipeline.concurrency_limit() >= 2

    def test_no_model_fits_reports_the_arithmetic(self):
        from slmflow.errors import ScheduleInfeasible

        with pytest.raises(ScheduleInfeasible) as excinfo:
            Pipeline.from_dict(
                {"name": "c", "steps": [{"name": "s", "prompt": "x"}]},
                registry=self._registry(),
                budget=MemoryBudget.single_slot(1000),
            )
        assert "Model sizes" in str(excinfo.value)

    def test_plan_shows_the_residency_plan_without_running(self):
        pipeline = Pipeline.from_dict(
            {
                "name": "p",
                "steps": [
                    {"name": "a", "prompt": "x", "task_type": "classify"},
                    {"name": "b", "prompt": "y", "task_type": "qa"},
                ],
            },
            registry=self._registry(),
            budget=MemoryBudget.single_slot(2_500_000_000),
        )
        schedule = pipeline.plan()
        assert len(schedule) == 2
        assert schedule.predicted_loads >= 1
        assert schedule.predicted_seconds is not None

    def test_describe_carries_everything_a_result_needs(self):
        pipeline = Pipeline.from_dict(
            {"name": "p", "steps": [{"name": "s", "prompt": "x"}]},
            registry=self._registry(),
            budget=MemoryBudget.single_slot(2_500_000_000),
        )
        described = pipeline.describe()
        for key in (
            "config_digest",
            "device_tier",
            "model_digests",
            "budget",
            "unprofiled_models",
            "concurrency_limit",
        ):
            assert key in described

    def test_calibrate_tunes_the_declared_guarantee(self):
        from slmflow.reliability.calibration import Precision

        registry = self._registry()
        pipeline = Pipeline.from_dict(
            {
                "name": "cal",
                "steps": [
                    {
                        "name": "label",
                        "prompt": "Classify {drug}",
                        "task_type": "classify",
                        "strategy": "adaptive",
                        "k": 3,
                        "temperature": 0.7,
                    }
                ],
            },
            registry=registry,
            budget=MemoryBudget.single_slot(2_500_000_000),
            guarantee=Precision(0.5, delta=0.3),
        )
        items = [{"drug": f"drug{i}", "answer": "investigational"} for i in range(12)]
        with pipeline:
            policy = pipeline.calibrate(items)
        assert policy.n_calibration == 12
        assert 0.0 <= policy.achieved_coverage <= 1.0
        assert pipeline.threshold_policy is policy

    def test_calibrate_without_a_guarantee_is_an_error(self):
        pipeline = Pipeline.from_dict(
            {"name": "p", "steps": [{"name": "s", "prompt": "x"}]},
            registry=self._registry(),
            budget=MemoryBudget.single_slot(2_500_000_000),
        )
        with pytest.raises(ConfigError, match="precision target"):
            pipeline.calibrate([{"answer": "x"}])


class TestCLI:
    def test_probe_runs(self, capsys):
        from slmflow.cli import main

        assert main(["probe"]) == 0
        assert "Device tier" in capsys.readouterr().out

    def test_plan_with_mock_backend(self, tmp_path, capsys):
        from slmflow.cli import main

        config = tmp_path / "p.yaml"
        config.write_text(
            "name: cli_demo\nsteps:\n  - name: s\n    prompt: hello\n", encoding="utf-8"
        )
        assert main(["plan", str(config), "--mock"]) == 0
        out = capsys.readouterr().out
        assert "predicted loads" in out

    def test_run_with_mock_backend(self, tmp_path, capsys):
        from slmflow.cli import main

        config = tmp_path / "p.yaml"
        config.write_text(
            "name: cli_run\nsteps:\n  - name: s\n    prompt: hello\n", encoding="utf-8"
        )
        code = main(["run", str(config), "--mock", "--trace", str(tmp_path / "t.jsonl")])
        assert code in (0, 3)  # 3 means it abstained, which is a valid outcome
        out = capsys.readouterr().out
        assert "model loads" in out


@pytest.mark.real_model
class TestWithRealWeights:
    """Runs only where real model weights exist. Deselect with ``-m 'not real_model'``."""

    def test_real_backend_available(self):
        from slmflow.models.backends import available_backends

        status = available_backends()
        if not (status.get("llama_cpp") or status.get("ollama")):
            pytest.skip("no real backend on this machine")
        assert status["mock"] is True
