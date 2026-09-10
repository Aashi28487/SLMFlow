"""Tests for Phase 10 / C3: confidence signals, calibration, abstention, metrics."""

from __future__ import annotations

import math

import pytest

from slmflow.errors import GuaranteeUnmet
from slmflow.reliability.abstain import AbstentionPolicy
from slmflow.reliability.calibration import (
    CalibrationPoint,
    Precision,
    binomial_lower_bound,
    coverage_at_precision,
    regularised_incomplete_beta,
    risk_coverage_curve,
    tune_threshold,
)
from slmflow.reliability.confidence import (
    CombinedConfidence,
    CrossModelAgreement,
    LogProbMargin,
    RetrievalSupport,
    SelfConsistency,
    contradiction_rate,
    normalise_answer,
)
from slmflow.reliability.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    mcnemar,
    mean_std,
    paired_bootstrap,
)
from slmflow.strategies.adaptive import posterior_majority
from slmflow.types import AbstentionReason, Generation


def gen(text: str, model: str = "m1", margin: float | None = None) -> Generation:
    """Build a generation for a confidence test."""
    return Generation(text=text, model_id=model, top_token_margin=margin)


# --------------------------------------------------------------------------- #
class TestNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  The Investigational. ", "investigational"),
            ("APPROVED", "approved"),
            ("a hypothesis", "hypothesis"),
            ("yes -- investigational!", "yes investigational"),
        ],
    )
    def test_canonicalises_for_exact_match(self, raw, expected):
        assert normalise_answer(raw) == expected


class TestConfidenceSignals:
    def test_self_consistency_is_modal_share(self):
        gens = [gen("approved"), gen("approved"), gen("hypothesis")]
        assert SelfConsistency().score(gens, {}) == pytest.approx(2 / 3)

    def test_self_consistency_of_nothing_is_zero_not_one(self):
        assert SelfConsistency().score([], {}) == 0.0

    def test_logprob_margin_needs_no_repeated_sampling(self):
        """The only confidence signal that does not multiply cost by k."""
        high = LogProbMargin().score([gen("x", margin=6.0)], {})
        low = LogProbMargin().score([gen("x", margin=0.1)], {})
        assert high > low
        assert 0.0 <= low <= high <= 1.0

    def test_logprob_margin_absent_returns_zero_not_a_guess(self):
        assert LogProbMargin().score([gen("x", margin=None)], {}) == 0.0

    def test_cross_model_agreement_gives_each_model_one_vote(self):
        """A model sampled ten times must not outvote a model sampled once."""
        gens = [gen("a", "m1")] * 10 + [gen("b", "m2")]
        assert CrossModelAgreement().score(gens, {}) == pytest.approx(0.5)

    def test_cross_model_agreement_needs_two_models(self):
        assert CrossModelAgreement().score([gen("a", "m1")] * 5, {}) == 0.0

    def test_contradiction_rate_is_pairwise_disagreement(self):
        gens = [gen("a", "m1"), gen("b", "m2"), gen("a", "m3")]
        # Pairs: (a,b) disagree, (a,a) agree, (b,a) disagree -> 2/3
        assert contradiction_rate(gens) == pytest.approx(2 / 3)

    def test_retrieval_support_measures_groundedness(self):
        context = {"retrieved": ["metformin reduces cancer incidence in cohorts"]}
        grounded = RetrievalSupport().score([gen("metformin cancer incidence")], context)
        ungrounded = RetrievalSupport().score([gen("pembrolizumab melanoma survival")], context)
        assert grounded > ungrounded

    def test_retrieval_support_without_evidence_is_zero(self):
        assert RetrievalSupport().score([gen("anything")], {}) == 0.0

    def test_combined_reports_components_separately(self):
        """Phase 10 compares signals, so the combiner must not collapse them."""
        combined = CombinedConfidence.default()
        gens = [gen("a", margin=3.0), gen("a", margin=3.0)]
        parts = combined.components(gens, {})
        assert set(parts) == {"self_consistency", "logprob_margin"}
        assert combined.score(gens, {}) == pytest.approx(
            sum(parts.values()) / len(parts), abs=1e-9
        )


# --------------------------------------------------------------------------- #
class TestExactBinomial:
    def test_incomplete_beta_matches_known_values(self):
        # I_0.5(1,1) = 0.5 for the uniform case.
        assert regularised_incomplete_beta(0.5, 1, 1) == pytest.approx(0.5, abs=1e-6)
        assert regularised_incomplete_beta(0.0, 2, 3) == 0.0
        assert regularised_incomplete_beta(1.0, 2, 3) == 1.0

    def test_lower_bound_is_below_the_point_estimate(self):
        """A bound that exceeded the observed rate would be unsound."""
        assert binomial_lower_bound(9, 10, 0.1) < 0.9
        assert binomial_lower_bound(90, 100, 0.1) < 0.9

    def test_lower_bound_tightens_with_more_data(self):
        few = binomial_lower_bound(9, 10, 0.1)
        many = binomial_lower_bound(900, 1000, 0.1)
        assert many > few

    def test_perfect_record_is_still_not_certainty(self):
        bound = binomial_lower_bound(5, 5, 0.1)
        assert 0.0 < bound < 1.0

    def test_degenerate_inputs(self):
        assert binomial_lower_bound(0, 0, 0.1) == 0.0
        assert binomial_lower_bound(0, 10, 0.1) == 0.0


class TestThresholdTuning:
    def _points(self) -> list[CalibrationPoint]:
        """High confidence mostly correct; low confidence mostly wrong."""
        points = [CalibrationPoint(0.95, True) for _ in range(18)]
        points += [CalibrationPoint(0.95, False) for _ in range(1)]
        points += [CalibrationPoint(0.55, True) for _ in range(6)]
        points += [CalibrationPoint(0.55, False) for _ in range(15)]
        return points

    def test_tuning_meets_the_target_and_reports_coverage(self):
        policy = tune_threshold(self._points(), Precision(0.80, delta=0.2))
        assert policy.precision_lower_bound >= 0.80
        assert policy.achieved_precision >= 0.80
        # Coverage is the price of the guarantee, and must be reported.
        assert 0.0 < policy.achieved_coverage < 1.0
        assert policy.abstention_rate == pytest.approx(1 - policy.achieved_coverage)

    def test_tuning_maximises_coverage_subject_to_the_guarantee(self):
        """A threshold of 1.0 satisfies any target by answering nothing."""
        lenient = tune_threshold(self._points(), Precision(0.5, delta=0.2))
        strict = tune_threshold(self._points(), Precision(0.8, delta=0.2))
        assert lenient.achieved_coverage >= strict.achieved_coverage

    def test_unreachable_target_raises_with_the_best_achievable(self):
        with pytest.raises(GuaranteeUnmet) as excinfo:
            tune_threshold(self._points(), Precision(0.999, delta=0.01))
        assert 0.0 <= excinfo.value.best_achieved < 0.999
        assert "lower the target" in str(excinfo.value).lower()

    def test_no_calibration_data_raises(self):
        with pytest.raises(GuaranteeUnmet):
            tune_threshold([], Precision(0.9))

    def test_precision_rejects_impossible_targets(self):
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            Precision(1.5)
        with pytest.raises(ValueError, match="delta"):
            Precision(0.9, delta=0.0)

    def test_risk_coverage_curve_falls_as_coverage_falls(self):
        curve = risk_coverage_curve(self._points())
        assert curve
        # Coverage is monotonically non-increasing in threshold.
        coverages = [c for _, c, _ in curve]
        assert coverages == sorted(coverages, reverse=True)
        # A useful signal has lower risk at the strictest threshold.
        assert curve[-1][2] <= curve[0][2]

    def test_empirical_and_guaranteed_coverage_differ(self):
        """The gap between them is the price of the statistical guarantee."""
        points = self._points()
        empirical = coverage_at_precision(points, 0.80)
        guaranteed = tune_threshold(points, Precision(0.80, delta=0.2))
        assert empirical is not None
        assert empirical[1] >= guaranteed.achieved_coverage


class TestAbstentionPolicy:
    def test_low_confidence_abstains(self):
        decision = AbstentionPolicy(threshold=0.8).decide(0.6)
        assert decision.abstained
        assert decision.reason is AbstentionReason.LOW_CONFIDENCE

    def test_schema_failure_is_never_an_answer(self):
        decision = AbstentionPolicy(threshold=0.0).decide(1.0, schema_valid=False)
        assert decision.abstained
        assert decision.reason is AbstentionReason.SCHEMA_FAILURE

    def test_contradiction_abstains_despite_confidence(self):
        gens = [gen("a", "m1"), gen("b", "m2")]
        policy = AbstentionPolicy(threshold=0.0, max_contradiction=0.4)
        decision = policy.decide(1.0, generations=gens)
        assert decision.abstained
        assert decision.reason is AbstentionReason.CONTRADICTION

    def test_missing_retrieval_support_abstains(self):
        policy = AbstentionPolicy(threshold=0.0, require_retrieval_support=0.5)
        decision = policy.decide(1.0, signals={"retrieval_support": 0.1})
        assert decision.abstained
        assert decision.reason is AbstentionReason.NO_RETRIEVAL_SUPPORT

    def test_confident_grounded_answer_is_returned(self):
        policy = AbstentionPolicy(threshold=0.5, require_retrieval_support=0.3)
        decision = policy.decide(0.9, signals={"retrieval_support": 0.8})
        assert decision.answer
        assert decision.reason is None

    def test_tuned_policy_overrides_the_static_threshold(self):
        tuned = tune_threshold(
            [CalibrationPoint(0.9, True)] * 10 + [CalibrationPoint(0.2, False)] * 10,
            Precision(0.7, delta=0.2),
        )
        policy = AbstentionPolicy(threshold=0.01, policy=tuned)
        assert policy.effective_threshold == tuned.threshold

    def test_abstention_improves_precision_on_the_answered_subset(self):
        """Phase 10 DoD: abstaining must demonstrably raise precision."""
        points = [CalibrationPoint(0.9, True) for _ in range(15)]
        points += [CalibrationPoint(0.9, False) for _ in range(2)]
        points += [CalibrationPoint(0.3, True) for _ in range(4)]
        points += [CalibrationPoint(0.3, False) for _ in range(12)]

        never_abstain = sum(1 for p in points if p.correct) / len(points)
        policy = tune_threshold(points, Precision(0.7, delta=0.2))
        answered = [p for p in points if p.confidence >= policy.threshold]
        with_abstention = sum(1 for p in answered if p.correct) / len(answered)
        assert with_abstention > never_abstain


# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_ece_is_the_confidence_accuracy_gap(self):
        ece, bins = expected_calibration_error([0.9, 0.9], [True, True])
        assert ece == pytest.approx(0.1, abs=1e-9)
        rows = bins.as_rows()
        assert len(rows) == 1
        assert rows[0]["n"] == 2

    def test_perfect_calibration_scores_zero(self):
        # Half correct at confidence 0.5 is perfectly calibrated.
        ece, _ = expected_calibration_error([0.5] * 4, [True, True, False, False])
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_ece_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="differ in length"):
            expected_calibration_error([0.5], [True, False])

    def test_brier_penalises_overconfident_errors(self):
        assert brier_score([1.0], [False]) == pytest.approx(1.0)
        assert brier_score([0.0], [False]) == pytest.approx(0.0)

    def test_mean_std_handles_a_single_seed(self):
        assert mean_std([0.5]) == (0.5, 0.0)
        assert mean_std([]) == (0.0, 0.0)
        mean, std = mean_std([0.4, 0.5, 0.6])
        assert mean == pytest.approx(0.5)
        assert std > 0

    def test_accuracy_normalises_before_comparing(self):
        assert accuracy(["The Approved."], ["approved"]) == 1.0

    def test_paired_bootstrap_detects_a_real_difference(self):
        a = [1.0] * 40 + [0.0] * 10
        b = [0.0] * 40 + [0.0] * 10
        result = paired_bootstrap(a, b, n_resamples=800, seed=0)
        assert result.observed_difference > 0
        assert result.significant
        assert "significant" in result.sentence("a", "b")

    def test_paired_bootstrap_finds_no_difference_when_identical(self):
        scores = [1.0, 0.0] * 20
        result = paired_bootstrap(scores, list(scores), n_resamples=500, seed=0)
        assert result.observed_difference == pytest.approx(0.0)
        assert not result.significant

    def test_paired_bootstrap_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal lengths"):
            paired_bootstrap([1.0], [1.0, 0.0])

    def test_mcnemar_uses_exact_test_on_few_discordant_pairs(self):
        a = [True] * 5 + [False] * 5
        b = [False] * 5 + [False] * 5
        result = mcnemar(a, b)
        assert result.exact
        assert result.b == 5
        assert result.c == 0
        assert result.p_value < 0.1

    def test_mcnemar_uses_chi_square_when_pairs_are_many(self):
        a = [True] * 30 + [False] * 30
        b = [False] * 30 + [True] * 5 + [False] * 25
        result = mcnemar(a, b)
        assert not result.exact
        assert 0.0 <= result.p_value <= 1.0

    def test_mcnemar_with_no_disagreement_is_not_significant(self):
        outcomes = [True, False, True]
        result = mcnemar(outcomes, list(outcomes))
        assert result.p_value == 1.0
        assert not result.significant


class TestAdaptivePosterior:
    def test_posterior_rises_with_agreement(self):
        assert posterior_majority(3, 3) > posterior_majority(2, 3)
        assert posterior_majority(2, 3) > posterior_majority(1, 3)

    def test_no_samples_is_maximum_ignorance(self):
        assert posterior_majority(0, 0) == 0.5

    def test_unanimity_needs_several_samples_to_be_confident(self):
        """Two identical samples are weak evidence; the rule must know that."""
        assert posterior_majority(2, 2) < 0.95
        assert posterior_majority(6, 6) > 0.95

    def test_posterior_is_a_probability(self):
        for agree in range(6):
            value = posterior_majority(agree, 5)
            assert 0.0 <= value <= 1.0
            assert not math.isnan(value)
