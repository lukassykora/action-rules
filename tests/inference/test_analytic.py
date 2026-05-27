"""Tests for AnalyticEngine (Wald normal-approximation CI)."""

import math
from math import sqrt

import pandas as pd
import pytest
from scipy.stats import norm

from action_rules.inference.analytic import AnalyticEngine
from action_rules.inference.base import ConfidenceIntervalResult, RuleCategory, RuleMasks, compute_group_counts

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataset: two attributes, one target.

    Groups
    ------
    age=0, class=0 -> target=0 (n//4 rows)
    age=0, class=1 -> target=1 (n//4 rows)
    age=1, class=0 -> target=0 (n//4 rows)
    age=1, class=1 -> target=1 (n//4 rows)

    The rule "age stays at 0, class changes 0->1, target changes 0->1"
    has perfect confidence on both sides in this dataset.
    """
    n_each = n // 4
    g1 = pd.DataFrame({'age': ['0'] * n_each, 'class': ['0'] * n_each, 'target': ['0'] * n_each})
    g2 = pd.DataFrame({'age': ['0'] * n_each, 'class': ['1'] * n_each, 'target': ['1'] * n_each})
    g3 = pd.DataFrame({'age': ['1'] * n_each, 'class': ['0'] * n_each, 'target': ['0'] * n_each})
    g4 = pd.DataFrame({'age': ['1'] * n_each, 'class': ['1'] * n_each, 'target': ['1'] * n_each})
    data = pd.concat([g1, g2, g3, g4], ignore_index=True)
    return data.sample(frac=1, random_state=seed).reset_index(drop=True)


def _make_rule(rule_index: int = 0) -> RuleMasks:
    """RuleMasks for: age=0 stable, class 0->1, target 0->1."""
    return RuleMasks(
        mask_undesired={'age': '0', 'class': '0'},
        mask_desired={'age': '0', 'class': '1'},
        target_attribute='target',
        target_undesired='0',
        target_desired='1',
        rule_index=rule_index,
        undesired_itemset=(0, 1),
        desired_itemset=(0, 2),
    )


def _make_column_values() -> dict:
    """Integer-index -> (attribute, value) mapping matching _make_rule."""
    return {
        0: ('age', '0'),
        1: ('class', '0'),
        2: ('class', '1'),
        3: ('target', '0'),
        4: ('target', '1'),
    }


def _make_utility_tables():
    """Simple utility tables with known numeric values."""
    intrinsic = {
        ('class', '0'): -1.0,
        ('class', '1'): 1.0,
        ('target', '0'): -2.0,
        ('target', '1'): 2.0,
    }
    transition = {
        ('class', '0', '1'): 0.5,
    }
    return intrinsic, transition


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------


class TestAnalyticEngineBasic:
    """Fundamental correctness checks."""

    def test_returns_list_of_correct_length(self):
        """One result per rule."""
        data = _make_data()
        engine = AnalyticEngine()
        results = engine.compute(data, [_make_rule(0), _make_rule(1)])
        assert len(results) == 2

    def test_result_type(self):
        """Each element is a ConfidenceIntervalResult."""
        data = _make_data()
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert isinstance(result, ConfidenceIntervalResult)

    def test_method_field(self):
        """The method field must be 'analytic'."""
        data = _make_data()
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert result.method == 'analytic'

    def test_rule_index_preserved(self):
        """rule_index from RuleMasks is carried through."""
        data = _make_data()
        result = AnalyticEngine().compute(data, [_make_rule(rule_index=9)])[0]
        assert result.rule_index == 9

    def test_confidence_level_stored(self):
        """confidence_level passed to compute() is stored on the result."""
        data = _make_data()
        result = AnalyticEngine().compute(data, [_make_rule()], confidence_level=0.90)[0]
        assert result.confidence_level == pytest.approx(0.90)

    def test_empty_rules_list(self):
        """An empty rules list returns an empty list without error."""
        data = _make_data()
        assert AnalyticEngine().compute(data, []) == []

    def test_no_samples_stored(self):
        """AnalyticEngine never stores sample arrays."""
        data = _make_data()
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert result.samples_uplift is None
        assert result.samples_gain is None


# ---------------------------------------------------------------------------
# Known-value test (hand-computed SE)
# ---------------------------------------------------------------------------


class TestAnalyticKnownValues:
    """Verify the formula against hand-computed expected values."""

    def test_perfect_confidence_uplift_and_se(self):
        """With perfect confidences p_u = p_d = 1, var terms collapse to zero.

        For the synthetic dataset (n_each = 50, n = 200):
            n_u_ante = 50, n_u_match = 50  -> p_u = 1.0
            n_d_ante = 50, n_d_match = 50  -> p_d = 1.0
            d = p_d + p_u - 1 = 1.0
            uplift = d * n_u_ante / n_total = 1.0 * 50 / 200 = 0.25
            var_p_u = 1*(1-1)/50 = 0.0
            var_p_d = 0.0
            se_uplift = 0.0
        """
        data = _make_data(n=200)
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert result.uplift_point == pytest.approx(0.25)
        assert result.uplift_se == pytest.approx(0.0)
        # With se=0 the CI collapses to the point estimate.
        assert result.uplift_ci_lower == pytest.approx(0.25)
        assert result.uplift_ci_upper == pytest.approx(0.25)

    def test_partial_confidence_known_se(self):
        """Test with controlled partial confidences, checking SE formula.

        Build a dataset where:
            mask_undesired rows: 80 total, 60 match target=0   -> p_u = 0.75
            mask_desired  rows: 40 total, 30 match target=1   -> p_d = 0.75
            n_total = 200

        Expected:
            d = 0.75 + 0.75 - 1 = 0.5
            uplift = 0.5 * 80 / 200 = 0.2
            var_p_u = 0.75 * 0.25 / 80 = 0.00234375
            var_p_d = 0.75 * 0.25 / 40 = 0.0046875
            var_d   = 0.00703125
            scale   = 80 / 200 = 0.4
            var_uplift = 0.16 * 0.00703125 = 0.001125
            se_uplift  = sqrt(0.001125) ≈ 0.033541...
        """
        # age=0, class=0, target=0: 60 rows
        g1 = pd.DataFrame({'age': ['0'] * 60, 'class': ['0'] * 60, 'target': ['0'] * 60})
        # age=0, class=0, target=1: 20 rows (antecedent but wrong target)
        g2 = pd.DataFrame({'age': ['0'] * 20, 'class': ['0'] * 20, 'target': ['1'] * 20})
        # age=0, class=1, target=1: 30 rows
        g3 = pd.DataFrame({'age': ['0'] * 30, 'class': ['1'] * 30, 'target': ['1'] * 30})
        # age=0, class=1, target=0: 10 rows (antecedent but wrong target)
        g4 = pd.DataFrame({'age': ['0'] * 10, 'class': ['1'] * 10, 'target': ['0'] * 10})
        # Filler to reach n_total = 200 (age=1 rows don't affect the rule masks).
        g5 = pd.DataFrame({'age': ['1'] * 80, 'class': ['0'] * 80, 'target': ['0'] * 80})
        data = pd.concat([g1, g2, g3, g4, g5], ignore_index=True)

        result = AnalyticEngine().compute(data, [_make_rule()])[0]

        expected_uplift = 0.2
        expected_se = sqrt(0.001125)

        assert result.uplift_point == pytest.approx(expected_uplift, rel=1e-9)
        assert result.uplift_se == pytest.approx(expected_se, rel=1e-6)

        z95 = norm.ppf(0.975)
        assert result.uplift_ci_lower == pytest.approx(expected_uplift - z95 * expected_se, rel=1e-6)
        assert result.uplift_ci_upper == pytest.approx(expected_uplift + z95 * expected_se, rel=1e-6)


# ---------------------------------------------------------------------------
# CI structural properties
# ---------------------------------------------------------------------------


class TestAnalyticCIProperties:
    """Structural invariants of the confidence interval."""

    def test_ci_symmetric_around_point_estimate(self):
        """The Wald CI is symmetric: point - lower == upper - point."""
        data = _make_data(n=400)
        # Introduce imperfect confidences with mixed outcome data.
        extra = pd.DataFrame({'age': ['0'] * 20, 'class': ['0'] * 20, 'target': ['1'] * 20})
        mixed = pd.concat([data, extra], ignore_index=True)
        result = AnalyticEngine().compute(mixed, [_make_rule()])[0]
        half_width_left = result.uplift_point - result.uplift_ci_lower
        half_width_right = result.uplift_ci_upper - result.uplift_point
        assert half_width_left == pytest.approx(half_width_right, rel=1e-9)

    def test_ci_lower_le_upper(self):
        """lower bound must be <= upper bound."""
        data = _make_data()
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert result.uplift_ci_lower <= result.uplift_ci_upper

    def test_se_non_negative(self):
        """SE must be >= 0."""
        data = _make_data()
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert result.uplift_se >= 0.0

    def test_wider_ci_at_higher_coverage(self):
        """99 % CI must be at least as wide as 95 % CI on the same data."""
        # Use imperfect data so CI is non-degenerate.
        data = _make_data(n=200)
        extra = pd.DataFrame({'age': ['0'] * 10, 'class': ['0'] * 10, 'target': ['1'] * 10})
        mixed = pd.concat([data, extra], ignore_index=True)
        engine = AnalyticEngine()
        r95 = engine.compute(mixed, [_make_rule()], confidence_level=0.95)[0]
        r99 = engine.compute(mixed, [_make_rule()], confidence_level=0.99)[0]
        width95 = r95.uplift_ci_upper - r95.uplift_ci_lower
        width99 = r99.uplift_ci_upper - r99.uplift_ci_lower
        assert width99 >= width95


# ---------------------------------------------------------------------------
# Determinism (no randomness)
# ---------------------------------------------------------------------------


class TestAnalyticDeterminism:
    """AnalyticEngine must produce identical results on repeated calls."""

    def test_same_data_same_result(self):
        """Two calls with identical inputs produce bit-for-bit equal results."""
        data = _make_data()
        engine = AnalyticEngine()
        r1 = engine.compute(data, [_make_rule()])[0]
        r2 = engine.compute(data, [_make_rule()])[0]
        assert r1.uplift_point == r2.uplift_point
        assert r1.uplift_ci_lower == r2.uplift_ci_lower
        assert r1.uplift_ci_upper == r2.uplift_ci_upper
        assert r1.uplift_se == r2.uplift_se

    def test_different_engine_instances_same_result(self):
        """Two separate AnalyticEngine instances give the same answer."""
        data = _make_data()
        r1 = AnalyticEngine().compute(data, [_make_rule()])[0]
        r2 = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert r1.uplift_point == r2.uplift_point
        assert r1.uplift_ci_lower == r2.uplift_ci_lower


# ---------------------------------------------------------------------------
# Zero-support edge case
# ---------------------------------------------------------------------------


class TestZeroSupportEdgeCase:
    """Rules with no matching rows must not raise and must return NaN."""

    def _zero_rule(self) -> RuleMasks:
        return RuleMasks(
            mask_undesired={'age': 'MISSING', 'class': '0'},
            mask_desired={'age': 'MISSING', 'class': '1'},
            target_attribute='target',
            target_undesired='0',
            target_desired='1',
            rule_index=0,
            undesired_itemset=(99,),
            desired_itemset=(100,),
        )

    def test_no_undesired_antecedent_returns_nan(self):
        """Zero n_u_ante -> all CI fields are NaN."""
        data = _make_data()
        result = AnalyticEngine().compute(data, [self._zero_rule()])[0]
        assert math.isnan(result.uplift_point)
        assert math.isnan(result.uplift_ci_lower)
        assert math.isnan(result.uplift_ci_upper)
        assert math.isnan(result.uplift_se)

    def test_zero_support_category_is_none(self):
        """Category cannot be determined when support is zero."""
        data = _make_data()
        result = AnalyticEngine().compute(data, [self._zero_rule()])[0]
        assert result.category is None

    def test_zero_support_with_gain_requested_returns_nan(self):
        """Gain fields are NaN (not None) when support is zero but gain is requested."""
        data = _make_data()
        intrinsic, transition = _make_utility_tables()
        cv = _make_column_values()
        result = AnalyticEngine().compute(
            data,
            [self._zero_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert math.isnan(result.realistic_rule_gain_point)
        assert math.isnan(result.realistic_rule_gain_ci_lower)
        assert math.isnan(result.realistic_rule_gain_ci_upper)
        assert math.isnan(result.realistic_rule_gain_se)


# ---------------------------------------------------------------------------
# Without utility tables — gain fields must be None
# ---------------------------------------------------------------------------


class TestNoUtilityTables:
    """Gain fields are None when no utility tables are supplied."""

    def test_gain_fields_are_none(self):
        data = _make_data()
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert result.realistic_rule_gain_point is None
        assert result.realistic_rule_gain_ci_lower is None
        assert result.realistic_rule_gain_ci_upper is None
        assert result.realistic_rule_gain_se is None

    def test_samples_gain_is_none(self):
        data = _make_data()
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert result.samples_gain is None


# ---------------------------------------------------------------------------
# With utility tables
# ---------------------------------------------------------------------------


class TestWithUtilityTables:
    """Gain statistics are populated when utility tables are provided."""

    def test_gain_fields_not_none(self):
        """All gain fields are populated (not None) when tables are given."""
        data = _make_data(n=200)
        intrinsic, transition = _make_utility_tables()
        cv = _make_column_values()
        result = AnalyticEngine().compute(
            data,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_point is not None
        assert result.realistic_rule_gain_ci_lower is not None
        assert result.realistic_rule_gain_ci_upper is not None
        assert result.realistic_rule_gain_se is not None

    def test_gain_ci_lower_le_upper(self):
        """Gain CI lower must be <= upper."""
        data = _make_data(n=200)
        # Use imperfect data so variance is non-zero.
        extra = pd.DataFrame({'age': ['0'] * 10, 'class': ['0'] * 10, 'target': ['1'] * 10})
        mixed = pd.concat([data, extra], ignore_index=True)
        intrinsic, transition = _make_utility_tables()
        cv = _make_column_values()
        result = AnalyticEngine().compute(
            mixed,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_ci_lower <= result.realistic_rule_gain_ci_upper

    def test_gain_ci_symmetric_around_point(self):
        """Gain CI is symmetric around the gain point estimate."""
        data = _make_data(n=200)
        extra = pd.DataFrame({'age': ['0'] * 10, 'class': ['0'] * 10, 'target': ['1'] * 10})
        mixed = pd.concat([data, extra], ignore_index=True)
        intrinsic, transition = _make_utility_tables()
        cv = _make_column_values()
        result = AnalyticEngine().compute(
            mixed,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        half_left = result.realistic_rule_gain_point - result.realistic_rule_gain_ci_lower
        half_right = result.realistic_rule_gain_ci_upper - result.realistic_rule_gain_point
        assert half_left == pytest.approx(half_right, rel=1e-9)

    def test_gain_se_non_negative(self):
        """Gain SE must be >= 0."""
        data = _make_data(n=200)
        extra = pd.DataFrame({'age': ['0'] * 10, 'class': ['0'] * 10, 'target': ['1'] * 10})
        mixed = pd.concat([data, extra], ignore_index=True)
        intrinsic, transition = _make_utility_tables()
        cv = _make_column_values()
        result = AnalyticEngine().compute(
            mixed,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_se >= 0.0

    def test_gain_se_zero_when_target_gain_zero(self):
        """When target_gain = 0 the gain SE is zero regardless of var_d."""
        # Utility table with zero net target gain: u_desired - u_undesired + trans = 0.
        intrinsic = {
            ('target', '0'): 1.0,
            ('target', '1'): 1.0,  # same -> target_gain = 0 + 0 = 0
        }
        transition: dict = {}
        data = _make_data(n=200)
        extra = pd.DataFrame({'age': ['0'] * 10, 'class': ['0'] * 10, 'target': ['1'] * 10})
        mixed = pd.concat([data, extra], ignore_index=True)
        cv = _make_column_values()
        result = AnalyticEngine().compute(
            mixed,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_se == pytest.approx(0.0)

    def test_gain_known_value(self):
        """Gain point estimate matches hand-computed value for perfect-confidence data.

        With p_u = p_d = 1.0 and the synthetic utility tables:
            rule_gain   = (u_desired(class=1) - u_desired(class=0)) + trans(class, 0->1)
                        = (1.0 - (-1.0)) + 0.5 = 2.5
            d           = 1.0 + 1.0 - 1 = 1.0
            target_gain = u(target=1) - u(target=0) + trans(target) = 2.0 - (-2.0) + 0.0 = 4.0
            gain        = 2.5 + 1.0 * 4.0 = 6.5
        """
        data = _make_data(n=200)
        intrinsic, transition = _make_utility_tables()
        cv = _make_column_values()
        result = AnalyticEngine().compute(
            data,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_point == pytest.approx(6.5)

    def test_gain_deterministic(self):
        """Repeated analytic gain calls return identical values."""
        data = _make_data(n=200)
        extra = pd.DataFrame({'age': ['0'] * 10, 'class': ['0'] * 10, 'target': ['1'] * 10})
        mixed = pd.concat([data, extra], ignore_index=True)
        intrinsic, transition = _make_utility_tables()
        cv = _make_column_values()
        engine = AnalyticEngine()
        r1 = engine.compute(
            mixed,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        r2 = engine.compute(
            mixed,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert r1.realistic_rule_gain_point == r2.realistic_rule_gain_point
        assert r1.realistic_rule_gain_ci_lower == r2.realistic_rule_gain_ci_lower
        assert r1.realistic_rule_gain_ci_upper == r2.realistic_rule_gain_ci_upper


# ---------------------------------------------------------------------------
# Support and confidence fields
# ---------------------------------------------------------------------------


class TestSupportAndConfidence:
    """Support and confidence should mirror the full-dataset counts."""

    def test_support_equals_undesired_antecedent_count(self):
        data = _make_data(n=200)
        rule = _make_rule()
        result = AnalyticEngine().compute(data, [rule])[0]
        n_u_ante, _, _, _, _ = compute_group_counts(data, rule)
        assert result.support == n_u_ante

    def test_confidence_matches_full_data(self):
        data = _make_data(n=200)
        rule = _make_rule()
        result = AnalyticEngine().compute(data, [rule])[0]
        n_u_ante, n_u_match, _, _, _ = compute_group_counts(data, rule)
        expected_conf = n_u_match / n_u_ante if n_u_ante > 0 else 0.0
        assert result.confidence == pytest.approx(expected_conf)


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------


class TestCategorisation:
    """Category is derived from the CI vs. zero threshold."""

    def test_positive_uplift_accepted(self):
        """Clearly positive uplift CI -> ACCEPT."""
        data = _make_data(n=400)
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert result.category == RuleCategory.ACCEPT

    def test_negative_uplift_rejected(self):
        """Clearly negative uplift CI -> REJECT."""
        n = 50
        g1 = pd.DataFrame({'age': ['0'] * n, 'class': ['0'] * n, 'target': ['1'] * n})
        g2 = pd.DataFrame({'age': ['0'] * n, 'class': ['1'] * n, 'target': ['0'] * n})
        data = pd.concat([g1, g2], ignore_index=True)
        result = AnalyticEngine().compute(data, [_make_rule()])[0]
        assert result.category == RuleCategory.REJECT


# ---------------------------------------------------------------------------
# Multiple rules
# ---------------------------------------------------------------------------


class TestMultipleRules:
    """Multiple rules processed in one call."""

    def test_order_preserved(self):
        data = _make_data()
        rules = [_make_rule(rule_index=0), _make_rule(rule_index=5)]
        results = AnalyticEngine().compute(data, rules)
        assert results[0].rule_index == 0
        assert results[1].rule_index == 5

    def test_independent_results(self):
        """Two identical rules in one call both produce valid outputs."""
        data = _make_data()
        results = AnalyticEngine().compute(data, [_make_rule(), _make_rule()])
        for r in results:
            assert not math.isnan(r.uplift_point)


# ---------------------------------------------------------------------------
# Wilson Score Interval tests
# ---------------------------------------------------------------------------


class TestWilsonScoreInterval:
    """Tests for the Wilson score interval (analytic_type='wilson')."""

    def test_invalid_analytic_type_raises(self):
        """Unknown analytic_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown analytic_type"):
            AnalyticEngine(analytic_type="invalid")

    def test_default_is_wald(self):
        """AnalyticEngine() with no args uses Wald (backward compatibility)."""
        engine = AnalyticEngine()
        assert engine.analytic_type == "wald"

    def test_wilson_alias_canonicalises_to_newcombe_wilson(self):
        """'wilson' is accepted and canonicalised to 'newcombe_wilson'."""
        engine = AnalyticEngine(analytic_type="wilson")
        assert engine.analytic_type == "newcombe_wilson"

    def test_newcombe_wilson_and_wilson_alias_produce_identical_results(self):
        """The 'wilson' alias must produce bit-for-bit identical CI values
        to 'newcombe_wilson'."""
        data = _make_data()
        nw = AnalyticEngine(analytic_type="newcombe_wilson").compute(data, [_make_rule()])[0]
        wn = AnalyticEngine(analytic_type="wilson").compute(data, [_make_rule()])[0]
        assert nw.uplift_point == wn.uplift_point
        assert nw.uplift_ci_lower == wn.uplift_ci_lower
        assert nw.uplift_ci_upper == wn.uplift_ci_upper

    def test_wilson_returns_valid_results(self):
        """Wilson produces non-NaN results for well-supported rules."""
        data = _make_data()
        result = AnalyticEngine(analytic_type="wilson").compute(data, [_make_rule()])[0]
        assert not math.isnan(result.uplift_point)
        assert not math.isnan(result.uplift_se)
        assert result.uplift_ci_lower <= result.uplift_ci_upper

    def test_wilson_method_field_is_analytic(self):
        """Wilson still reports method='analytic'."""
        data = _make_data()
        result = AnalyticEngine(analytic_type="wilson").compute(data, [_make_rule()])[0]
        assert result.method == "analytic"

    def test_wilson_matches_wald_point_but_differs_in_bounds(self):
        """Point estimate uses p_hat for both methods; the bounds differ at small n.

        With Newcombe-Wilson the point estimate is the MLE plug-in, identical to
        Wald.  The intervals differ in width and (importantly) in symmetry:
        Newcombe-Wilson is asymmetric, Wald is symmetric.
        """
        # Imperfect confidence both sides so neither bound saturates at 0/1.
        g1 = pd.DataFrame({'age': ['0'] * 9, 'class': ['0'] * 9, 'target': ['0'] * 9})
        g2 = pd.DataFrame({'age': ['0'] * 3, 'class': ['0'] * 3, 'target': ['1'] * 3})
        g3 = pd.DataFrame({'age': ['0'] * 7, 'class': ['1'] * 7, 'target': ['1'] * 7})
        g4 = pd.DataFrame({'age': ['0'] * 2, 'class': ['1'] * 2, 'target': ['0'] * 2})
        g5 = pd.DataFrame({'age': ['1'] * 10, 'class': ['0'] * 10, 'target': ['0'] * 10})
        data = pd.concat([g1, g2, g3, g4, g5], ignore_index=True)

        wald = AnalyticEngine(analytic_type="wald").compute(data, [_make_rule()])[0]
        wilson = AnalyticEngine(analytic_type="wilson").compute(data, [_make_rule()])[0]

        # Point estimates match (both use MLE plug-in).
        assert wald.uplift_point == pytest.approx(wilson.uplift_point, rel=1e-9)
        # CI bounds differ at small n.
        assert wald.uplift_ci_lower != pytest.approx(wilson.uplift_ci_lower, abs=1e-6)
        assert wald.uplift_ci_upper != pytest.approx(wilson.uplift_ci_upper, abs=1e-6)

    def test_wilson_converges_to_wald_large_n(self):
        """For large n with moderate p, Wilson and Wald should be very close."""
        data = _make_data(n=4000)
        # Add some imperfect rows to avoid p=1.
        extra = pd.DataFrame({'age': ['0'] * 100, 'class': ['0'] * 100, 'target': ['1'] * 100})
        big_data = pd.concat([data, extra], ignore_index=True)

        wald = AnalyticEngine(analytic_type="wald").compute(big_data, [_make_rule()])[0]
        wilson = AnalyticEngine(analytic_type="wilson").compute(big_data, [_make_rule()])[0]

        assert wald.uplift_point == pytest.approx(wilson.uplift_point, rel=0.02)
        assert wald.uplift_se == pytest.approx(wilson.uplift_se, rel=0.05)

    def test_newcombe_wilson_ci_asymmetric(self):
        """Newcombe-Wilson CI is asymmetric around the MLE point estimate."""
        g1 = pd.DataFrame({'age': ['0'] * 30, 'class': ['0'] * 30, 'target': ['0'] * 30})
        g2 = pd.DataFrame({'age': ['0'] * 10, 'class': ['0'] * 10, 'target': ['1'] * 10})
        g3 = pd.DataFrame({'age': ['0'] * 20, 'class': ['1'] * 20, 'target': ['1'] * 20})
        g4 = pd.DataFrame({'age': ['1'] * 40, 'class': ['0'] * 40, 'target': ['0'] * 40})
        data = pd.concat([g1, g2, g3, g4], ignore_index=True)
        result = AnalyticEngine(analytic_type="wilson").compute(data, [_make_rule()])[0]
        left = result.uplift_point - result.uplift_ci_lower
        right = result.uplift_ci_upper - result.uplift_point
        # Asymmetric in general (boundary effects); only equal when p_u = p_d = 0.5.
        assert left != pytest.approx(right, rel=1e-3)

    def test_wilson_bounds_known_formula(self):
        """Verify _wilson_bounds against the closed-form Wilson interval.

        For x=15, n=20, z=1.96:
            p_hat = 0.75
            centre = 0.75 + 1.96^2/(2*20) = 0.75 + 0.0960 = 0.8460/1.1921 (after denom)
            denom  = 1 + 1.96^2/20 = 1.1921
            radius = 1.96 * sqrt(0.75*0.25/20 + 1.96^2/(4*400))
        """
        p_hat, l, u = AnalyticEngine._wilson_bounds(15, 20, 1.96)
        assert p_hat == pytest.approx(0.75, rel=1e-9)
        # Lower < p_hat < Upper (Wilson is shifted but still contains the MLE
        # for non-degenerate counts).
        assert l < p_hat < u
        # Both bounds in [0, 1].
        assert 0.0 <= l <= 1.0
        assert 0.0 <= u <= 1.0
        # Hand check (closed-form Wilson for x=15, n=20, z=1.96):
        #     centre_num = 0.75 + 1.96^2/40       = 0.84604
        #     radius_num = 1.96 * sqrt(0.009375 + 0.0024010) = 0.21270
        #     denom      = 1 + 1.96^2/20          = 1.19208
        #     lower      = (0.84604 - 0.21270) / 1.19208 = 0.5313
        #     upper      = (0.84604 + 0.21270) / 1.19208 = 0.8881
        assert l == pytest.approx(0.5313, abs=1e-3)
        assert u == pytest.approx(0.8881, abs=1e-3)

    def test_newcombe_delta_interval_known(self):
        """Newcombe combination on two known Wilson intervals.

        With p_u = p_d = 0.75 from x=15, n=20:
            delta_hat = 0.5
            Each side contributes about (0.75 - 0.5360) = 0.2140 below and
            (0.8843 - 0.75) = 0.1343 above the MLE.
            L_delta = 0.5 - sqrt(2 * 0.2140^2) = 0.5 - 0.3026 = 0.1974
            U_delta = 0.5 + sqrt(2 * 0.1343^2) = 0.5 + 0.1899 = 0.6899
        """
        p_hat, l, u = AnalyticEngine._wilson_bounds(15, 20, 1.96)
        delta_hat, lo, hi = AnalyticEngine._newcombe_delta_interval(
            p_hat, l, u, p_hat, l, u
        )
        assert delta_hat == pytest.approx(0.5, rel=1e-9)
        assert lo == pytest.approx(0.5 - math.sqrt(2) * (p_hat - l), rel=1e-9)
        assert hi == pytest.approx(0.5 + math.sqrt(2) * (u - p_hat), rel=1e-9)
        assert lo < hi

    def test_wilson_gain_point_matches_wald(self):
        """Gain point estimate uses MLE proportions for both methods.

        Only the *interval* differs between Wald and Newcombe-Wilson.  The
        point estimate is computed with `compute_realistic_gain` from the
        raw confidences regardless of the branch.
        """
        g1 = pd.DataFrame({'age': ['0'] * 10, 'class': ['0'] * 10, 'target': ['0'] * 10})
        g2 = pd.DataFrame({'age': ['0'] * 8, 'class': ['1'] * 8, 'target': ['1'] * 8})
        g3 = pd.DataFrame({'age': ['1'] * 12, 'class': ['0'] * 12, 'target': ['0'] * 12})
        data = pd.concat([g1, g2, g3], ignore_index=True)

        intrinsic, transition = _make_utility_tables()
        cvs = _make_column_values()

        wald = AnalyticEngine(analytic_type="wald").compute(
            data,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cvs,
        )[0]
        wilson = AnalyticEngine(analytic_type="wilson").compute(
            data,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cvs,
        )[0]

        # Gain point estimates match.
        assert wald.realistic_rule_gain_point == pytest.approx(
            wilson.realistic_rule_gain_point, rel=1e-9
        )
        # But the bounds differ (small n).
        assert wald.realistic_rule_gain_ci_lower != pytest.approx(
            wilson.realistic_rule_gain_ci_lower, abs=1e-6
        )

    def test_gain_endpoints_ordered_under_negative_target_gain(self):
        """When Δu_target < 0 the gain interval lower bound must still be <= upper."""
        # Utility tables with NEGATIVE target_gain: u(target=0) > u(target=1).
        intrinsic = {
            ('target', '0'): 5.0,
            ('target', '1'): 1.0,
        }
        transition: dict = {}
        data = _make_data(n=200)
        extra = pd.DataFrame({'age': ['0'] * 10, 'class': ['0'] * 10, 'target': ['1'] * 10})
        mixed = pd.concat([data, extra], ignore_index=True)
        cv = _make_column_values()

        wald = AnalyticEngine(analytic_type="wald").compute(
            mixed,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        wilson = AnalyticEngine(analytic_type="wilson").compute(
            mixed,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert wald.realistic_rule_gain_ci_lower <= wald.realistic_rule_gain_ci_upper
        assert wilson.realistic_rule_gain_ci_lower <= wilson.realistic_rule_gain_ci_upper


class TestAutoMode:
    """Tests for analytic_type='auto'."""

    def test_auto_uses_wilson_for_small_n(self):
        """With n < 40, auto should match Newcombe-Wilson, not Wald.

        Point estimates of all three methods agree (MLE plug-in); the
        switching is observed in the CI bounds.
        """
        g1 = pd.DataFrame({'age': ['0'] * 15, 'class': ['0'] * 15, 'target': ['0'] * 15})
        g2 = pd.DataFrame({'age': ['0'] * 10, 'class': ['1'] * 10, 'target': ['1'] * 10})
        g3 = pd.DataFrame({'age': ['1'] * 10, 'class': ['0'] * 10, 'target': ['0'] * 10})
        data = pd.concat([g1, g2, g3], ignore_index=True)

        wald = AnalyticEngine(analytic_type="wald").compute(data, [_make_rule()])[0]
        auto = AnalyticEngine(analytic_type="auto").compute(data, [_make_rule()])[0]
        wilson = AnalyticEngine(analytic_type="wilson").compute(data, [_make_rule()])[0]

        # Auto matches Wilson on CI bounds (n < 40 triggers switch).
        assert auto.uplift_ci_lower == pytest.approx(wilson.uplift_ci_lower, rel=1e-9)
        assert auto.uplift_ci_upper == pytest.approx(wilson.uplift_ci_upper, rel=1e-9)
        # Wald bounds differ from auto on this small dataset.
        assert auto.uplift_ci_lower != pytest.approx(wald.uplift_ci_lower, abs=1e-6)

    def test_auto_uses_wald_for_large_n_moderate_p(self):
        """With n >= 40 and moderate p (all in 0.05-0.95), auto should match Wald exactly."""
        # Build data where both p_u and p_d are moderate (~0.75).
        # mask_undesired (age=0, class=0): 60 match / 80 total -> p_u = 0.75
        # mask_desired   (age=0, class=1): 45 match / 60 total -> p_d = 0.75
        g1 = pd.DataFrame({'age': ['0'] * 60, 'class': ['0'] * 60, 'target': ['0'] * 60})
        g2 = pd.DataFrame({'age': ['0'] * 20, 'class': ['0'] * 20, 'target': ['1'] * 20})
        g3 = pd.DataFrame({'age': ['0'] * 45, 'class': ['1'] * 45, 'target': ['1'] * 45})
        g4 = pd.DataFrame({'age': ['0'] * 15, 'class': ['1'] * 15, 'target': ['0'] * 15})
        g5 = pd.DataFrame({'age': ['1'] * 60, 'class': ['0'] * 60, 'target': ['0'] * 60})
        big_data = pd.concat([g1, g2, g3, g4, g5], ignore_index=True)

        wald = AnalyticEngine(analytic_type="wald").compute(big_data, [_make_rule()])[0]
        auto = AnalyticEngine(analytic_type="auto").compute(big_data, [_make_rule()])[0]

        assert auto.uplift_point == pytest.approx(wald.uplift_point, rel=1e-12)
        assert auto.uplift_se == pytest.approx(wald.uplift_se, rel=1e-12)
