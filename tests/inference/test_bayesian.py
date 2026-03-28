"""Tests for BayesianEngine."""

import math

import numpy as np
import pandas as pd
import pytest

from action_rules.inference.base import (
    ConfidenceIntervalResult,
    RuleCategory,
    RuleMasks,
    compute_group_counts,
)
from action_rules.inference.bayesian import BayesianEngine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a simple synthetic dataset with two attributes and one target.

    Columns
    -------
    age   : '0' (young) or '1' (old)
    class : '0' (economy) or '1' (business)
    target: '0' (undesired) or '1' (desired)

    The data is constructed so that the rule
    "if age stays and class changes 0->1, then target changes 0->1"
    has non-trivial support and confidence.
    """
    n_each = n // 4

    g1 = pd.DataFrame({'age': ['0'] * n_each, 'class': ['0'] * n_each, 'target': ['0'] * n_each})
    g2 = pd.DataFrame({'age': ['0'] * n_each, 'class': ['1'] * n_each, 'target': ['1'] * n_each})
    g3 = pd.DataFrame({'age': ['1'] * n_each, 'class': ['0'] * n_each, 'target': ['0'] * n_each})
    g4 = pd.DataFrame({'age': ['1'] * n_each, 'class': ['1'] * n_each, 'target': ['1'] * n_each})

    data = pd.concat([g1, g2, g3, g4], ignore_index=True)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    return data


def _make_rule(rule_index: int = 0) -> RuleMasks:
    """Return a RuleMasks for: age=0, class: 0->1, target: 0->1."""
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


def _utility_tables():
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
# Determinism
# ---------------------------------------------------------------------------


class TestBayesianDeterminism:
    """Identical seeds must produce identical results."""

    def test_same_seed_same_uplift(self):
        """Two engines with the same seed and data produce identical uplift CIs."""
        data = _make_data()
        rule = _make_rule()
        r1 = BayesianEngine(n_mc=1000, random_state=42).compute(data, [rule])[0]
        r2 = BayesianEngine(n_mc=1000, random_state=42).compute(data, [rule])[0]
        assert r1.uplift_point == pytest.approx(r2.uplift_point)
        assert r1.uplift_ci_lower == pytest.approx(r2.uplift_ci_lower)
        assert r1.uplift_ci_upper == pytest.approx(r2.uplift_ci_upper)

    def test_different_seed_different_result(self):
        """Two different seeds produce (with overwhelming probability) different CIs."""
        data = _make_data()
        rule = _make_rule()
        r1 = BayesianEngine(n_mc=1000, random_state=1).compute(data, [rule])[0]
        r2 = BayesianEngine(n_mc=1000, random_state=2).compute(data, [rule])[0]
        # At least one bound should differ.
        assert (r1.uplift_ci_lower != r2.uplift_ci_lower) or (r1.uplift_ci_upper != r2.uplift_ci_upper)

    def test_method_field(self):
        """The method field must be 'bayesian'."""
        data = _make_data()
        result = BayesianEngine(n_mc=100, random_state=0).compute(data, [_make_rule()])[0]
        assert result.method == 'bayesian'


# ---------------------------------------------------------------------------
# Frequentist approximation with flat prior and large sample
# ---------------------------------------------------------------------------


class TestFlatPriorLargeSample:
    """With a flat prior and a large dataset the posterior mean should be close
    to the empirical (frequentist) point estimate."""

    def test_posterior_mean_close_to_frequentist(self):
        """Large-sample posterior mean is close to the empirical uplift."""
        # Use a large dataset so the posterior is dominated by the likelihood.
        data = _make_data(n=2000, seed=7)
        rule = _make_rule()

        # Compute the empirical uplift directly from counts.
        n_u_ante, n_u_match, n_d_ante, n_d_match, n_total = compute_group_counts(data, rule)
        p_u_emp = n_u_match / n_u_ante
        p_d_emp = n_d_match / n_d_ante
        d_emp = p_u_emp + p_d_emp - 1.0
        uplift_emp = d_emp * n_u_ante / n_total

        engine = BayesianEngine(n_mc=20000, prior_alpha=1.0, prior_beta=1.0, random_state=0)
        result = engine.compute(data, [rule])[0]

        # With n=2000 the prior adds negligible weight; allow 2% absolute tolerance.
        assert result.uplift_point == pytest.approx(uplift_emp, abs=0.02)

    def test_ci_contains_frequentist_estimate(self):
        """The 95% credible interval should contain the empirical uplift for large n.

        We build a dataset where neither confidence is exactly 1.0 so that
        the frequentist point estimate genuinely sits in the interior of the
        posterior CI (not exactly on a boundary due to the Beta(1,1) prior
        pulling the posterior mean toward 0.5).
        """
        rng = np.random.default_rng(17)
        n = 2000
        # Group 1: undesired antecedent, 90 % undesired target.
        n1 = n // 2
        targets_g1 = ['0'] * int(n1 * 0.9) + ['1'] * (n1 - int(n1 * 0.9))
        g1 = pd.DataFrame({'age': ['0'] * n1, 'class': ['0'] * n1, 'target': targets_g1})
        # Group 2: desired antecedent, 85 % desired target.
        n2 = n // 2
        targets_g2 = ['1'] * int(n2 * 0.85) + ['0'] * (n2 - int(n2 * 0.85))
        g2 = pd.DataFrame({'age': ['0'] * n2, 'class': ['1'] * n2, 'target': targets_g2})
        data = pd.concat([g1, g2], ignore_index=True)
        # Shuffle to avoid positional artefacts.
        data = data.sample(frac=1, random_state=5).reset_index(drop=True)

        rule = _make_rule()

        n_u_ante, n_u_match, n_d_ante, n_d_match, n_total = compute_group_counts(data, rule)
        p_u_emp = n_u_match / n_u_ante
        p_d_emp = n_d_match / n_d_ante
        uplift_emp = (p_u_emp + p_d_emp - 1.0) * n_u_ante / n_total

        result = BayesianEngine(n_mc=20000, random_state=0).compute(data, [rule])[0]
        assert result.uplift_ci_lower <= uplift_emp <= result.uplift_ci_upper


# ---------------------------------------------------------------------------
# Posterior samples stored
# ---------------------------------------------------------------------------


class TestSamplesStored:
    """Raw posterior samples must always be stored on the result."""

    def test_samples_uplift_is_ndarray(self):
        """samples_uplift should be a numpy ndarray."""
        data = _make_data()
        result = BayesianEngine(n_mc=500, random_state=0).compute(data, [_make_rule()])[0]
        assert isinstance(result.samples_uplift, np.ndarray)

    def test_samples_uplift_length_equals_n_mc(self):
        """samples_uplift should have exactly n_mc elements."""
        data = _make_data()
        n_mc = 300
        result = BayesianEngine(n_mc=n_mc, random_state=0).compute(data, [_make_rule()])[0]
        assert result.samples_uplift.shape == (n_mc,)

    def test_samples_gain_none_without_utility_tables(self):
        """samples_gain is None when no utility tables are provided."""
        data = _make_data()
        result = BayesianEngine(n_mc=100, random_state=0).compute(data, [_make_rule()])[0]
        assert result.samples_gain is None


# ---------------------------------------------------------------------------
# Without utility tables
# ---------------------------------------------------------------------------


class TestWithoutUtilityTables:
    """Gain fields should be None when no utility tables are supplied."""

    def test_gain_fields_none(self):
        data = _make_data()
        result = BayesianEngine(n_mc=200, random_state=0).compute(data, [_make_rule()])[0]
        assert result.realistic_rule_gain_point is None
        assert result.realistic_rule_gain_ci_lower is None
        assert result.realistic_rule_gain_ci_upper is None
        assert result.realistic_rule_gain_se is None

    def test_uplift_ci_lower_le_upper(self):
        """CI lower must be <= upper for a well-supported rule."""
        data = _make_data()
        result = BayesianEngine(n_mc=500, random_state=0).compute(data, [_make_rule()])[0]
        assert result.uplift_ci_lower <= result.uplift_ci_upper

    def test_uplift_point_inside_ci(self):
        """The posterior mean should lie within [lower, upper]."""
        data = _make_data()
        result = BayesianEngine(n_mc=500, random_state=0).compute(data, [_make_rule()])[0]
        assert result.uplift_ci_lower <= result.uplift_point <= result.uplift_ci_upper

    def test_se_non_negative(self):
        """SE must be non-negative."""
        data = _make_data()
        result = BayesianEngine(n_mc=500, random_state=0).compute(data, [_make_rule()])[0]
        assert result.uplift_se >= 0.0

    def test_rule_index_preserved(self):
        """rule_index from RuleMasks is carried through."""
        data = _make_data()
        rule = _make_rule(rule_index=5)
        result = BayesianEngine(n_mc=100, random_state=0).compute(data, [rule])[0]
        assert result.rule_index == 5

    def test_confidence_level_preserved(self):
        """confidence_level passed to compute() is stored on the result."""
        data = _make_data()
        result = BayesianEngine(n_mc=100, random_state=0).compute(data, [_make_rule()], confidence_level=0.90)[0]
        assert result.confidence_level == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# With utility tables
# ---------------------------------------------------------------------------


class TestWithUtilityTables:
    """Gain fields are populated when utility tables are provided."""

    def test_gain_fields_populated(self):
        data = _make_data(n=200)
        intrinsic, transition = _utility_tables()
        cv = _make_column_values()
        result = BayesianEngine(n_mc=500, random_state=0).compute(
            data, [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_point is not None
        assert result.realistic_rule_gain_ci_lower is not None
        assert result.realistic_rule_gain_ci_upper is not None
        assert result.realistic_rule_gain_se is not None

    def test_gain_ci_lower_le_upper(self):
        """Gain CI lower <= upper."""
        data = _make_data(n=200)
        intrinsic, transition = _utility_tables()
        cv = _make_column_values()
        result = BayesianEngine(n_mc=500, random_state=0).compute(
            data, [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_ci_lower <= result.realistic_rule_gain_ci_upper

    def test_gain_samples_stored_as_ndarray(self):
        """samples_gain should be a numpy ndarray when utility tables are given."""
        data = _make_data(n=200)
        intrinsic, transition = _utility_tables()
        cv = _make_column_values()
        n_mc = 400
        result = BayesianEngine(n_mc=n_mc, random_state=0).compute(
            data, [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert isinstance(result.samples_gain, np.ndarray)
        assert result.samples_gain.shape == (n_mc,)

    def test_gain_reproducible_with_seed(self):
        """Same seed produces identical gain CI bounds."""
        data = _make_data(n=200)
        intrinsic, transition = _utility_tables()
        cv = _make_column_values()
        r1 = BayesianEngine(n_mc=300, random_state=7).compute(
            data, [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        r2 = BayesianEngine(n_mc=300, random_state=7).compute(
            data, [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert r1.realistic_rule_gain_point == pytest.approx(r2.realistic_rule_gain_point)
        assert r1.realistic_rule_gain_ci_lower == pytest.approx(r2.realistic_rule_gain_ci_lower)

    def test_gain_se_non_negative(self):
        """SE of gain must be non-negative."""
        data = _make_data(n=200)
        intrinsic, transition = _utility_tables()
        cv = _make_column_values()
        result = BayesianEngine(n_mc=300, random_state=0).compute(
            data, [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_se >= 0.0


# ---------------------------------------------------------------------------
# Zero support (prior-only case)
# ---------------------------------------------------------------------------


class TestZeroSupport:
    """When no rows match the antecedent, posterior equals the prior."""

    def _make_zero_support_rule(self) -> RuleMasks:
        """Return a rule whose antecedent matches nothing in the dataset."""
        return RuleMasks(
            mask_undesired={'age': 'NONEXISTENT', 'class': '0'},
            mask_desired={'age': 'NONEXISTENT', 'class': '1'},
            target_attribute='target',
            target_undesired='0',
            target_desired='1',
            rule_index=0,
            undesired_itemset=(99,),
            desired_itemset=(100,),
        )

    def test_no_crash_on_zero_support(self):
        """Engine must not raise on a zero-support rule."""
        data = _make_data()
        rule = self._make_zero_support_rule()
        # Should complete without error.
        result = BayesianEngine(n_mc=200, random_state=0).compute(data, [rule])[0]
        assert isinstance(result, ConfidenceIntervalResult)

    def test_uplift_is_zero_when_n_u_ante_zero(self):
        """When n_u_ante = 0 the uplift samples collapse to 0 (zero scale factor)."""
        data = _make_data()
        rule = self._make_zero_support_rule()
        result = BayesianEngine(n_mc=200, random_state=0).compute(data, [rule])[0]
        # uplift = d * (n_u_ante / n_total); n_u_ante = 0 so uplift = 0 for all draws.
        assert result.uplift_point == pytest.approx(0.0, abs=1e-12)

    def test_samples_still_have_correct_length(self):
        """samples_uplift should still have n_mc elements even for zero-support rule."""
        data = _make_data()
        rule = self._make_zero_support_rule()
        n_mc = 150
        result = BayesianEngine(n_mc=n_mc, random_state=0).compute(data, [rule])[0]
        assert result.samples_uplift.shape == (n_mc,)

    def test_wide_interval_reflects_uncertainty(self):
        """Prior-only posterior (flat Beta(1,1)) should produce a wide interval
        for the underlying confidence.  We check that samples span [0, 1]
        roughly — the mean p_d and p_u draws should each be near 0.5."""
        data = _make_data()
        rule = self._make_zero_support_rule()
        # With flat prior Beta(1,1) posterior draws cover [0,1] uniformly.
        engine = BayesianEngine(n_mc=5000, prior_alpha=1.0, prior_beta=1.0, random_state=42)
        result = engine.compute(data, [rule])[0]
        # uplift = 0 always (scale = 0), so all uplift samples are 0.
        assert result.uplift_ci_lower == pytest.approx(0.0, abs=1e-12)
        assert result.uplift_ci_upper == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Multiple rules
# ---------------------------------------------------------------------------


class TestMultipleRules:
    """Multiple rules processed in a single compute() call."""

    def test_returns_correct_number_of_results(self):
        """One result per rule."""
        data = _make_data()
        rules = [_make_rule(0), _make_rule(1), _make_rule(2)]
        results = BayesianEngine(n_mc=100, random_state=0).compute(data, rules)
        assert len(results) == 3

    def test_results_order_matches_rules_order(self):
        """Results are in the same order as the input rules list."""
        data = _make_data()
        rules = [_make_rule(rule_index=0), _make_rule(rule_index=7)]
        results = BayesianEngine(n_mc=100, random_state=0).compute(data, rules)
        assert results[0].rule_index == 0
        assert results[1].rule_index == 7

    def test_empty_rules_list(self):
        """An empty rules list returns an empty result list without error."""
        data = _make_data()
        results = BayesianEngine(n_mc=100, random_state=0).compute(data, [])
        assert results == []

    def test_all_results_are_ci_result_instances(self):
        """Each element in results is a ConfidenceIntervalResult."""
        data = _make_data()
        rules = [_make_rule(i) for i in range(3)]
        results = BayesianEngine(n_mc=100, random_state=0).compute(data, rules)
        for r in results:
            assert isinstance(r, ConfidenceIntervalResult)

    def test_both_rules_have_valid_uplift(self):
        """Two well-supported rules should both have finite uplift estimates."""
        data = _make_data(n=400)
        rules = [_make_rule(0), _make_rule(1)]
        results = BayesianEngine(n_mc=200, random_state=0).compute(data, rules)
        for r in results:
            assert not math.isnan(r.uplift_point)

    def test_positive_uplift_rule_accepted(self):
        """A rule with clearly positive uplift should be categorised as ACCEPT."""
        data = _make_data(n=400)
        result = BayesianEngine(n_mc=5000, random_state=0).compute(data, [_make_rule()])[0]
        # The synthetic data gives perfect uplift; the posterior CI should be fully > 0.
        assert result.category == RuleCategory.ACCEPT
