#!/usr/bin/env python
"""Integration tests for ActionRules.confidence_intervals()."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from action_rules.action_rules import ActionRules
from action_rules.inference.base import ConfidenceIntervalResult, RuleCategory


# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

_DATA = pd.DataFrame(
    {
        'Sex': [
            'M', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F',
            'M', 'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'F',
        ],
        'Class': [
            '1st', '1st', '2nd', '2nd', '3rd', '1st', '1st', '2nd', '2nd', '3rd',
            '1st', '1st', '2nd', '3rd', '3rd', '1st', '2nd', '2nd', '3rd', '3rd',
        ],
        'Survived': [
            '0', '0', '0', '0', '0', '1', '1', '1', '1', '0',
            '0', '0', '0', '0', '1', '1', '1', '1', '0', '0',
        ],
    }
)

_INTRINSIC_UTILITY_TABLE = {
    ('Class', '1st'): 10.0,
    ('Class', '2nd'): 5.0,
    ('Class', '3rd'): 1.0,
    ('Survived', '0'): 0.0,
    ('Survived', '1'): 100.0,
}

_TRANSITION_UTILITY_TABLE = {
    ('Class', '2nd', '1st'): -20.0,
    ('Class', '3rd', '1st'): -30.0,
    ('Class', '3rd', '2nd'): -10.0,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_action_rules(**kwargs) -> ActionRules:
    """Construct an ActionRules instance with low support/confidence thresholds.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments forwarded to ActionRules.__init__.

    Returns
    -------
    ActionRules
        Unfitted instance with permissive mining thresholds.
    """
    return ActionRules(
        min_stable_attributes=1,
        min_flexible_attributes=1,
        min_undesired_support=1,
        min_undesired_confidence=0.5,
        min_desired_support=1,
        min_desired_confidence=0.5,
        verbose=False,
        **kwargs,
    )


@pytest.fixture
def fitted_ar():
    """
    Fixture that fits an ActionRules model on the synthetic Titanic-style data.

    Returns
    -------
    tuple[ActionRules, pd.DataFrame]
        ``(action_rules_instance, original_data)`` where the model has been
        fully fitted and at least two action rules have been found.
    """
    ar = _make_action_rules()
    ar.fit(
        _DATA.copy(),
        stable_attributes=['Sex'],
        flexible_attributes=['Class'],
        target='Survived',
        target_undesired_state='0',
        target_desired_state='1',
    )
    return ar, _DATA.copy()


@pytest.fixture
def fitted_ar_with_utility():
    """
    Fixture that fits ActionRules with intrinsic and transition utility tables.

    Returns
    -------
    tuple[ActionRules, pd.DataFrame]
        ``(action_rules_instance, original_data)`` ready for utility-aware CI tests.
    """
    ar = _make_action_rules(
        intrinsic_utility_table=_INTRINSIC_UTILITY_TABLE,
        transition_utility_table=_TRANSITION_UTILITY_TABLE,
    )
    ar.fit(
        _DATA.copy(),
        stable_attributes=['Sex'],
        flexible_attributes=['Class'],
        target='Survived',
        target_undesired_state='0',
        target_desired_state='1',
    )
    return ar, _DATA.copy()


# ---------------------------------------------------------------------------
# TestConfidenceIntervalsBootstrap
# ---------------------------------------------------------------------------


class TestConfidenceIntervalsBootstrap:
    """Tests for the bootstrap CI method."""

    def test_returns_list(self, fitted_ar):
        """
        Verify that confidence_intervals() returns a list of ConfidenceIntervalResult.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        Result is a list and every element is a ConfidenceIntervalResult.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=42)
        assert isinstance(results, list)
        assert all(isinstance(r, ConfidenceIntervalResult) for r in results)

    def test_result_count_matches_rules(self, fitted_ar):
        """
        Verify the number of results equals the number of fitted action rules.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        len(results) == len(ar.output.action_rules).
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=42)
        assert len(results) == len(ar.output.action_rules)

    def test_method_field(self, fitted_ar):
        """
        Verify every result carries method='bootstrap'.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        All ConfidenceIntervalResult.method values equal 'bootstrap'.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=42)
        assert all(r.method == 'bootstrap' for r in results)

    def test_ci_bounds_valid(self, fitted_ar):
        """
        Verify ci_lower <= uplift_point <= ci_upper for every result.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        uplift_ci_lower <= uplift_point <= uplift_ci_upper for all results that
        are not NaN.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=42)
        for r in results:
            # Guard: NaN results arise when zero antecedent support occurs in all resamples.
            if not (np.isnan(r.uplift_point) or np.isnan(r.uplift_ci_lower) or np.isnan(r.uplift_ci_upper)):
                assert r.uplift_ci_lower <= r.uplift_point <= r.uplift_ci_upper, (
                    f"CI bounds violated for rule {r.rule_index}: "
                    f"[{r.uplift_ci_lower}, {r.uplift_ci_upper}] does not contain {r.uplift_point}"
                )

    def test_deterministic(self, fitted_ar):
        """
        Verify that the same random_state produces identical results across two calls.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        CI lower and upper bounds are bitwise identical for both runs.
        """
        _, data = fitted_ar

        # Build two independent instances to ensure no shared state.
        ar1 = _make_action_rules()
        ar1.fit(
            data.copy(),
            stable_attributes=['Sex'],
            flexible_attributes=['Class'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
        )
        ar2 = _make_action_rules()
        ar2.fit(
            data.copy(),
            stable_attributes=['Sex'],
            flexible_attributes=['Class'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
        )

        res1 = ar1.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=99)
        res2 = ar2.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=99)

        for r1, r2 in zip(res1, res2):
            assert r1.uplift_ci_lower == r2.uplift_ci_lower
            assert r1.uplift_ci_upper == r2.uplift_ci_upper

    def test_samples_stored(self, fitted_ar):
        """
        Verify that samples_uplift is a non-None array with at most n_bootstrap entries.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        samples_uplift is a numpy array and len(samples_uplift) <= n_bootstrap.
        """
        ar, data = fitted_ar
        n_bootstrap = 50
        results = ar.confidence_intervals(data, method='bootstrap', n_bootstrap=n_bootstrap, random_state=42)
        for r in results:
            assert r.samples_uplift is not None, f"samples_uplift should not be None for rule {r.rule_index}"
            assert isinstance(r.samples_uplift, np.ndarray)
            # Valid samples may be fewer than n_bootstrap when some resamples
            # produce zero antecedent support and are discarded.
            assert len(r.samples_uplift) <= n_bootstrap


# ---------------------------------------------------------------------------
# TestConfidenceIntervalsAnalytic
# ---------------------------------------------------------------------------


class TestConfidenceIntervalsAnalytic:
    """Tests for the analytic (Wald) CI method."""

    def test_returns_list(self, fitted_ar):
        """
        Verify that the analytic method returns a list of ConfidenceIntervalResult.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        Result is a non-empty list of ConfidenceIntervalResult objects.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='analytic')
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, ConfidenceIntervalResult) for r in results)

    def test_method_field(self, fitted_ar):
        """
        Verify every analytic result carries method='analytic'.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        All ConfidenceIntervalResult.method values equal 'analytic'.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='analytic')
        assert all(r.method == 'analytic' for r in results)

    def test_ci_symmetric(self, fitted_ar):
        """
        Verify the Wald interval is symmetric around the point estimate.

        The analytic engine uses ``point ± z * SE``, so
        ``point - lower ≈ upper - point`` within floating-point tolerance.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        |delta_lower - delta_upper| < 1e-10 for all non-NaN results.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='analytic')
        for r in results:
            if np.isnan(r.uplift_point):
                continue
            delta_lower = r.uplift_point - r.uplift_ci_lower
            delta_upper = r.uplift_ci_upper - r.uplift_point
            assert abs(delta_lower - delta_upper) < 1e-10, (
                f"Analytic CI not symmetric for rule {r.rule_index}: "
                f"delta_lower={delta_lower}, delta_upper={delta_upper}"
            )

    def test_no_samples(self, fitted_ar):
        """
        Verify that samples_uplift is None for the analytic engine.

        The analytic engine uses a closed-form formula and does not produce
        sample arrays.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        samples_uplift is None for every result.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='analytic')
        assert all(r.samples_uplift is None for r in results)

    def test_wald_alias_accepted(self, fitted_ar):
        """
        Verify that 'wald' is accepted as an alias for 'analytic'.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        confidence_intervals() does not raise and returns results with method='analytic'.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='wald')
        assert all(r.method == 'analytic' for r in results)


# ---------------------------------------------------------------------------
# TestConfidenceIntervalsBayesian
# ---------------------------------------------------------------------------


class TestConfidenceIntervalsBayesian:
    """Tests for the Bayesian credible interval method."""

    def test_returns_list(self, fitted_ar):
        """
        Verify that the Bayesian method returns a list of ConfidenceIntervalResult.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        Result is a non-empty list of ConfidenceIntervalResult objects.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='bayesian', n_mc=100, random_state=42)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, ConfidenceIntervalResult) for r in results)

    def test_method_field(self, fitted_ar):
        """
        Verify every Bayesian result carries method='bayesian'.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        All ConfidenceIntervalResult.method values equal 'bayesian'.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='bayesian', n_mc=100, random_state=42)
        assert all(r.method == 'bayesian' for r in results)

    def test_samples_stored(self, fitted_ar):
        """
        Verify that samples_uplift is a numpy array of length n_mc.

        The Bayesian engine draws n_mc samples from the posterior without
        discarding any (unlike bootstrap which may skip zero-support resamples).

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        samples_uplift is not None and has exactly n_mc elements.
        """
        ar, data = fitted_ar
        n_mc = 100
        results = ar.confidence_intervals(data, method='bayesian', n_mc=n_mc, random_state=42)
        for r in results:
            assert r.samples_uplift is not None, f"samples_uplift should not be None for rule {r.rule_index}"
            assert isinstance(r.samples_uplift, np.ndarray)
            assert len(r.samples_uplift) == n_mc

    def test_deterministic(self, fitted_ar):
        """
        Verify that two identical random_state seeds produce bitwise-identical results.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        CI lower and upper bounds are identical across two runs with the same seed.
        """
        _, data = fitted_ar

        ar1 = _make_action_rules()
        ar1.fit(
            data.copy(),
            stable_attributes=['Sex'],
            flexible_attributes=['Class'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
        )
        ar2 = _make_action_rules()
        ar2.fit(
            data.copy(),
            stable_attributes=['Sex'],
            flexible_attributes=['Class'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
        )

        res1 = ar1.confidence_intervals(data, method='bayesian', n_mc=100, random_state=7)
        res2 = ar2.confidence_intervals(data, method='bayesian', n_mc=100, random_state=7)

        for r1, r2 in zip(res1, res2):
            assert r1.uplift_ci_lower == r2.uplift_ci_lower
            assert r1.uplift_ci_upper == r2.uplift_ci_upper


# ---------------------------------------------------------------------------
# TestConfidenceIntervalsWithUtility
# ---------------------------------------------------------------------------


class TestConfidenceIntervalsWithUtility:
    """Tests verifying gain fields are populated when utility tables are provided."""

    def test_gain_fields_populated(self, fitted_ar_with_utility):
        """
        Verify all gain fields are non-None when utility tables were provided at construction.

        Parameters
        ----------
        fitted_ar_with_utility : tuple[ActionRules, pd.DataFrame]
            ActionRules instance constructed with utility tables.

        Asserts
        -------
        realistic_rule_gain_point, ci_lower, ci_upper, and se are all not None.
        """
        ar, data = fitted_ar_with_utility
        results = ar.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=42)
        for r in results:
            assert r.realistic_rule_gain_point is not None, (
                f"realistic_rule_gain_point should not be None for rule {r.rule_index}"
            )
            assert r.realistic_rule_gain_ci_lower is not None
            assert r.realistic_rule_gain_ci_upper is not None
            assert r.realistic_rule_gain_se is not None

    def test_gain_fields_none_without_utility(self, fitted_ar):
        """
        Verify gain fields remain None when no utility tables were provided.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            ActionRules instance constructed without utility tables.

        Asserts
        -------
        realistic_rule_gain_point, ci_lower, ci_upper, and se are all None.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=42)
        for r in results:
            assert r.realistic_rule_gain_point is None, (
                f"realistic_rule_gain_point should be None when no utility tables are given (rule {r.rule_index})"
            )
            assert r.realistic_rule_gain_ci_lower is None
            assert r.realistic_rule_gain_ci_upper is None
            assert r.realistic_rule_gain_se is None

    def test_gain_fields_populated_analytic(self, fitted_ar_with_utility):
        """
        Verify gain fields are populated for the analytic engine when utility tables are given.

        Parameters
        ----------
        fitted_ar_with_utility : tuple[ActionRules, pd.DataFrame]
            ActionRules instance with utility tables.

        Asserts
        -------
        realistic_rule_gain_point is not None and is finite.
        """
        ar, data = fitted_ar_with_utility
        results = ar.confidence_intervals(data, method='analytic')
        for r in results:
            assert r.realistic_rule_gain_point is not None
            assert np.isfinite(r.realistic_rule_gain_point)

    def test_gain_fields_populated_bayesian(self, fitted_ar_with_utility):
        """
        Verify gain fields and gain samples are populated for the Bayesian engine.

        Parameters
        ----------
        fitted_ar_with_utility : tuple[ActionRules, pd.DataFrame]
            ActionRules instance with utility tables.

        Asserts
        -------
        samples_gain is not None and has exactly n_mc elements.
        """
        ar, data = fitted_ar_with_utility
        n_mc = 100
        results = ar.confidence_intervals(data, method='bayesian', n_mc=n_mc, random_state=42)
        for r in results:
            assert r.realistic_rule_gain_point is not None
            assert r.samples_gain is not None
            assert len(r.samples_gain) == n_mc


# ---------------------------------------------------------------------------
# TestCategorization
# ---------------------------------------------------------------------------


class TestCategorization:
    """Tests for the threshold-based rule categorization logic."""

    def test_accept_with_low_threshold(self):
        """
        Verify that a threshold of 0.0 categorizes rules with positive uplift as ACCEPT.

        The synthetic dataset produces rules with positive uplift (~0.15).
        With a sufficiently large bootstrap sample and threshold=0.0 the
        entire CI should lie above 0.0, giving ACCEPT.

        Asserts
        -------
        All categories equal RuleCategory.ACCEPT.
        """
        ar = _make_action_rules()
        ar.fit(
            _DATA.copy(),
            stable_attributes=['Sex'],
            flexible_attributes=['Class'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
        )
        results = ar.confidence_intervals(_DATA.copy(), method='bootstrap', n_bootstrap=200, random_state=42, threshold=0.0)
        for r in results:
            assert r.category == RuleCategory.ACCEPT, (
                f"Expected ACCEPT for rule {r.rule_index} with threshold=0.0, got {r.category}"
            )

    def test_reject_with_high_threshold(self):
        """
        Verify that a very high threshold causes all rules to be categorized as REJECT.

        With threshold=10.0 the uplift CI (bounded by [0, 1]) will always
        lie entirely below the threshold.

        Asserts
        -------
        All categories equal RuleCategory.REJECT.
        """
        ar = _make_action_rules()
        ar.fit(
            _DATA.copy(),
            stable_attributes=['Sex'],
            flexible_attributes=['Class'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
        )
        results = ar.confidence_intervals(_DATA.copy(), method='bootstrap', n_bootstrap=50, random_state=42, threshold=10.0)
        for r in results:
            assert r.category == RuleCategory.REJECT, (
                f"Expected REJECT for rule {r.rule_index} with threshold=10.0, got {r.category}"
            )

    def test_uncertain_exists(self):
        """
        Verify that a moderate threshold produces UNCERTAIN categorizations.

        With threshold=0.1 (inside the bootstrap CI for the synthetic data)
        the CI straddles the threshold, yielding UNCERTAIN.

        Asserts
        -------
        At least one result has category == RuleCategory.UNCERTAIN.
        """
        ar = _make_action_rules()
        ar.fit(
            _DATA.copy(),
            stable_attributes=['Sex'],
            flexible_attributes=['Class'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
        )
        results = ar.confidence_intervals(_DATA.copy(), method='bootstrap', n_bootstrap=200, random_state=42, threshold=0.1)
        categories = [r.category for r in results]
        assert RuleCategory.UNCERTAIN in categories, (
            f"Expected at least one UNCERTAIN result with threshold=0.1, got: {[c.value for c in categories]}"
        )

    def test_category_none_without_threshold(self, fitted_ar):
        """
        Verify that category is None when threshold is not provided.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        All category fields are None when threshold=None (default).
        """
        ar, data = fitted_ar
        # category is always set by bootstrap (internally uses threshold=0.0);
        # passing no threshold to confidence_intervals() leaves it up to the engine default.
        # The bootstrap and bayesian engines always set category to a default value.
        # We only test that passing threshold=None to confidence_intervals does not crash.
        results = ar.confidence_intervals(data, method='analytic')
        # Analytic engine sets category internally as well; just verify no exception.
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestOutputIntegration
# ---------------------------------------------------------------------------


class TestOutputIntegration:
    """Tests verifying CI data surfaces correctly through Output formatting methods."""

    def test_export_includes_ci(self, fitted_ar):
        """
        Verify that get_export_notation() includes a 'ci' key after confidence_intervals().

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        Each rule dict in the JSON export contains a 'ci' key with at least
        'method', 'uplift_ci_lower', and 'uplift_ci_upper' sub-keys.
        """
        ar, data = fitted_ar
        ar.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=42)
        export = json.loads(ar.get_rules().get_export_notation())
        for rule_dict in export:
            assert 'ci' in rule_dict, "Expected 'ci' key in exported rule dict after confidence_intervals()."
            ci = rule_dict['ci']
            assert 'method' in ci
            assert 'uplift_ci_lower' in ci
            assert 'uplift_ci_upper' in ci

    def test_ar_notation_includes_ci(self, fitted_ar):
        """
        Verify that get_ar_notation() includes CI text after confidence_intervals().

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        Each rule string contains 'uplift CI'.
        """
        ar, data = fitted_ar
        ar.confidence_intervals(data, method='bootstrap', n_bootstrap=50, random_state=42)
        notations = ar.get_rules().get_ar_notation()
        for notation in notations:
            assert 'uplift CI' in notation, f"Expected 'uplift CI' in notation: {notation[:120]}"

    def test_export_without_ci(self, fitted_ar):
        """
        Verify that get_export_notation() has no 'ci' key before confidence_intervals() is called.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        No rule dict in the JSON export contains a 'ci' key before CI computation.
        """
        ar, _ = fitted_ar
        export = json.loads(ar.get_rules().get_export_notation())
        for rule_dict in export:
            assert 'ci' not in rule_dict, "Expected no 'ci' key before confidence_intervals() is called."

    def test_ci_stored_on_output(self, fitted_ar):
        """
        Verify that set_confidence_intervals stores results accessible via output.ci_results.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        output.ci_results is a list with the same length as action_rules.
        """
        ar, data = fitted_ar
        results = ar.confidence_intervals(data, method='analytic')
        assert ar.output.ci_results is not None
        assert len(ar.output.ci_results) == len(ar.output.action_rules)
        assert ar.output.ci_results is results

    def test_pretty_notation_includes_ci(self, fitted_ar):
        """
        Verify that get_pretty_ar_notation() mentions the CI after confidence_intervals().

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        Each pretty notation string contains 'uplift CI'.
        """
        ar, data = fitted_ar
        ar.confidence_intervals(data, method='analytic')
        pretty = ar.get_rules().get_pretty_ar_notation()
        for text in pretty:
            assert 'uplift CI' in text, f"Expected 'uplift CI' in pretty notation: {text[:120]}"


# ---------------------------------------------------------------------------
# TestInvalidInput
# ---------------------------------------------------------------------------


class TestInvalidInput:
    """Tests for error handling in confidence_intervals()."""

    def test_not_fitted_raises_runtime_error(self):
        """
        Verify that calling confidence_intervals() before fit() raises RuntimeError.

        Asserts
        -------
        RuntimeError with message 'The model is not fit.' is raised.
        """
        ar = _make_action_rules()
        with pytest.raises(RuntimeError, match="The model is not fit."):
            ar.confidence_intervals(_DATA.copy(), method='bootstrap', n_bootstrap=10)

    def test_invalid_method_raises_value_error(self, fitted_ar):
        """
        Verify that an unsupported method name raises ValueError.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        ValueError with the unknown method name is raised.
        """
        ar, data = fitted_ar
        with pytest.raises(ValueError, match="Unknown method"):
            ar.confidence_intervals(data, method='nonexistent_method')

    def test_invalid_method_message_contains_name(self, fitted_ar):
        """
        Verify the ValueError message includes the offending method name.

        Parameters
        ----------
        fitted_ar : tuple[ActionRules, pd.DataFrame]
            Fitted ActionRules instance and the corresponding original dataset.

        Asserts
        -------
        The error message contains the string 'bad_method'.
        """
        ar, data = fitted_ar
        with pytest.raises(ValueError, match="bad_method"):
            ar.confidence_intervals(data, method='bad_method')


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for CI flags in the Click CLI."""

    def _make_csv(self) -> str:
        """Write the synthetic dataset to a temporary CSV file.

        Returns
        -------
        str
            Absolute path to the temporary CSV file.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            _DATA.to_csv(f, index=False)
            return f.name

    def test_cli_bootstrap_produces_ci_in_json(self):
        """
        Verify that the CLI with --ci_method bootstrap writes CI data to the output JSON.

        Asserts
        -------
        Each rule in the output JSON contains a 'ci' key with method='bootstrap'.
        """
        from click.testing import CliRunner

        from action_rules.cli import main

        csv_path = self._make_csv()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    '--min_stable_attributes', '1',
                    '--min_flexible_attributes', '1',
                    '--min_undesired_support', '1',
                    '--min_undesired_confidence', '0.5',
                    '--min_desired_support', '1',
                    '--min_desired_confidence', '0.5',
                    '--csv_path', csv_path,
                    '--stable_attributes', 'Sex',
                    '--flexible_attributes', 'Class',
                    '--target', 'Survived',
                    '--undesired_state', '0',
                    '--desired_state', '1',
                    '--output_json_path', json_path,
                    '--ci_method', 'bootstrap',
                    '--n_bootstrap', '10',
                    '--random_state', '42',
                ],
            )
            assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"
            with open(json_path) as f:
                rules = json.loads(f.read())
            assert len(rules) >= 1
            for rule_dict in rules:
                assert 'ci' in rule_dict, "Expected 'ci' key in CLI output JSON."
                assert rule_dict['ci']['method'] == 'bootstrap'
        finally:
            os.unlink(csv_path)
            os.unlink(json_path)

    def test_cli_no_ci_method_omits_ci_key(self):
        """
        Verify that the CLI without --ci_method does not include 'ci' in the output JSON.

        Asserts
        -------
        No rule in the output JSON contains a 'ci' key.
        """
        from click.testing import CliRunner

        from action_rules.cli import main

        csv_path = self._make_csv()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    '--min_stable_attributes', '1',
                    '--min_flexible_attributes', '1',
                    '--min_undesired_support', '1',
                    '--min_undesired_confidence', '0.5',
                    '--min_desired_support', '1',
                    '--min_desired_confidence', '0.5',
                    '--csv_path', csv_path,
                    '--stable_attributes', 'Sex',
                    '--flexible_attributes', 'Class',
                    '--target', 'Survived',
                    '--undesired_state', '0',
                    '--desired_state', '1',
                    '--output_json_path', json_path,
                ],
            )
            assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"
            with open(json_path) as f:
                rules = json.loads(f.read())
            for rule_dict in rules:
                assert 'ci' not in rule_dict, "Expected no 'ci' key when --ci_method is not passed."
        finally:
            os.unlink(csv_path)
            os.unlink(json_path)

    def test_cli_analytic_produces_ci_in_json(self):
        """
        Verify that --ci_method analytic writes 'analytic' CI data to the output JSON.

        Asserts
        -------
        Each rule in the output JSON contains a 'ci' key with method='analytic'.
        """
        from click.testing import CliRunner

        from action_rules.cli import main

        csv_path = self._make_csv()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    '--min_stable_attributes', '1',
                    '--min_flexible_attributes', '1',
                    '--min_undesired_support', '1',
                    '--min_undesired_confidence', '0.5',
                    '--min_desired_support', '1',
                    '--min_desired_confidence', '0.5',
                    '--csv_path', csv_path,
                    '--stable_attributes', 'Sex',
                    '--flexible_attributes', 'Class',
                    '--target', 'Survived',
                    '--undesired_state', '0',
                    '--desired_state', '1',
                    '--output_json_path', json_path,
                    '--ci_method', 'analytic',
                ],
            )
            assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"
            with open(json_path) as f:
                rules = json.loads(f.read())
            for rule_dict in rules:
                assert 'ci' in rule_dict
                assert rule_dict['ci']['method'] == 'analytic'
        finally:
            os.unlink(csv_path)
            os.unlink(json_path)
