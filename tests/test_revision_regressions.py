"""Independent checks for the manuscript revision's corrected contracts."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from action_rules.evaluation.cv import _metric_value, _score_rule_on_data
from action_rules.evaluation.metrics import qini_coefficient
from action_rules.inference.analytic import AnalyticEngine
from action_rules.inference.base import RuleCategory, RuleMasks
from action_rules.inference.bayesian import BayesianEngine
from action_rules.inference.bootstrap import BootstrapEngine


def rule():
    """Return a disjoint binary-antecedent rule."""
    return RuleMasks({'F': 'u'}, {'F': 'd'}, 'Y', '0', '1', 0, (0,), (1,))


@pytest.mark.parametrize('missing', ['u', 'd'])
@pytest.mark.parametrize('engine', [AnalyticEngine(), BayesianEngine(n_mc=200), BootstrapEngine(n_bootstrap=200)])
def test_either_empty_antecedent_is_explicitly_unsupported(engine, missing):
    """Reject a missing group consistently across inference engines."""
    data = pd.DataFrame({'F': ['d' if missing == 'u' else 'u'] * 20, 'Y': ['0', '1'] * 10})
    result = engine.compute(data, [rule()])[0]
    assert result.category is RuleCategory.UNSUPPORTED
    assert np.isnan(result.uplift_ci_lower)
    assert result.decision_status == 'unsupported'


def test_population_gain_uses_undesired_antecedent_count():
    """Weight gains by the population they describe."""
    data = pd.DataFrame({'F': ['u'] * 10 + ['d'] * 30, 'Y': ['0'] * 10 + ['1'] * 30})
    scored = _score_rule_on_data(rule(), data, {('Y', '1'): 30.0}, {}, {0: ('F', 'u'), 1: ('F', 'd')})
    assert scored['support'] == 10
    assert _metric_value('profit_at_k', np.array([1.0]), np.array([30.0]), np.array([scored['support']]), 1.0) == 300.0


def test_cv_qini_is_rule_counted():
    """Check the curve against its hand-computed two-rule area."""
    scores, outcomes, supports = np.array([2.0, 1.0]), np.array([1.0, 3.0]), np.array([10.0, 30.0])
    assert _metric_value('qini', scores, outcomes, supports, 0.2) == pytest.approx(-0.125)


@pytest.mark.parametrize('outcomes', [[0.0, 0.0], [-1.0, -2.0]])
def test_invalid_qini_normalization_is_nan(outcomes):
    """Reject a curve whose endpoint cannot be normalized."""
    assert np.isnan(qini_coefficient([2.0, 1.0], outcomes))


def test_rectangular_cp_has_enumerated_coverage():
    """Enumerate both binomials rather than echoing interval implementation."""
    n, alpha = 7, 0.05
    for pu, pd_ in itertools.product([0.01, 0.2, 0.5, 0.95], repeat=2):
        coverage = 0.0
        for xu, xd in itertools.product(range(n + 1), repeat=2):
            _, lu, uu = AnalyticEngine._clopper_pearson_bounds(xu, n, alpha / 2)
            _, ld, ud = AnalyticEngine._clopper_pearson_bounds(xd, n, alpha / 2)
            if lu + ld - 1 <= pu + pd_ - 1 <= uu + ud - 1:
                coverage += binom.pmf(xu, n, pu) * binom.pmf(xd, n, pd_)
        assert coverage >= 1 - alpha - 1e-12


def test_sparse_bca_cannot_substitute_zero_for_undefined_jackknife():
    """Decline BCa when deleting a row empties an antecedent."""
    data = pd.DataFrame({'F': ['u'] + ['d'] * 20, 'Y': ['0'] + ['1'] * 20})
    with pytest.warns(UserWarning):
        result = BootstrapEngine(n_bootstrap=499, random_state=7, bootstrap_type='bca').compute(data, [rule()])[0]
    assert result.category is RuleCategory.UNSUPPORTED
    assert np.isnan(result.uplift_ci_lower)


def test_vectorized_gain_matches_scalar_formula_for_negative_b():
    """Compare vectorized utility draws with independent scalar evaluations."""
    from action_rules.inference.base import compute_realistic_gain

    data = pd.DataFrame({'F': ['u'] * 40 + ['d'] * 50, 'Y': ['0', '1'] * 45})
    intrinsic = {('Y', '0'): 100.0, ('F', 'u'): 10.0, ('F', 'd'): 5.0}
    cv = {0: ('F', 'u'), 1: ('F', 'd')}
    result = BootstrapEngine(n_bootstrap=499, random_state=2, resample_mode='multinomial').compute(
        data, [rule()], intrinsic_utility_table=intrinsic, column_values=cv
    )[0]
    assert len(result.samples_gain) == 499
    assert np.all(np.isfinite(result.samples_gain))
    # Reproduce five-cell draws independently and evaluate the scalar utility function.
    counts = np.array([20, 20, 25, 25, 0])
    draws = np.random.default_rng(2).multinomial(90, counts / 90, size=499)
    expected = [
        compute_realistic_gain(rule(), x[0] / (x[0] + x[1]), x[2] / (x[2] + x[3]), intrinsic, None, cv) for x in draws
    ]
    np.testing.assert_allclose(result.samples_gain, expected)


def test_random_design_variances_match_multinomial_delta_method():
    """Check closed forms using the full five-cell covariance and numerical gradients."""
    prob = np.array([0.14, 0.06, 0.24, 0.06, 0.50])

    def targets(x):
        q = x[0] + x[1]
        delta = x[0] / q + x[2] / (x[2] + x[3]) - 1
        return np.array([q * delta, q * (-30 + 100 * delta)])

    eps = 1e-6
    grad = np.column_stack([(targets(prob + eps * e) - targets(prob - eps * e)) / (2 * eps) for e in np.eye(5)])
    covariance = (np.diag(prob) - np.outer(prob, prob)) / 200
    actual = np.diag(grad @ covariance @ grad.T)
    delta, q = 0.5, 0.2
    vd = 0.7 * 0.3 / (200 * 0.2) + 0.8 * 0.2 / (200 * 0.3)
    expected = [q * q * vd + delta * delta * q * (1 - q) / 200, q * q * 100**2 * vd + 20**2 * q * (1 - q) / 200]
    np.testing.assert_allclose(actual, expected, rtol=1e-7)
