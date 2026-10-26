"""Independent checks for the Final manuscript's explicit-design count interface."""

import numpy as np
import pytest

from action_rules.inference.analytic import AnalyticEngine
from action_rules.inference.counts import infer_counts


@pytest.mark.parametrize(
    'method', ['wald', 'wald_rd', 'wilson', 'cp', 'percentile', 'bca', 'bayes', 'jeffreys', 'bayes_rd']
)
def test_batched_results_match_separate_rows(method):
    """Compare batch and scalar deterministic methods and check stochastic support."""
    c = np.array([[27, 13, 64, 27, 1339], [20, 9, 61, 9, 1371]])
    r = infer_counts(c, a=[100, -3900], b=5000, method=method, random_state=4)
    for key in ['G', 'U', 'H']:
        assert r[key]['supported'].all()
        assert np.all(r[key]['lower'] <= r[key]['upper'])
    if method in ['wald', 'wald_rd', 'wilson', 'cp']:
        for i in range(2):
            s = infer_counts(c[i], a=[100, -3900][i], b=5000, method=method)
            for key in ['G', 'U', 'H']:
                np.testing.assert_allclose(r[key]['lower'][i], s[key]['lower'][0])
                np.testing.assert_allclose(r[key]['upper'][i], s[key]['upper'][0])


def test_wilson_gain_agrees_with_existing_engine_formula():
    """Check utility endpoints through the separate component construction."""
    r = infer_counts([27, 13, 64, 27, 1339], a=100, b=5000, method='wilson')['G']
    z = 1.959963984540054
    pu, lu, uu = AnalyticEngine._wilson_bounds(27, 40, z)
    pd, ld, ud = AnalyticEngine._wilson_bounds(64, 91, z)
    _, lo, hi = AnalyticEngine._newcombe_delta_interval(pu, lu, uu, pd, ld, ud)
    np.testing.assert_allclose([r['lower'][0], r['upper'][0]], [100 + 5000 * lo, 100 + 5000 * hi])


@pytest.mark.parametrize('method', ['wald', 'wilson', 'cp', 'percentile', 'bca', 'bayes_rd'])
def test_empty_groups_are_not_accepted(method):
    """Preserve an unavailable estimate across every new method family."""
    r = infer_counts([0, 0, 8, 2, 20], method=method, random_state=4)
    for target in r.values():
        assert not target['supported'][0]
        assert np.isnan(target['lower'][0])


def test_cp_population_corners_allow_negative_gains():
    """Check a population interval whose gain box crosses zero."""
    r = infer_counts([10, 10, 10, 10, 60], a=-1, b=10, method='cp')
    assert r['H']['lower'][0] < 0 < r['H']['upper'][0]
    assert r['G']['lower'][0] < r['G']['upper'][0]


def test_zero_width_is_explicitly_unsupported():
    """Decline an empirical boundary collapse rather than calling it precise."""
    for method in ['wald', 'percentile', 'bca']:
        r = infer_counts([20, 0, 30, 0, 50], method=method, random_state=1)
        assert not r['G']['supported'][0]
    assert infer_counts([20, 0, 30, 0, 50], method='wilson')['G']['supported'][0]


def test_strict_boundary_decision_differs_from_legacy_weak_acceptance():
    """Keep a discrete threshold tie explicit instead of calling it positive."""
    from action_rules.inference.counts import decide_intervals

    r = {
        'lower': np.array([0.0, 0.1, np.nan]),
        'upper': np.array([0.0, 0.2, np.nan]),
        'supported': np.array([True, True, False]),
    }
    assert decide_intervals(r).tolist() == ['uncertain', 'accept', 'unsupported']
    assert decide_intervals(r, strict=False).tolist() == ['accept', 'accept', 'unsupported']


@pytest.mark.parametrize('method', ['wald', 'wilson', 'cp', 'percentile', 'bca', 'bayes_rd'])
def test_known_constant_gain_needs_no_jackknife_in_singleton_groups(method):
    """Keep a known utility coefficient separate from outcome uncertainty."""
    r = infer_counts([1, 0, 0, 1, 20], a=3, b=0, method=method, random_state=1)['G']
    assert r['supported'][0]
    np.testing.assert_array_equal([r['point'][0], r['lower'][0], r['upper'][0]], [3, 3, 3])
