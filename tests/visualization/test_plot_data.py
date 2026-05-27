"""Tests for the plot-data extraction helpers.

These helpers must not require matplotlib — they return plain Python /
NumPy structures that downstream consumers (article figure scripts, custom
dashboards) use to render their own polished figures.
"""

import numpy as np
import pytest

from action_rules.inference.base import ConfidenceIntervalResult
from action_rules.visualization.plots import (
    bootstrap_histogram_data,
    forest_plot_data,
    grouped_forest_plot_data,
    posterior_plot_data,
)


def _make_result(
    rule_index: int = 0,
    method: str = "bootstrap",
    uplift_point: float = 0.15,
    ci_lower: float = 0.05,
    ci_upper: float = 0.25,
    with_samples: bool = True,
    with_gain: bool = False,
) -> ConfidenceIntervalResult:
    rng = np.random.default_rng(seed=rule_index + 7)
    samples_uplift = rng.normal(loc=uplift_point, scale=0.05, size=200) if with_samples else None
    samples_gain = rng.normal(loc=2.5, scale=0.5, size=200) if (with_samples and with_gain) else None
    return ConfidenceIntervalResult(
        rule_index=rule_index,
        method=method,
        confidence_level=0.95,
        uplift_point=uplift_point,
        uplift_ci_lower=ci_lower,
        uplift_ci_upper=ci_upper,
        uplift_se=0.05,
        realistic_rule_gain_point=2.5 if with_gain else None,
        realistic_rule_gain_ci_lower=1.5 if with_gain else None,
        realistic_rule_gain_ci_upper=3.5 if with_gain else None,
        realistic_rule_gain_se=0.5 if with_gain else None,
        support=50,
        confidence=0.8,
        samples_uplift=samples_uplift,
        samples_gain=samples_gain,
    )


class TestBootstrapHistogramData:
    def test_returns_required_keys(self):
        data = bootstrap_histogram_data(_make_result(), bins=20)
        for key in (
            'samples',
            'hist',
            'bin_edges',
            'point',
            'ci_lower',
            'ci_upper',
            'confidence_level',
            'threshold',
            'rule_index',
            'method',
            'metric',
        ):
            assert key in data

    def test_bin_edges_length(self):
        data = bootstrap_histogram_data(_make_result(), bins=20)
        assert len(data['bin_edges']) == 21

    def test_threshold_passes_through(self):
        data = bootstrap_histogram_data(_make_result(), threshold=0.0)
        assert data['threshold'] == 0.0

    def test_threshold_none_passes_through(self):
        data = bootstrap_histogram_data(_make_result())
        assert data['threshold'] is None

    def test_raises_on_missing_samples(self):
        with pytest.raises(ValueError, match="samples are None"):
            bootstrap_histogram_data(_make_result(with_samples=False))

    def test_gain_metric(self):
        data = bootstrap_histogram_data(_make_result(with_gain=True), metric='realistic_rule_gain')
        assert data['metric'] == 'realistic_rule_gain'


class TestPosteriorPlotData:
    def test_returns_required_keys(self):
        scipy = pytest.importorskip("scipy")  # noqa: F841
        data = posterior_plot_data(_make_result(method='bayesian'))
        for key in (
            'samples',
            'x_grid',
            'kde_density',
            'point',
            'ci_lower',
            'ci_upper',
            'posterior_mean',
            'confidence_level',
        ):
            assert key in data

    def test_grid_shape(self):
        pytest.importorskip("scipy")
        data = posterior_plot_data(_make_result(method='bayesian'), n_grid=200)
        assert data['x_grid'].shape == (200,)
        assert data['kde_density'].shape == (200,)

    def test_raises_on_missing_samples(self):
        pytest.importorskip("scipy")
        with pytest.raises(ValueError, match="samples are None"):
            posterior_plot_data(_make_result(with_samples=False))


class TestForestPlotData:
    def test_empty_returns_zero_n(self):
        data = forest_plot_data([])
        assert data['n'] == 0
        assert data['labels'] == []

    def test_sorted_by_point_ascending(self):
        results = [
            _make_result(rule_index=0, uplift_point=0.30, ci_lower=0.20, ci_upper=0.40),
            _make_result(rule_index=1, uplift_point=0.10, ci_lower=0.00, ci_upper=0.20),
            _make_result(rule_index=2, uplift_point=0.20, ci_lower=0.10, ci_upper=0.30),
        ]
        data = forest_plot_data(results)
        # Lowest point comes first, highest last.
        assert data['points'] == [0.10, 0.20, 0.30]
        assert data['rule_indices'] == [1, 2, 0]

    def test_categorization_with_threshold(self):
        results = [
            _make_result(rule_index=0, uplift_point=0.30, ci_lower=0.10, ci_upper=0.50),  # accept
            _make_result(rule_index=1, uplift_point=-0.20, ci_lower=-0.40, ci_upper=-0.10),  # reject
            _make_result(rule_index=2, uplift_point=0.05, ci_lower=-0.05, ci_upper=0.15),  # uncertain
        ]
        data = forest_plot_data(results, threshold=0.0)
        # Already sorted ascending by point: reject, uncertain, accept.
        assert data['categories'] == ['reject', 'uncertain', 'accept']

    def test_categories_none_without_threshold(self):
        results = [_make_result(rule_index=i) for i in range(3)]
        data = forest_plot_data(results)
        assert data['categories'] == [None, None, None]

    def test_xerr_non_negative(self):
        results = [_make_result(rule_index=i, uplift_point=0.1 * i) for i in range(4)]
        data = forest_plot_data(results)
        assert all(e >= 0 for e in data['xerr_lower'])
        assert all(e >= 0 for e in data['xerr_upper'])

    def test_custom_labels(self):
        results = [_make_result(rule_index=i) for i in range(3)]
        data = forest_plot_data(results, labels=["A", "B", "C"])
        # Labels are re-sorted alongside points; checking the set is sufficient
        # because all three results have the same uplift_point and sort order
        # is implementation-defined among ties.
        assert set(data['labels']) == {"A", "B", "C"}

    def test_label_length_mismatch_raises(self):
        results = [_make_result(rule_index=i) for i in range(3)]
        with pytest.raises(ValueError, match="labels has length"):
            forest_plot_data(results, labels=["A", "B"])


class TestGroupedForestPlotData:
    def test_empty_dict(self):
        data = grouped_forest_plot_data({})
        assert data['n_rules'] == 0
        assert data['methods'] == []

    def test_single_method_offset_is_zero(self):
        results_dict = {'bootstrap': [_make_result(rule_index=i) for i in range(3)]}
        data = grouped_forest_plot_data(results_dict)
        assert data['offsets'] == [0.0]

    def test_multiple_methods_symmetric_offsets(self):
        results_dict = {
            'bootstrap': [_make_result(rule_index=i) for i in range(2)],
            'analytic': [_make_result(rule_index=i, method='analytic') for i in range(2)],
            'bayesian': [_make_result(rule_index=i, method='bayesian') for i in range(2)],
        }
        data = grouped_forest_plot_data(results_dict)
        # Symmetric about zero.
        assert data['offsets'][0] == -data['offsets'][-1]

    def test_rule_indices_union(self):
        results_dict = {
            'bootstrap': [_make_result(rule_index=0), _make_result(rule_index=1)],
            'analytic': [_make_result(rule_index=1, method='analytic'), _make_result(rule_index=2, method='analytic')],
        }
        data = grouped_forest_plot_data(results_dict)
        assert data['rule_indices'] == [0, 1, 2]

    def test_missing_rule_filled_with_none(self):
        results_dict = {
            'bootstrap': [_make_result(rule_index=0), _make_result(rule_index=1)],
            'analytic': [_make_result(rule_index=1, method='analytic')],
        }
        data = grouped_forest_plot_data(results_dict)
        # 'analytic' is missing rule_index 0 → first slot is None.
        analytic_points = data['per_method']['analytic']['points']
        assert analytic_points[0] is None
        assert analytic_points[1] is not None
