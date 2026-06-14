"""Smoke tests for the visualization module.

These tests verify that each plot function returns a Figure and does not raise
exceptions when given valid input. They do not perform pixel-level assertions.
matplotlib is a soft dependency — the entire module is skipped when it is not
installed.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

matplotlib = pytest.importorskip("matplotlib")

# Import after confirming matplotlib is present.
import matplotlib.pyplot as plt  # noqa: E402

from action_rules.inference.base import ConfidenceIntervalResult  # noqa: E402
from action_rules.visualization import (  # noqa: E402
    bootstrap_histogram,
    forest_plot,
    grouped_forest_plot,
    posterior_plot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    rule_index: int = 0,
    method: str = "bootstrap",
    uplift_point: float = 0.15,
    ci_lower: float = 0.05,
    ci_upper: float = 0.25,
    with_samples: bool = True,
    with_gain: bool = False,
    category=None,
) -> ConfidenceIntervalResult:
    """Build a minimal ConfidenceIntervalResult for testing."""
    rng = np.random.default_rng(seed=rule_index + 42)
    samples_uplift = rng.normal(loc=uplift_point, scale=0.05, size=1000) if with_samples else None
    samples_gain = rng.normal(loc=2.5, scale=0.5, size=1000) if (with_samples and with_gain) else None

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
        category=category,
        samples_uplift=samples_uplift,
        samples_gain=samples_gain,
    )


# ---------------------------------------------------------------------------
# bootstrap_histogram
# ---------------------------------------------------------------------------


class TestBootstrapHistogram:
    """Tests for bootstrap_histogram."""

    def teardown_method(self):
        """Close all figures after each test to keep memory usage low."""
        plt.close('all')

    def test_returns_figure(self):
        """bootstrap_histogram returns a matplotlib Figure."""
        result = _make_result()
        fig = bootstrap_histogram(result)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_threshold(self):
        """bootstrap_histogram with a threshold draws without error."""
        result = _make_result()
        fig = bootstrap_histogram(result, threshold=0.0)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_bins(self):
        """bootstrap_histogram respects the bins parameter."""
        result = _make_result()
        fig = bootstrap_histogram(result, bins=20)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_accepts_axes(self):
        """bootstrap_histogram uses a provided Axes and returns its Figure."""
        result = _make_result()
        fig_pre, ax = plt.subplots()
        fig = bootstrap_histogram(result, ax=ax)
        assert fig is fig_pre

    def test_raises_on_no_samples(self):
        """bootstrap_histogram raises ValueError when samples are None."""
        result = _make_result(with_samples=False)
        with pytest.raises(ValueError, match="samples are None"):
            bootstrap_histogram(result)

    def test_metric_realistic_rule_gain(self):
        """bootstrap_histogram works with realistic_rule_gain metric."""
        result = _make_result(with_gain=True)
        fig = bootstrap_histogram(result, metric='realistic_rule_gain')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_invalid_metric_raises(self):
        """bootstrap_histogram raises ValueError for unknown metric."""
        result = _make_result()
        with pytest.raises(ValueError, match="Unknown metric"):
            bootstrap_histogram(result, metric='nonexistent')


# ---------------------------------------------------------------------------
# posterior_plot
# ---------------------------------------------------------------------------


scipy = pytest.importorskip("scipy")


class TestPosteriorPlot:
    """Tests for posterior_plot."""

    def teardown_method(self):
        """Close all figures after each test to keep memory usage low."""
        plt.close('all')

    def test_returns_figure(self):
        """posterior_plot returns a matplotlib Figure."""
        result = _make_result(method="bayesian")
        fig = posterior_plot(result)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_accepts_axes(self):
        """posterior_plot uses a provided Axes and returns its Figure."""
        result = _make_result(method="bayesian")
        fig_pre, ax = plt.subplots()
        fig = posterior_plot(result, ax=ax)
        assert fig is fig_pre

    def test_raises_on_no_samples(self):
        """posterior_plot raises ValueError when samples are None."""
        result = _make_result(with_samples=False)
        with pytest.raises(ValueError, match="samples are None"):
            posterior_plot(result)

    def test_metric_realistic_rule_gain(self):
        """posterior_plot works with realistic_rule_gain metric."""
        result = _make_result(method="bayesian", with_gain=True)
        fig = posterior_plot(result, metric='realistic_rule_gain')
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# forest_plot
# ---------------------------------------------------------------------------


class TestForestPlot:
    """Tests for forest_plot."""

    def teardown_method(self):
        """Close all figures after each test to keep memory usage low."""
        plt.close('all')

    def test_returns_figure_single(self):
        """forest_plot returns a Figure for a single result."""
        results = [_make_result(rule_index=0)]
        fig = forest_plot(results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_multiple(self):
        """forest_plot returns a Figure for multiple results."""
        results = [_make_result(rule_index=i, uplift_point=0.05 * i) for i in range(5)]
        fig = forest_plot(results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_threshold_coloring(self):
        """forest_plot colour-codes rules when threshold is provided."""
        results = [
            _make_result(rule_index=0, uplift_point=0.3, ci_lower=0.1, ci_upper=0.5),  # accept
            _make_result(rule_index=1, uplift_point=-0.2, ci_lower=-0.4, ci_upper=-0.1),  # reject
            _make_result(rule_index=2, uplift_point=0.05, ci_lower=-0.05, ci_upper=0.15),  # uncertain
        ]
        fig = forest_plot(results, threshold=0.0)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_labels(self):
        """forest_plot accepts custom y-axis labels."""
        results = [_make_result(rule_index=i) for i in range(3)]
        fig = forest_plot(results, labels=["Rule A", "Rule B", "Rule C"])
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_results(self):
        """forest_plot handles an empty list without raising."""
        fig = forest_plot([])
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_accepts_axes(self):
        """forest_plot uses a provided Axes and returns its Figure."""
        results = [_make_result(rule_index=0)]
        fig_pre, ax = plt.subplots()
        fig = forest_plot(results, ax=ax)
        assert fig is fig_pre

    def test_auto_height_scaling(self):
        """forest_plot auto-scales the figure height for many rules."""
        results = [_make_result(rule_index=i) for i in range(20)]
        fig = forest_plot(results)
        # Height should be at least 4 inches.
        assert fig.get_size_inches()[1] >= 4

    def test_metric_realistic_rule_gain(self):
        """forest_plot works with realistic_rule_gain metric."""
        results = [_make_result(rule_index=i, with_gain=True) for i in range(3)]
        fig = forest_plot(results, metric='realistic_rule_gain')
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# grouped_forest_plot
# ---------------------------------------------------------------------------


class TestGroupedForestPlot:
    """Tests for grouped_forest_plot."""

    def teardown_method(self):
        """Close all figures after each test to keep memory usage low."""
        plt.close('all')

    def test_returns_figure_single_method(self):
        """grouped_forest_plot works with a single method."""
        results_dict = {
            'bootstrap': [_make_result(rule_index=i, uplift_point=0.1 * i) for i in range(3)],
        }
        fig = grouped_forest_plot(results_dict)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_multiple_methods(self):
        """grouped_forest_plot overlays multiple methods."""
        results_dict = {
            'bootstrap': [_make_result(rule_index=i, method='bootstrap') for i in range(3)],
            'analytic': [_make_result(rule_index=i, method='analytic') for i in range(3)],
            'bayesian': [_make_result(rule_index=i, method='bayesian') for i in range(3)],
        }
        fig = grouped_forest_plot(results_dict)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_threshold(self):
        """grouped_forest_plot draws threshold line without error."""
        results_dict = {
            'bootstrap': [_make_result(rule_index=i) for i in range(2)],
            'analytic': [_make_result(rule_index=i, method='analytic') for i in range(2)],
        }
        fig = grouped_forest_plot(results_dict, threshold=0.0)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_dict(self):
        """grouped_forest_plot handles an empty dict without raising."""
        fig = grouped_forest_plot({})
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_accepts_axes(self):
        """grouped_forest_plot uses a provided Axes and returns its Figure."""
        results_dict = {
            'bootstrap': [_make_result(rule_index=0)],
        }
        fig_pre, ax = plt.subplots()
        fig = grouped_forest_plot(results_dict, ax=ax)
        assert fig is fig_pre

    def test_partial_rule_overlap(self):
        """grouped_forest_plot handles methods that don't share all rule indices."""
        results_dict = {
            'bootstrap': [_make_result(rule_index=0), _make_result(rule_index=1)],
            'analytic': [_make_result(rule_index=1), _make_result(rule_index=2)],
        }
        fig = grouped_forest_plot(results_dict)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_metric_realistic_rule_gain(self):
        """grouped_forest_plot works with realistic_rule_gain metric."""
        results_dict = {
            'bootstrap': [_make_result(rule_index=i, with_gain=True) for i in range(2)],
            'analytic': [_make_result(rule_index=i, method='analytic', with_gain=True) for i in range(2)],
        }
        fig = grouped_forest_plot(results_dict, metric='realistic_rule_gain')
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# Regression: threshold / zero line must not stretch x-axis
# ---------------------------------------------------------------------------


class TestXAxisNotStretchedByReferenceLines:
    """Regression tests that reference lines do not stretch the x-axis.

    Covers the bug where threshold=0.0 or the zero reference line forces the
    x-axis to include 0, creating a huge empty space when the data is far from
    zero (e.g. realistic_rule_gain ~ 250).
    """

    def teardown_method(self):
        """Close all figures after each test to keep memory usage low."""
        plt.close('all')

    def _make_far_from_zero_result(self, rule_index=0):
        """Create a result with data centred around 250, far from 0."""
        rng = np.random.default_rng(seed=rule_index + 99)
        samples = rng.normal(loc=250.0, scale=15.0, size=500)
        return ConfidenceIntervalResult(
            rule_index=rule_index,
            method='bootstrap',
            confidence_level=0.95,
            uplift_point=250.0,
            uplift_ci_lower=230.0,
            uplift_ci_upper=270.0,
            uplift_se=10.0,
            realistic_rule_gain_point=250.0,
            realistic_rule_gain_ci_lower=230.0,
            realistic_rule_gain_ci_upper=270.0,
            realistic_rule_gain_se=10.0,
            support=50,
            confidence=0.8,
            category=None,
            samples_uplift=samples,
            samples_gain=samples.copy(),
        )

    def test_bootstrap_histogram_xlim_not_stretched(self):
        """bootstrap_histogram with threshold=0.0 should not include 0 in xlim when data is at ~250."""
        result = self._make_far_from_zero_result()
        fig = bootstrap_histogram(result, metric='realistic_rule_gain', threshold=0.0)
        ax = fig.axes[0]
        x_lo, x_hi = ax.get_xlim()
        # x_lo should be near 200, not near 0
        assert x_lo > 100, f"x_lo={x_lo} is too low — threshold stretched the axis"

    def test_forest_plot_xlim_not_stretched(self):
        """forest_plot with threshold=0.0 should not include 0 in xlim when data is at ~250."""
        results = [self._make_far_from_zero_result(i) for i in range(3)]
        fig = forest_plot(results, metric='realistic_rule_gain', threshold=0.0)
        ax = fig.axes[0]
        x_lo, x_hi = ax.get_xlim()
        assert x_lo > 100, f"x_lo={x_lo} is too low — threshold/zero line stretched the axis"

    def test_forest_plot_with_categories_xlim_not_stretched(self):
        """forest_plot with show_categories=True and threshold=0.0 should not stretch."""
        results = [self._make_far_from_zero_result(i) for i in range(3)]
        fig = forest_plot(results, metric='realistic_rule_gain', threshold=0.0, show_categories=True)
        ax = fig.axes[0]
        x_lo, x_hi = ax.get_xlim()
        assert x_lo > 100, f"x_lo={x_lo} is too low — threshold stretched the axis"

    def test_grouped_forest_plot_xlim_not_stretched(self):
        """grouped_forest_plot with threshold=0.0 should not include 0 in xlim when data is at ~250."""
        results = [self._make_far_from_zero_result(i) for i in range(3)]
        results_dict = {'bootstrap': results}
        fig = grouped_forest_plot(results_dict, metric='realistic_rule_gain', threshold=0.0)
        ax = fig.axes[0]
        x_lo, x_hi = ax.get_xlim()
        assert x_lo > 100, f"x_lo={x_lo} is too low — threshold/zero line stretched the axis"
