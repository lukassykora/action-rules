"""Tests for the shared inference helpers in :mod:`action_rules.inference.base`.

Focuses on the small, pure helpers that have no engine of their own:
``ConfidenceIntervalResult.to_dict``, ``results_to_dataframe``, and
``categorize_rule``. The engine-specific behaviours are covered by
``test_bootstrap.py``, ``test_analytic.py``, and ``test_bayesian.py``.
"""

import numpy as np
import pandas as pd

from action_rules.inference.base import (
    ConfidenceIntervalResult,
    RuleCategory,
    categorize_rule,
    results_to_dataframe,
)


def _make_result(
    rule_index: int = 0,
    *,
    with_gain: bool = False,
    with_samples: bool = False,
    category=None,
) -> ConfidenceIntervalResult:
    """Build a minimal result for tests."""
    rng = np.random.default_rng(seed=rule_index + 1)
    samples_uplift = rng.normal(0.1, 0.02, size=50) if with_samples else None
    samples_gain = rng.normal(2.0, 0.5, size=50) if (with_samples and with_gain) else None
    return ConfidenceIntervalResult(
        rule_index=rule_index,
        method='bootstrap',
        confidence_level=0.95,
        uplift_point=0.10,
        uplift_ci_lower=0.05,
        uplift_ci_upper=0.15,
        uplift_se=0.025,
        realistic_rule_gain_point=2.0 if with_gain else None,
        realistic_rule_gain_ci_lower=1.0 if with_gain else None,
        realistic_rule_gain_ci_upper=3.0 if with_gain else None,
        realistic_rule_gain_se=0.5 if with_gain else None,
        support=42,
        confidence=0.81,
        category=category,
        samples_uplift=samples_uplift,
        samples_gain=samples_gain,
    )


class TestToDict:
    """Tests for ``ConfidenceIntervalResult.to_dict``."""

    def test_returns_dict_with_core_keys(self):
        """Verify the serialized dict contains all core result keys."""
        r = _make_result()
        d = r.to_dict()
        for key in (
            'rule_index',
            'method',
            'confidence_level',
            'uplift_point',
            'uplift_ci_lower',
            'uplift_ci_upper',
            'uplift_se',
            'realistic_rule_gain_point',
            'realistic_rule_gain_ci_lower',
            'realistic_rule_gain_ci_upper',
            'realistic_rule_gain_se',
            'support',
            'confidence',
            'category',
        ):
            assert key in d, f"missing key: {key}"

    def test_excludes_samples_by_default(self):
        """Verify sample arrays are omitted from the dict by default."""
        r = _make_result(with_samples=True)
        d = r.to_dict()
        assert 'samples_uplift' not in d
        assert 'samples_gain' not in d

    def test_includes_samples_when_requested(self):
        """Verify sample arrays are included as lists when requested."""
        r = _make_result(with_samples=True, with_gain=True)
        d = r.to_dict(include_samples=True)
        assert isinstance(d['samples_uplift'], list)
        assert isinstance(d['samples_gain'], list)
        assert len(d['samples_uplift']) == 50

    def test_samples_none_when_engine_did_not_produce_them(self):
        """Verify sample keys are None when no samples were produced."""
        r = _make_result(with_samples=False)
        d = r.to_dict(include_samples=True)
        assert d['samples_uplift'] is None
        assert d['samples_gain'] is None

    def test_category_serialized_as_string(self):
        """Verify a RuleCategory is serialized to its string value."""
        r = _make_result(category=RuleCategory.ACCEPT)
        assert r.to_dict()['category'] == 'accept'

    def test_category_none_passed_through(self):
        """Verify a None category is passed through unchanged."""
        r = _make_result(category=None)
        assert r.to_dict()['category'] is None


class TestResultsToDataFrame:
    """Tests for ``results_to_dataframe``."""

    def test_empty_list_returns_empty_frame(self):
        """Verify an empty input list yields an empty DataFrame."""
        df = results_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_row_count_matches_input(self):
        """Verify the DataFrame row count matches the number of results."""
        results = [_make_result(i) for i in range(4)]
        df = results_to_dataframe(results)
        assert len(df) == 4

    def test_columns_present(self):
        """Verify expected core columns are present in the DataFrame."""
        results = [_make_result(0, with_gain=True, category=RuleCategory.ACCEPT)]
        df = results_to_dataframe(results)
        for col in ('rule_index', 'uplift_point', 'realistic_rule_gain_point', 'category'):
            assert col in df.columns

    def test_samples_excluded_by_default(self):
        """Verify sample columns are excluded from the DataFrame by default."""
        results = [_make_result(0, with_samples=True)]
        df = results_to_dataframe(results)
        assert 'samples_uplift' not in df.columns

    def test_samples_included_when_requested(self):
        """Verify sample columns are included in the DataFrame when requested."""
        results = [_make_result(0, with_samples=True)]
        df = results_to_dataframe(results, include_samples=True)
        assert 'samples_uplift' in df.columns

    def test_categories_serialized_as_strings(self):
        """Verify category values are serialized to strings in the DataFrame."""
        results = [
            _make_result(0, category=RuleCategory.ACCEPT),
            _make_result(1, category=RuleCategory.REJECT),
            _make_result(2, category=RuleCategory.UNCERTAIN),
            _make_result(3, category=None),
        ]
        df = results_to_dataframe(results)
        assert df['category'].tolist() == ['accept', 'reject', 'uncertain', None]


class TestCategorizeRule:
    """Sanity checks for the ``categorize_rule`` helper."""

    def test_accept_when_ci_above_threshold(self):
        """Verify a CI entirely above the threshold yields ACCEPT."""
        assert categorize_rule(0.1, 0.3, threshold=0.0) is RuleCategory.ACCEPT

    def test_reject_when_ci_below_threshold(self):
        """Verify a CI entirely below the threshold yields REJECT."""
        assert categorize_rule(-0.3, -0.1, threshold=0.0) is RuleCategory.REJECT

    def test_uncertain_when_ci_straddles(self):
        """Verify a CI straddling the threshold yields UNCERTAIN."""
        assert categorize_rule(-0.1, 0.1, threshold=0.0) is RuleCategory.UNCERTAIN

    def test_accept_when_lower_equals_threshold(self):
        """Verify a CI lower bound equal to the threshold yields ACCEPT."""
        # Boundary case: ci_lower exactly at threshold counts as ACCEPT.
        assert categorize_rule(0.0, 0.2, threshold=0.0) is RuleCategory.ACCEPT
