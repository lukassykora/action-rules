"""Tests for the rule-level targeting metrics.

These cover correctness of the elementary metrics on small hand-crafted
inputs plus property tests (random ranking ≈ 0 Qini, AUUC bounded, etc.).
"""

import numpy as np
import pytest

from action_rules.evaluation.metrics import (
    auuc,
    incremental_profit_at_k,
    qini_coefficient,
    qini_curve,
    realistic_gain_at_k,
    uplift_at_k,
)


class TestUpliftAtK:
    """Tests for the ``uplift_at_k`` metric."""

    def test_picks_top_score(self):
        """Check that k=0.2 selects only the single top-scoring rule's outcome."""
        scores = [0.1, 0.5, 0.3, 0.7, -0.2]
        outcomes = [0.05, 0.4, 0.2, 0.6, -0.3]
        # k=0.2 -> top 1 rule (rank-1 score 0.7 -> outcome 0.6)
        assert uplift_at_k(scores, outcomes, k_fraction=0.2) == pytest.approx(0.6)

    def test_top_40pct_averages_two(self):
        """Check that k=0.4 averages the outcomes of the top two rules."""
        scores = [0.1, 0.5, 0.3, 0.7, -0.2]
        outcomes = [0.05, 0.4, 0.2, 0.6, -0.3]
        # top 2 by score: 0.7 -> 0.6, 0.5 -> 0.4; mean = 0.5
        assert uplift_at_k(scores, outcomes, k_fraction=0.4) == pytest.approx(0.5)

    def test_full_fraction_returns_mean(self):
        """Check that k=1.0 returns the mean of all outcomes."""
        outcomes = [0.1, 0.2, 0.3, 0.4]
        assert uplift_at_k([1, 2, 3, 4], outcomes, k_fraction=1.0) == pytest.approx(np.mean(outcomes))

    def test_empty_returns_zero(self):
        """Check that empty inputs return 0.0."""
        assert uplift_at_k([], [], k_fraction=0.2) == 0.0

    def test_length_mismatch_raises(self):
        """Check that mismatched score and outcome lengths raise ``ValueError``."""
        with pytest.raises(ValueError):
            uplift_at_k([1.0, 2.0], [3.0, 4.0, 5.0])

    def test_invalid_k_fraction(self):
        """Check that out-of-range ``k_fraction`` values raise ``ValueError``."""
        with pytest.raises(ValueError):
            uplift_at_k([1.0], [1.0], k_fraction=0.0)
        with pytest.raises(ValueError):
            uplift_at_k([1.0], [1.0], k_fraction=1.5)


class TestProfitAndGainAtK:
    """Tests for the ``incremental_profit_at_k`` and ``realistic_gain_at_k`` metrics."""

    def test_profit_weighted_by_support(self):
        """Check that incremental profit weights the top rule's gain by its support."""
        scores = [0.7, 0.5, 0.3]
        gains = [100.0, 200.0, 300.0]
        supports = [5, 3, 2]
        # k_fraction=0.33 -> ceil(0.99) = 1 rule (top 1 by score → gain=100, support=5)
        assert incremental_profit_at_k(scores, gains, supports, k_fraction=0.33) == pytest.approx(500.0)

    def test_profit_without_supports_sums_gains(self):
        """Check that incremental profit sums the selected gains when no supports are given."""
        scores = [0.7, 0.5, 0.3]
        gains = [100.0, 200.0, 300.0]
        # k_fraction=0.66 -> ceil(1.98) = 2 rules -> gains [100, 200] -> sum 300
        assert incremental_profit_at_k(scores, gains, k_fraction=0.66) == pytest.approx(300.0)

    def test_realistic_gain_at_k_is_uplift_at_k_alias_full_select(self):
        """Check that ``realistic_gain_at_k`` averages all gains at k=1.0."""
        scores = [0.1, 0.4, 0.2]
        gains = [10.0, 40.0, 20.0]
        # k_fraction=1.0 → average of all gains
        assert realistic_gain_at_k(scores, gains, k_fraction=1.0) == pytest.approx((10.0 + 40.0 + 20.0) / 3)

    def test_realistic_gain_at_k_is_uplift_at_k_alias(self):
        """Check that ``realistic_gain_at_k`` matches ``uplift_at_k`` for partial k."""
        scores = [0.1, 0.4, 0.2]
        gains = [10.0, 40.0, 20.0]
        assert realistic_gain_at_k(scores, gains, k_fraction=0.33) == pytest.approx(
            uplift_at_k(scores, gains, k_fraction=0.33)
        )


class TestQiniCurve:
    """Tests for the ``qini_curve`` helper."""

    def test_starts_at_origin(self):
        """Check that the Qini curve starts at the origin (0, 0)."""
        scores = [0.5, 0.3, 0.7]
        outcomes = [0.2, 0.1, 0.4]
        x, y = qini_curve(scores, outcomes)
        assert x[0] == 0.0
        assert y[0] == 0.0

    def test_x_ends_at_one(self):
        """Check that the Qini curve's x-axis ends at 1.0."""
        scores = [0.5, 0.3, 0.7]
        outcomes = [0.2, 0.1, 0.4]
        x, _ = qini_curve(scores, outcomes)
        assert x[-1] == pytest.approx(1.0)

    def test_y_ends_at_one_when_total_positive(self):
        """Check that the Qini curve's y-axis ends at 1.0 when total outcome is positive."""
        scores = [0.5, 0.3, 0.7]
        outcomes = [0.2, 0.1, 0.4]
        _, y = qini_curve(scores, outcomes)
        assert y[-1] == pytest.approx(1.0)

    def test_support_weighting_affects_x(self):
        """Check that support weights rescale the x-axis steps of the Qini curve."""
        scores = [0.7, 0.5, 0.3]
        outcomes = [0.4, 0.3, 0.1]
        supports = [1, 1, 8]
        x, _ = qini_curve(scores, outcomes, supports=supports)
        # First step covers only 1 out of 10 total support.
        assert x[1] == pytest.approx(0.1)

    def test_empty_input(self):
        """Check that an empty Qini curve degenerates to a single origin point."""
        x, y = qini_curve([], [])
        assert x.tolist() == [0.0]
        assert y.tolist() == [0.0]


class TestAuucAndQini:
    """Tests for the ``auuc`` and ``qini_coefficient`` metrics."""

    def test_perfect_ranking_high_auuc(self):
        """Check that an aligned ranking yields higher AUUC than a reversed one."""
        # If ranking is perfectly aligned with outcomes, AUUC should be high.
        outcomes = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
        # Use outcomes themselves as scores → perfect alignment.
        good = auuc(outcomes, outcomes)
        bad = auuc(-outcomes, outcomes)
        assert good > bad

    def test_qini_equals_auuc_minus_half(self):
        """Check that the Qini coefficient equals AUUC minus 0.5."""
        rng = np.random.default_rng(0)
        scores = rng.normal(size=20)
        outcomes = scores + rng.normal(scale=0.1, size=20)
        assert qini_coefficient(scores, outcomes) == pytest.approx(auuc(scores, outcomes) - 0.5)

    def test_random_ranking_qini_near_zero(self):
        """Check that a random ranking gives a Qini coefficient near zero."""
        rng = np.random.default_rng(0)
        n = 500
        outcomes = rng.normal(loc=0.1, scale=0.2, size=n)
        scores = rng.normal(size=n)  # uncorrelated with outcomes
        coef = qini_coefficient(scores, outcomes)
        # With 500 rules, |coef| should be quite small.
        assert abs(coef) < 0.10

    def test_auuc_empty(self):
        """Check that AUUC of empty inputs is 0.0."""
        assert auuc([], []) == 0.0
