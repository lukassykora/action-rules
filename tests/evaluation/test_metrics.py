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
    def test_picks_top_score(self):
        scores = [0.1, 0.5, 0.3, 0.7, -0.2]
        outcomes = [0.05, 0.4, 0.2, 0.6, -0.3]
        # k=0.2 -> top 1 rule (rank-1 score 0.7 -> outcome 0.6)
        assert uplift_at_k(scores, outcomes, k_fraction=0.2) == pytest.approx(0.6)

    def test_top_40pct_averages_two(self):
        scores = [0.1, 0.5, 0.3, 0.7, -0.2]
        outcomes = [0.05, 0.4, 0.2, 0.6, -0.3]
        # top 2 by score: 0.7 -> 0.6, 0.5 -> 0.4; mean = 0.5
        assert uplift_at_k(scores, outcomes, k_fraction=0.4) == pytest.approx(0.5)

    def test_full_fraction_returns_mean(self):
        outcomes = [0.1, 0.2, 0.3, 0.4]
        assert uplift_at_k([1, 2, 3, 4], outcomes, k_fraction=1.0) == pytest.approx(np.mean(outcomes))

    def test_empty_returns_zero(self):
        assert uplift_at_k([], [], k_fraction=0.2) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            uplift_at_k([1.0, 2.0], [3.0, 4.0, 5.0])

    def test_invalid_k_fraction(self):
        with pytest.raises(ValueError):
            uplift_at_k([1.0], [1.0], k_fraction=0.0)
        with pytest.raises(ValueError):
            uplift_at_k([1.0], [1.0], k_fraction=1.5)


class TestProfitAndGainAtK:
    def test_profit_weighted_by_support(self):
        scores = [0.7, 0.5, 0.3]
        gains = [100.0, 200.0, 300.0]
        supports = [5, 3, 2]
        # k_fraction=0.33 -> ceil(0.99) = 1 rule (top 1 by score → gain=100, support=5)
        assert incremental_profit_at_k(scores, gains, supports, k_fraction=0.33) == pytest.approx(500.0)

    def test_profit_without_supports_sums_gains(self):
        scores = [0.7, 0.5, 0.3]
        gains = [100.0, 200.0, 300.0]
        # k_fraction=0.66 -> ceil(1.98) = 2 rules -> gains [100, 200] -> sum 300
        assert incremental_profit_at_k(scores, gains, k_fraction=0.66) == pytest.approx(300.0)

    def test_realistic_gain_at_k_is_uplift_at_k_alias_full_select(self):
        scores = [0.1, 0.4, 0.2]
        gains = [10.0, 40.0, 20.0]
        # k_fraction=1.0 → average of all gains
        assert realistic_gain_at_k(scores, gains, k_fraction=1.0) == pytest.approx((10.0 + 40.0 + 20.0) / 3)

    def test_realistic_gain_at_k_is_uplift_at_k_alias(self):
        scores = [0.1, 0.4, 0.2]
        gains = [10.0, 40.0, 20.0]
        assert realistic_gain_at_k(scores, gains, k_fraction=0.33) == pytest.approx(
            uplift_at_k(scores, gains, k_fraction=0.33)
        )


class TestQiniCurve:
    def test_starts_at_origin(self):
        scores = [0.5, 0.3, 0.7]
        outcomes = [0.2, 0.1, 0.4]
        x, y = qini_curve(scores, outcomes)
        assert x[0] == 0.0
        assert y[0] == 0.0

    def test_x_ends_at_one(self):
        scores = [0.5, 0.3, 0.7]
        outcomes = [0.2, 0.1, 0.4]
        x, _ = qini_curve(scores, outcomes)
        assert x[-1] == pytest.approx(1.0)

    def test_y_ends_at_one_when_total_positive(self):
        scores = [0.5, 0.3, 0.7]
        outcomes = [0.2, 0.1, 0.4]
        _, y = qini_curve(scores, outcomes)
        assert y[-1] == pytest.approx(1.0)

    def test_support_weighting_affects_x(self):
        scores = [0.7, 0.5, 0.3]
        outcomes = [0.4, 0.3, 0.1]
        supports = [1, 1, 8]
        x, _ = qini_curve(scores, outcomes, supports=supports)
        # First step covers only 1 out of 10 total support.
        assert x[1] == pytest.approx(0.1)

    def test_empty_input(self):
        x, y = qini_curve([], [])
        assert x.tolist() == [0.0]
        assert y.tolist() == [0.0]


class TestAuucAndQini:
    def test_perfect_ranking_high_auuc(self):
        # If ranking is perfectly aligned with outcomes, AUUC should be high.
        outcomes = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
        # Use outcomes themselves as scores → perfect alignment.
        good = auuc(outcomes, outcomes)
        bad = auuc(-outcomes, outcomes)
        assert good > bad

    def test_qini_equals_auuc_minus_half(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=20)
        outcomes = scores + rng.normal(scale=0.1, size=20)
        assert qini_coefficient(scores, outcomes) == pytest.approx(auuc(scores, outcomes) - 0.5)

    def test_random_ranking_qini_near_zero(self):
        rng = np.random.default_rng(0)
        n = 500
        outcomes = rng.normal(loc=0.1, scale=0.2, size=n)
        scores = rng.normal(size=n)  # uncorrelated with outcomes
        coef = qini_coefficient(scores, outcomes)
        # With 500 rules, |coef| should be quite small.
        assert abs(coef) < 0.10

    def test_auuc_empty(self):
        assert auuc([], []) == 0.0
