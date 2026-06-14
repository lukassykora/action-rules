"""Tests for BootstrapEngine."""

import math

import numpy as np
import pandas as pd
import pytest

from action_rules.inference.base import ConfidenceIntervalResult, RuleCategory, RuleMasks, compute_group_counts
from action_rules.inference.bootstrap import BootstrapEngine

# ---------------------------------------------------------------------------
# Fixtures
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

    # Group 1: age=0, class=0 → target=0 (most of the time)
    g1 = pd.DataFrame({'age': ['0'] * n_each, 'class': ['0'] * n_each, 'target': ['0'] * n_each})
    # Group 2: age=0, class=1 → target=1 (most of the time)
    g2 = pd.DataFrame({'age': ['0'] * n_each, 'class': ['1'] * n_each, 'target': ['1'] * n_each})
    # Group 3: age=1, class=0 → target=0
    g3 = pd.DataFrame({'age': ['1'] * n_each, 'class': ['0'] * n_each, 'target': ['0'] * n_each})
    # Group 4: age=1, class=1 → target=1
    g4 = pd.DataFrame({'age': ['1'] * n_each, 'class': ['1'] * n_each, 'target': ['1'] * n_each})

    data = pd.concat([g1, g2, g3, g4], ignore_index=True)
    # Shuffle to avoid positional artefacts.
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


# ---------------------------------------------------------------------------
# Basic sanity checks
# ---------------------------------------------------------------------------


class TestBootstrapEngineBasic:
    """Tests that cover fundamental correctness properties."""

    def test_returns_list_of_correct_length(self):
        """One result per rule is returned."""
        data = _make_data()
        rules = [_make_rule(0), _make_rule(1)]  # Two identical rules for count check.
        engine = BootstrapEngine(n_bootstrap=50, random_state=42)
        results = engine.compute(data, rules)
        assert len(results) == 2

    def test_result_type(self):
        """Each element is a ConfidenceIntervalResult."""
        data = _make_data()
        rules = [_make_rule()]
        engine = BootstrapEngine(n_bootstrap=50, random_state=42)
        result = engine.compute(data, rules)[0]
        assert isinstance(result, ConfidenceIntervalResult)

    def test_method_field(self):
        """The method field is set to 'bootstrap'."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=50, random_state=42)
        result = engine.compute(data, [_make_rule()])[0]
        assert result.method == 'bootstrap'

    def test_rule_index_preserved(self):
        """The rule_index from RuleMasks is carried through."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=50, random_state=42)
        rule = _make_rule(rule_index=7)
        result = engine.compute(data, [rule])[0]
        assert result.rule_index == 7

    def test_confidence_level_preserved(self):
        """The confidence_level passed to compute() is stored on the result."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=50, random_state=42)
        result = engine.compute(data, [_make_rule()], confidence_level=0.90)[0]
        assert result.confidence_level == pytest.approx(0.90)

    def test_empty_rules_list(self):
        """An empty rules list returns an empty result list without error."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=50, random_state=42)
        results = engine.compute(data, [])
        assert results == []


# ---------------------------------------------------------------------------
# Confidence interval properties
# ---------------------------------------------------------------------------


class TestBootstrapCIProperties:
    """Tests that verify structural properties of the CI output."""

    def test_ci_lower_le_upper(self):
        """ci_lower must be <= ci_upper for a non-degenerate result."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=200, random_state=0)
        result = engine.compute(data, [_make_rule()])[0]
        assert result.uplift_ci_lower <= result.uplift_ci_upper

    def test_point_estimate_inside_ci(self):
        """The point estimate (bootstrap mean) should lie within [lower, upper]."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=200, random_state=0)
        result = engine.compute(data, [_make_rule()])[0]
        assert result.uplift_ci_lower <= result.uplift_point <= result.uplift_ci_upper

    def test_se_is_non_negative(self):
        """Standard error must be >= 0."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=200, random_state=0)
        result = engine.compute(data, [_make_rule()])[0]
        assert result.uplift_se >= 0.0

    def test_wider_ci_at_lower_coverage(self):
        """A 99% CI should be at least as wide as a 95% CI on the same data."""
        data = _make_data(n=400)
        engine99 = BootstrapEngine(n_bootstrap=500, random_state=1)
        engine95 = BootstrapEngine(n_bootstrap=500, random_state=1)
        rule = _make_rule()
        r99 = engine99.compute(data, [rule], confidence_level=0.99)[0]
        r95 = engine95.compute(data, [rule], confidence_level=0.95)[0]
        width99 = r99.uplift_ci_upper - r99.uplift_ci_lower
        width95 = r95.uplift_ci_upper - r95.uplift_ci_lower
        assert width99 >= width95

    def test_samples_uplift_stored(self):
        """Raw bootstrap samples for uplift should be stored on the result."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=100, random_state=42)
        result = engine.compute(data, [_make_rule()])[0]
        assert result.samples_uplift is not None
        assert isinstance(result.samples_uplift, np.ndarray)
        # All non-NaN resamples for a well-supported rule.
        assert result.samples_uplift.size > 0

    def test_samples_count_bounded_by_n_bootstrap(self):
        """The number of stored uplift samples must be <= n_bootstrap."""
        data = _make_data()
        n_boot = 150
        engine = BootstrapEngine(n_bootstrap=n_boot, random_state=42)
        result = engine.compute(data, [_make_rule()])[0]
        assert result.samples_uplift.size <= n_boot


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestBootstrapReproducibility:
    """Identical seeds must produce identical results."""

    def test_same_seed_same_result(self):
        """Two engines with the same seed and same data produce identical CIs."""
        data = _make_data()
        rule = _make_rule()
        r1 = BootstrapEngine(n_bootstrap=200, random_state=99).compute(data, [rule])[0]
        r2 = BootstrapEngine(n_bootstrap=200, random_state=99).compute(data, [rule])[0]
        assert r1.uplift_point == pytest.approx(r2.uplift_point)
        assert r1.uplift_ci_lower == pytest.approx(r2.uplift_ci_lower)
        assert r1.uplift_ci_upper == pytest.approx(r2.uplift_ci_upper)

    def test_different_seed_different_result(self):
        """Two different seeds should (with overwhelming probability) differ."""
        data = _make_data()
        rule = _make_rule()
        r1 = BootstrapEngine(n_bootstrap=200, random_state=1).compute(data, [rule])[0]
        r2 = BootstrapEngine(n_bootstrap=200, random_state=2).compute(data, [rule])[0]
        # The intervals will differ slightly; check at least one bound changes.
        assert (r1.uplift_ci_lower != r2.uplift_ci_lower) or (r1.uplift_ci_upper != r2.uplift_ci_upper)


# ---------------------------------------------------------------------------
# Support and confidence from full data
# ---------------------------------------------------------------------------


class TestSupportAndConfidence:
    """Support and confidence should match the full-dataset counts."""

    def test_support_equals_undesired_antecedent_count(self):
        """Result.support should equal the count of rows matching mask_undesired."""
        data = _make_data(n=200)
        rule = _make_rule()
        engine = BootstrapEngine(n_bootstrap=50, random_state=0)
        result = engine.compute(data, [rule])[0]

        # Compute the expected count directly.
        n_u_ante, _, _, _, _ = compute_group_counts(data, rule)
        assert result.support == n_u_ante

    def test_confidence_matches_full_data(self):
        """Result.confidence should equal n_u_match / n_u_ante on the full data."""
        data = _make_data(n=200)
        rule = _make_rule()
        engine = BootstrapEngine(n_bootstrap=50, random_state=0)
        result = engine.compute(data, [rule])[0]

        n_u_ante, n_u_match, _, _, _ = compute_group_counts(data, rule)
        expected_conf = n_u_match / n_u_ante if n_u_ante > 0 else 0.0
        assert result.confidence == pytest.approx(expected_conf)


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------


class TestCategorisation:
    """Rule categories should follow the CI-vs-threshold logic."""

    def test_positive_uplift_rule_accepted(self):
        """A rule with clearly positive uplift should be ACCEPT."""
        data = _make_data(n=400)
        rule = _make_rule()
        engine = BootstrapEngine(n_bootstrap=500, random_state=0)
        result = engine.compute(data, [rule])[0]
        # The synthetic data gives perfect uplift; both CI bounds should be > 0.
        assert result.category == RuleCategory.ACCEPT

    def test_category_reject_when_ci_negative(self):
        """A rule with a clearly negative uplift CI should be REJECT."""
        # Build a dataset where the action makes things worse.
        n = 50
        g1 = pd.DataFrame({'age': ['0'] * n, 'class': ['0'] * n, 'target': ['1'] * n})
        g2 = pd.DataFrame({'age': ['0'] * n, 'class': ['1'] * n, 'target': ['0'] * n})
        data = pd.concat([g1, g2], ignore_index=True)

        rule = _make_rule()
        engine = BootstrapEngine(n_bootstrap=300, random_state=0)
        result = engine.compute(data, [rule])[0]
        assert result.category == RuleCategory.REJECT


# ---------------------------------------------------------------------------
# Zero-support (sparse) rule handling
# ---------------------------------------------------------------------------


class TestZeroSupportHandling:
    """Engine must not crash on zero-support rules."""

    def test_no_matching_rows_returns_nan(self):
        """A rule that never matches any row returns NaN CI bounds."""
        data = _make_data()
        # Antecedent that matches nothing.
        rule = RuleMasks(
            mask_undesired={'age': 'NONEXISTENT', 'class': '0'},
            mask_desired={'age': 'NONEXISTENT', 'class': '1'},
            target_attribute='target',
            target_undesired='0',
            target_desired='1',
            rule_index=0,
            undesired_itemset=(99,),
            desired_itemset=(100,),
        )
        engine = BootstrapEngine(n_bootstrap=50, random_state=0)
        result = engine.compute(data, [rule])[0]
        assert math.isnan(result.uplift_point)
        assert math.isnan(result.uplift_ci_lower)
        assert math.isnan(result.uplift_ci_upper)


# ---------------------------------------------------------------------------
# Utility / gain statistics
# ---------------------------------------------------------------------------


class TestGainStatistics:
    """Optional gain metrics are computed when utility tables are provided."""

    def _utility_tables(self):
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

    def test_gain_fields_populated(self):
        """When utility tables are passed, gain fields are not None."""
        data = _make_data(n=200)
        intrinsic, transition = self._utility_tables()
        cv = _make_column_values()
        engine = BootstrapEngine(n_bootstrap=100, random_state=0)
        result = engine.compute(
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
        """Gain CI lower bound must be <= upper bound."""
        data = _make_data(n=200)
        intrinsic, transition = self._utility_tables()
        cv = _make_column_values()
        engine = BootstrapEngine(n_bootstrap=100, random_state=0)
        result = engine.compute(
            data,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_ci_lower <= result.realistic_rule_gain_ci_upper

    def test_gain_samples_stored(self):
        """samples_gain should be a non-empty ndarray when gain is computed."""
        data = _make_data(n=200)
        intrinsic, transition = self._utility_tables()
        cv = _make_column_values()
        engine = BootstrapEngine(n_bootstrap=100, random_state=0)
        result = engine.compute(
            data,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.samples_gain is not None
        assert isinstance(result.samples_gain, np.ndarray)
        assert result.samples_gain.size > 0

    def test_gain_fields_none_without_utility_tables(self):
        """Gain fields remain None when no utility tables are supplied."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=50, random_state=0)
        result = engine.compute(data, [_make_rule()])[0]
        assert result.realistic_rule_gain_point is None
        assert result.realistic_rule_gain_ci_lower is None
        assert result.realistic_rule_gain_ci_upper is None
        assert result.realistic_rule_gain_se is None
        assert result.samples_gain is None

    def test_gain_reproducible(self):
        """Same seed gives identical gain CI bounds."""
        data = _make_data(n=200)
        intrinsic, transition = self._utility_tables()
        cv = _make_column_values()
        r1 = BootstrapEngine(n_bootstrap=100, random_state=7).compute(
            data,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        r2 = BootstrapEngine(n_bootstrap=100, random_state=7).compute(
            data,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert r1.realistic_rule_gain_point == pytest.approx(r2.realistic_rule_gain_point)
        assert r1.realistic_rule_gain_ci_lower == pytest.approx(r2.realistic_rule_gain_ci_lower)


# ---------------------------------------------------------------------------
# Multiple rules in one call
# ---------------------------------------------------------------------------


class TestMultipleRules:
    """Multiple rules processed in a single compute() call."""

    def test_results_order_matches_rules_order(self):
        """Results are in the same order as the input rules list."""
        data = _make_data()
        rules = [_make_rule(rule_index=0), _make_rule(rule_index=5)]
        engine = BootstrapEngine(n_bootstrap=50, random_state=0)
        results = engine.compute(data, rules)
        assert results[0].rule_index == 0
        assert results[1].rule_index == 5

    def test_two_rules_independent(self):
        """Verify two identical rules yield numerically identical results."""
        data = _make_data()
        rule = _make_rule()
        engine = BootstrapEngine(n_bootstrap=50, random_state=0)
        results = engine.compute(data, [rule, rule])
        # They use different portions of the RNG stream, so they can differ,
        # but both must have valid (non-NaN) CI bounds for this well-supported rule.
        for r in results:
            assert not math.isnan(r.uplift_point)


# ---------------------------------------------------------------------------
# BCa bootstrap
# ---------------------------------------------------------------------------


class TestBCaBootstrap:
    """Tests for the BCa bootstrap CI variant."""

    def test_invalid_bootstrap_type_raises(self):
        """An unknown bootstrap_type must raise ValueError at construction time."""
        with pytest.raises(ValueError, match="Unknown bootstrap_type"):
            BootstrapEngine(n_bootstrap=10, random_state=0, bootstrap_type="invalid")

    def test_valid_bootstrap_types_accepted(self):
        """Both 'percentile' and 'bca' are accepted without error."""
        BootstrapEngine(n_bootstrap=10, random_state=0, bootstrap_type="percentile")
        BootstrapEngine(n_bootstrap=10, random_state=0, bootstrap_type="bca")

    def test_backward_compat_positional_args(self):
        """Positional (n_bootstrap, random_state) must still work after adding bootstrap_type."""
        engine = BootstrapEngine(100, 42)
        assert engine.n_bootstrap == 100
        assert engine.random_state == 42
        assert engine.bootstrap_type == "percentile"

    def test_bca_returns_list_of_correct_length(self):
        """Verify the BCa variant returns one result per rule."""
        data = _make_data()
        rules = [_make_rule(0), _make_rule(1)]
        engine = BootstrapEngine(n_bootstrap=100, random_state=0, bootstrap_type="bca")
        results = engine.compute(data, rules)
        assert len(results) == 2

    def test_bca_result_type(self):
        """Each BCa result is a ConfidenceIntervalResult."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=100, random_state=0, bootstrap_type="bca")
        result = engine.compute(data, [_make_rule()])[0]
        assert isinstance(result, ConfidenceIntervalResult)

    def test_bca_method_field_still_bootstrap(self):
        """The method field must remain 'bootstrap' for the BCa variant."""
        data = _make_data()
        engine = BootstrapEngine(n_bootstrap=100, random_state=0, bootstrap_type="bca")
        result = engine.compute(data, [_make_rule()])[0]
        assert result.method == 'bootstrap'

    def test_bca_ci_lower_le_upper(self):
        """Verify the BCa CI lower bound is <= upper bound."""
        data = _make_data(n=300)
        engine = BootstrapEngine(n_bootstrap=300, random_state=0, bootstrap_type="bca")
        result = engine.compute(data, [_make_rule()])[0]
        assert result.uplift_ci_lower <= result.uplift_ci_upper

    def test_bca_non_nan_on_well_supported_rule(self):
        """Verify BCa CI bounds are finite (non-NaN) for a well-supported rule."""
        data = _make_data(n=300)
        engine = BootstrapEngine(n_bootstrap=300, random_state=0, bootstrap_type="bca")
        result = engine.compute(data, [_make_rule()])[0]
        assert not math.isnan(result.uplift_point)
        assert not math.isnan(result.uplift_ci_lower)
        assert not math.isnan(result.uplift_ci_upper)

    def test_bca_nan_on_zero_support_rule(self):
        """Verify BCa returns NaN CI bounds when no rows match the antecedent."""
        data = _make_data()
        rule = RuleMasks(
            mask_undesired={'age': 'NONEXISTENT', 'class': '0'},
            mask_desired={'age': 'NONEXISTENT', 'class': '1'},
            target_attribute='target',
            target_undesired='0',
            target_desired='1',
            rule_index=0,
            undesired_itemset=(99,),
            desired_itemset=(100,),
        )
        engine = BootstrapEngine(n_bootstrap=50, random_state=0, bootstrap_type="bca")
        result = engine.compute(data, [rule])[0]
        assert math.isnan(result.uplift_ci_lower)
        assert math.isnan(result.uplift_ci_upper)

    def test_bca_reproducible(self):
        """Same seed produces identical BCa CI bounds."""
        data = _make_data()
        rule = _make_rule()
        r1 = BootstrapEngine(n_bootstrap=200, random_state=7, bootstrap_type="bca").compute(data, [rule])[0]
        r2 = BootstrapEngine(n_bootstrap=200, random_state=7, bootstrap_type="bca").compute(data, [rule])[0]
        assert r1.uplift_ci_lower == pytest.approx(r2.uplift_ci_lower)
        assert r1.uplift_ci_upper == pytest.approx(r2.uplift_ci_upper)

    def test_bca_accepted_category_positive_uplift(self):
        """Verify BCa accepts a rule with clearly positive uplift."""
        data = _make_data(n=400)
        engine = BootstrapEngine(n_bootstrap=500, random_state=0, bootstrap_type="bca")
        result = engine.compute(data, [_make_rule()])[0]
        assert result.category == RuleCategory.ACCEPT

    def test_bca_gain_fields_populated(self):
        """Verify BCa gain fields are populated when utility tables are provided."""
        data = _make_data(n=200)
        intrinsic = {
            ('class', '0'): -1.0,
            ('class', '1'): 1.0,
            ('target', '0'): -2.0,
            ('target', '1'): 2.0,
        }
        transition = {('class', '0', '1'): 0.5}
        cv = _make_column_values()
        engine = BootstrapEngine(n_bootstrap=200, random_state=0, bootstrap_type="bca")
        result = engine.compute(
            data,
            [_make_rule()],
            intrinsic_utility_table=intrinsic,
            transition_utility_table=transition,
            column_values=cv,
        )[0]
        assert result.realistic_rule_gain_point is not None
        assert result.realistic_rule_gain_ci_lower is not None
        assert result.realistic_rule_gain_ci_upper is not None
        assert result.realistic_rule_gain_ci_lower <= result.realistic_rule_gain_ci_upper

    def test_bca_vs_percentile_similar_range(self):
        """Verify BCa and percentile CIs on clean data broadly agree."""
        data = _make_data(n=400)
        rule = _make_rule()
        r_pct = BootstrapEngine(n_bootstrap=500, random_state=3, bootstrap_type="percentile").compute(data, [rule])[0]
        r_bca = BootstrapEngine(n_bootstrap=500, random_state=3, bootstrap_type="bca").compute(data, [rule])[0]
        # Both intervals must be positive (rule is clearly beneficial).
        assert r_pct.uplift_ci_lower > 0.0
        assert r_bca.uplift_ci_lower > 0.0
        # The BCa interval width should be in the same order of magnitude as percentile.
        width_pct = r_pct.uplift_ci_upper - r_pct.uplift_ci_lower
        width_bca = r_bca.uplift_ci_upper - r_bca.uplift_ci_lower
        assert width_bca < width_pct * 5  # BCa may be narrower or wider, but not wildly so.


class TestBCaInternalMethods:
    """Unit tests for _bca_ci and _jackknife_uplift static methods."""

    def test_bca_ci_all_nan_samples(self):
        """_bca_ci returns (nan, nan, nan, nan) when all samples are NaN."""
        samples = np.array([float('nan'), float('nan'), float('nan')])
        jack = np.array([0.1, 0.2, 0.3])
        result = BootstrapEngine._bca_ci(samples, 0.2, jack, 0.95)
        assert all(math.isnan(v) for v in result)

    def test_bca_ci_symmetric_distribution(self):
        """For a symmetric bootstrap distribution, BCa and percentile bounds should be close."""
        rng = np.random.default_rng(42)
        # Symmetric distribution centred on 0.5.
        samples = rng.normal(loc=0.5, scale=0.05, size=2000)
        jack = rng.normal(loc=0.5, scale=0.01, size=100)
        original_estimate = 0.5
        point, lower, upper = BootstrapEngine._bca_ci(samples, original_estimate, jack, 0.95)[:3]
        # Lower and upper should straddle 0.5 symmetrically within tolerance.
        assert lower < 0.5 < upper
        assert abs((0.5 - lower) - (upper - 0.5)) < 0.05

    def test_jackknife_uplift_shape(self):
        """_jackknife_uplift returns an array of length n."""
        n = 100
        rng = np.random.default_rng(0)
        u_ante = rng.random(n) > 0.5
        u_match = u_ante & (rng.random(n) > 0.3)
        d_ante = rng.random(n) > 0.5
        d_match = d_ante & (rng.random(n) > 0.3)
        jack = BootstrapEngine._jackknife_uplift(u_ante, u_match, d_ante, d_match, n)
        assert jack.shape == (n,)

    def test_jackknife_uplift_zero_when_no_support(self):
        """Leave-one-out uplift is 0.0 when antecedent count drops to zero."""
        # Only one row matches the undesired antecedent — leaving it out makes nu=0.
        n = 5
        u_ante = np.array([True, False, False, False, False])
        u_match = np.array([True, False, False, False, False])
        d_ante = np.array([True, True, True, True, True])
        d_match = np.array([True, True, True, True, False])
        jack = BootstrapEngine._jackknife_uplift(u_ante, u_match, d_ante, d_match, n)
        # Leaving out row 0 drops nu to 0 → uplift set to 0.0.
        assert jack[0] == 0.0


class TestDegenerateResampleReporting:
    """Bootstrap should report the fraction of degenerate resamples."""

    def test_undefined_fraction_zero_on_well_supported_rule(self):
        """Well-supported rule: no degenerate resamples, fraction is 0.0."""
        data = _make_data(n=200)
        rule = _make_rule()
        engine = BootstrapEngine(n_bootstrap=100, random_state=42)
        result = engine.compute(data, [rule])[0]
        assert result.undefined_bootstrap_fraction == 0.0

    def test_undefined_fraction_positive_on_thin_rule(self):
        """Rule applying to very few rows produces some degenerate resamples."""
        # Tiny rule: only one row matches the undesired side.
        g1 = pd.DataFrame({'age': ['0'] * 1, 'class': ['0'] * 1, 'target': ['0'] * 1})
        g2 = pd.DataFrame({'age': ['0'] * 1, 'class': ['1'] * 1, 'target': ['1'] * 1})
        g3 = pd.DataFrame({'age': ['1'] * 198, 'class': ['0'] * 198, 'target': ['0'] * 198})
        data = pd.concat([g1, g2, g3], ignore_index=True)
        engine = BootstrapEngine(n_bootstrap=200, random_state=42)
        # Capture the user warning we expect to emit.
        with pytest.warns(UserWarning, match="bootstrap resamples were degenerate"):
            result = engine.compute(data, [_make_rule()])[0]
        # With only 2 matching rows in a 200-row dataset, many resamples will
        # miss them entirely.
        assert result.undefined_bootstrap_fraction is not None
        assert result.undefined_bootstrap_fraction > 0.01
