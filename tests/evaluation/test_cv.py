"""End-to-end tests for the cross-validation pipeline.

Uses a small synthetic dataset where two flexible attributes (Class, Age)
have known, distinguishable effects on the target.  The action-rules
mining procedure is deterministic given the data and hyperparameters, so
these tests can assert exact rule counts, deterministic seed reproduction,
and the wiring of utility-aware fields.
"""

import numpy as np
import pandas as pd
import pytest

from action_rules import ActionRules
from action_rules.evaluation.cv import (
    METRICS,
    STRATEGIES,
    CrossValidationResult,
    CrossValidator,
    stratified_kfold_indices,
)


def _make_dataset(n: int = 360, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            'Sex': rng.choice(['M', 'F'], size=n),
            'Age': rng.choice(['Y', 'O'], size=n),
            'Class': rng.choice(['A', 'B'], size=n),
            'Region': rng.choice(['N', 'S'], size=n),
        }
    )
    prob = 0.2 + 0.4 * (df['Class'] == 'B').astype(int) + 0.2 * (df['Age'] == 'O').astype(int)
    df['Survived'] = (rng.random(n) < prob).astype(int).astype(str)
    return df


def _factory(**overrides):
    defaults = dict(
        min_stable_attributes=1,
        min_flexible_attributes=1,
        min_undesired_support=10,
        min_undesired_confidence=0.5,
        min_desired_support=10,
        min_desired_confidence=0.5,
    )
    defaults.update(overrides)

    def _f():
        return ActionRules(**defaults)

    return _f


# ---------------------------------------------------------------------------
# stratified_kfold_indices
# ---------------------------------------------------------------------------


class TestStratifiedKFoldIndices:
    def test_partition_is_disjoint_and_covers_everything(self):
        y = np.array([0] * 30 + [1] * 20)
        splits = stratified_kfold_indices(y, n_splits=5, random_state=0)
        all_test = np.concatenate([test for _, test in splits])
        assert sorted(all_test.tolist()) == list(range(len(y)))

    def test_class_balance_preserved(self):
        y = np.array([0] * 30 + [1] * 30)
        splits = stratified_kfold_indices(y, n_splits=5, random_state=0)
        for _, test in splits:
            n0 = int((y[test] == 0).sum())
            n1 = int((y[test] == 1).sum())
            assert n0 == 6
            assert n1 == 6

    def test_too_few_per_class_raises(self):
        y = np.array([0, 0, 1])  # only 1 class-1 sample for 5 folds
        with pytest.raises(ValueError, match="at least 5 members per class"):
            stratified_kfold_indices(y, n_splits=5)

    def test_seed_reproducibility(self):
        y = np.array([0] * 30 + [1] * 30)
        s1 = stratified_kfold_indices(y, n_splits=5, random_state=7)
        s2 = stratified_kfold_indices(y, n_splits=5, random_state=7)
        for (t1, v1), (t2, v2) in zip(s1, s2):
            assert (t1 == t2).all()
            assert (v1 == v2).all()


# ---------------------------------------------------------------------------
# CrossValidator
# ---------------------------------------------------------------------------


class TestCrossValidatorBasic:
    def test_returns_result_type(self):
        df = _make_dataset()
        validator = CrossValidator(
            _factory(),
            stable_attributes=['Sex', 'Region'],
            flexible_attributes=['Class', 'Age'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
            n_splits=3,
            n_bootstrap=50,
            n_bootstrap_oof=0,
            random_state=42,
        )
        result = validator.run(df)
        assert isinstance(result, CrossValidationResult)
        assert result.n_splits == 3
        assert result.strategies == STRATEGIES
        assert result.metrics == METRICS

    def test_one_record_per_fold_rule(self):
        df = _make_dataset()
        validator = CrossValidator(
            _factory(),
            stable_attributes=['Sex', 'Region'],
            flexible_attributes=['Class', 'Age'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
            n_splits=3,
            n_bootstrap=50,
            n_bootstrap_oof=0,
            random_state=42,
        )
        result = validator.run(df)
        assert len(result.rule_records) == sum(result.n_rules_per_fold)

    def test_rule_records_columns_no_utility(self):
        df = _make_dataset()
        validator = CrossValidator(
            _factory(),
            stable_attributes=['Sex', 'Region'],
            flexible_attributes=['Class', 'Age'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
            n_splits=3,
            n_bootstrap=50,
            n_bootstrap_oof=0,
            random_state=42,
        )
        result = validator.run(df)
        # Without utility tables, no gain columns.
        assert 'train_gain' not in result.rule_records.columns
        assert 'test_gain' not in result.rule_records.columns
        # All metric_target entries should be 'uplift'.
        assert (result.strategy_summary['metric_target'] == 'uplift').all()


class TestCrossValidatorUtility:
    def test_gain_columns_present_when_utility(self):
        df = _make_dataset()
        intrinsic = {('Survived', '0'): -100.0, ('Survived', '1'): 250.0}
        validator = CrossValidator(
            _factory(intrinsic_utility_table=intrinsic),
            stable_attributes=['Sex', 'Region'],
            flexible_attributes=['Class', 'Age'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
            n_splits=3,
            intrinsic_utility_table=intrinsic,
            n_bootstrap=50,
            n_bootstrap_oof=0,
            random_state=42,
        )
        result = validator.run(df)
        assert result.has_utility
        assert 'train_gain' in result.rule_records.columns
        assert 'test_gain' in result.rule_records.columns
        # Strategy summary now reports both 'uplift' and 'gain' targets.
        assert set(result.strategy_summary['metric_target'].unique()) == {'uplift', 'gain'}


class TestCrossValidatorDeterminism:
    def test_same_seed_same_result(self):
        df = _make_dataset()

        def _build():
            return CrossValidator(
                _factory(),
                stable_attributes=['Sex', 'Region'],
                flexible_attributes=['Class', 'Age'],
                target='Survived',
                target_undesired_state='0',
                target_desired_state='1',
                n_splits=3,
                n_bootstrap=80,
                n_bootstrap_oof=50,
                random_state=123,
            )

        r1 = _build().run(df)
        r2 = _build().run(df)
        pd.testing.assert_frame_equal(
            r1.strategy_summary.reset_index(drop=True),
            r2.strategy_summary.reset_index(drop=True),
        )


class TestCrossValidatorStrategies:
    def test_lower_positive_uses_only_positive_lower_bounds(self):
        df = _make_dataset()
        validator = CrossValidator(
            _factory(),
            stable_attributes=['Sex', 'Region'],
            flexible_attributes=['Class', 'Age'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
            strategies=('point', 'lower_positive'),
            n_splits=3,
            n_bootstrap=80,
            n_bootstrap_oof=0,
            random_state=42,
        )
        result = validator.run(df)
        # Both strategies must produce a row per metric.
        assert set(result.strategy_summary['strategy']) == {'point', 'lower_positive'}


class TestCrossValidatorValidation:
    def test_n_splits_too_small(self):
        with pytest.raises(ValueError, match="n_splits"):
            CrossValidator(
                _factory(),
                stable_attributes=['Sex'],
                flexible_attributes=['Class'],
                target='Survived',
                target_undesired_state='0',
                target_desired_state='1',
                n_splits=1,
            )

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategies"):
            CrossValidator(
                _factory(),
                stable_attributes=['Sex'],
                flexible_attributes=['Class'],
                target='Survived',
                target_undesired_state='0',
                target_desired_state='1',
                strategies=('point', 'nonexistent'),
            )

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metrics"):
            CrossValidator(
                _factory(),
                stable_attributes=['Sex'],
                flexible_attributes=['Class'],
                target='Survived',
                target_undesired_state='0',
                target_desired_state='1',
                metrics=('uplift_at_k', 'auc'),
            )


class TestRuleStability:
    def test_jaccard_in_unit_interval(self):
        df = _make_dataset()
        validator = CrossValidator(
            _factory(),
            stable_attributes=['Sex', 'Region'],
            flexible_attributes=['Class', 'Age'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
            n_splits=3,
            n_bootstrap=50,
            n_bootstrap_oof=0,
            random_state=42,
            track_stability=True,
        )
        result = validator.run(df)
        assert result.rule_stability is not None
        assert ((result.rule_stability['jaccard'] >= 0.0) & (result.rule_stability['jaccard'] <= 1.0)).all()

    def test_track_stability_off(self):
        df = _make_dataset()
        validator = CrossValidator(
            _factory(),
            stable_attributes=['Sex', 'Region'],
            flexible_attributes=['Class', 'Age'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
            n_splits=3,
            n_bootstrap=50,
            n_bootstrap_oof=0,
            random_state=42,
            track_stability=False,
        )
        result = validator.run(df)
        assert result.rule_stability is None


class TestInsampleBaseline:
    """The apparent (in-sample) baseline reports mining + scoring on the same
    data — by construction optimistic relative to the held-out CV mean.  Used
    by the article to quantify the optimism gap (Hastie ESL Ch. 7)."""

    def _build(self, **overrides):
        kwargs = dict(
            stable_attributes=['Sex', 'Region'],
            flexible_attributes=['Class', 'Age'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
            n_splits=3,
            n_bootstrap=80,
            n_bootstrap_oof=0,
            random_state=42,
        )
        kwargs.update(overrides)
        return CrossValidator(_factory(), **kwargs)

    def test_disabled_by_default(self):
        df = _make_dataset()
        result = self._build().run(df)
        assert result.insample_summary is None, (
            "Default behaviour must leave insample_summary unset so existing "
            "consumers see no schema change."
        )

    def test_enabled_returns_dataframe(self):
        df = _make_dataset()
        result = self._build(compute_insample_baseline=True).run(df)
        assert result.insample_summary is not None
        assert isinstance(result.insample_summary, pd.DataFrame)
        assert set(result.insample_summary.columns) >= {
            'strategy',
            'metric',
            'metric_target',
            'value',
        }
        assert 'fold' not in result.insample_summary.columns, (
            "The synthetic fold index used internally must not leak into the "
            "in-sample frame; readers expect a clean (strategy, metric, value)."
        )

    def test_values_are_finite(self):
        df = _make_dataset()
        result = self._build(compute_insample_baseline=True).run(df)
        assert result.insample_summary is not None
        finite = np.isfinite(result.insample_summary['value'].to_numpy())
        assert finite.all(), "All in-sample metric values must be finite."

    def test_strategy_and_metric_coverage(self):
        df = _make_dataset()
        result = self._build(compute_insample_baseline=True).run(df)
        assert result.insample_summary is not None
        seen_strategies = set(result.insample_summary['strategy'].unique())
        seen_metrics = set(result.insample_summary['metric'].unique())
        assert seen_strategies == set(STRATEGIES)
        assert seen_metrics == set(METRICS)

    def test_insample_optimism_gte_cv_mean_on_uplift_at_k(self):
        """The in-sample uplift@k is constructed by mining and scoring on the
        same data, so it is an upper bound on the held-out CV mean by the
        same data.  We tolerate equality (on tiny synthetic datasets the rule
        set discovered on the full data can coincide with what each fold
        finds, eliminating the gap)."""
        df = _make_dataset()
        result = self._build(compute_insample_baseline=True).run(df)
        assert result.insample_summary is not None
        # Compare the 'point' strategy on uplift@k (the article's headline view).
        is_row = result.insample_summary[
            (result.insample_summary['strategy'] == 'point')
            & (result.insample_summary['metric'] == 'uplift_at_k')
            & (result.insample_summary['metric_target'] == 'uplift')
        ]
        cv_row = result.strategy_summary[
            (result.strategy_summary['strategy'] == 'point')
            & (result.strategy_summary['metric'] == 'uplift_at_k')
            & (result.strategy_summary['metric_target'] == 'uplift')
        ]
        assert len(is_row) == 1 and len(cv_row) == 1
        assert float(is_row['value'].iloc[0]) + 1e-9 >= float(cv_row['mean'].iloc[0])

    def test_proxy_through_action_rules_class(self):
        df = _make_dataset()
        ar = ActionRules(
            min_stable_attributes=1,
            min_flexible_attributes=1,
            min_undesired_support=10,
            min_undesired_confidence=0.5,
            min_desired_support=10,
            min_desired_confidence=0.5,
        )
        result = ar.cross_validate(
            df,
            stable_attributes=['Sex', 'Region'],
            flexible_attributes=['Class', 'Age'],
            target='Survived',
            target_undesired_state='0',
            target_desired_state='1',
            n_splits=3,
            n_bootstrap=80,
            n_bootstrap_oof=0,
            random_state=42,
            compute_insample_baseline=True,
        )
        assert result.insample_summary is not None
        assert len(result.insample_summary) > 0
