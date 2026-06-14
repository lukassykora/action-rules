"""Tests for ``ActionRules.cross_validate``.

The public method wraps :class:`action_rules.evaluation.cv.CrossValidator`
and uses the instance's own hyperparameters and utility tables.
"""

import numpy as np
import pandas as pd

from action_rules import ActionRules
from action_rules.evaluation.cv import CrossValidationResult


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


class TestActionRulesCrossValidate:
    """Tests for the ``ActionRules.cross_validate`` public method."""

    def test_returns_cross_validation_result(self):
        """Check that ``cross_validate`` returns a populated ``CrossValidationResult``."""
        ar = ActionRules(
            min_stable_attributes=1,
            min_flexible_attributes=1,
            min_undesired_support=10,
            min_undesired_confidence=0.5,
            min_desired_support=10,
            min_desired_confidence=0.5,
        )
        result = ar.cross_validate(
            _make_dataset(),
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
        assert isinstance(result, CrossValidationResult)
        assert result.n_splits == 3
        assert not result.strategy_summary.empty

    def test_utility_tables_propagated(self):
        """Check that the instance's intrinsic utility table reaches the CV results."""
        intrinsic = {('Survived', '0'): -100.0, ('Survived', '1'): 250.0}
        ar = ActionRules(
            min_stable_attributes=1,
            min_flexible_attributes=1,
            min_undesired_support=10,
            min_undesired_confidence=0.5,
            min_desired_support=10,
            min_desired_confidence=0.5,
            intrinsic_utility_table=intrinsic,
        )
        result = ar.cross_validate(
            _make_dataset(),
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
        assert result.has_utility
        assert 'gain' in set(result.strategy_summary['metric_target'])

    def test_does_not_fit_self(self):
        """``cross_validate`` must not mutate the calling instance's output."""
        ar = ActionRules(
            min_stable_attributes=1,
            min_flexible_attributes=1,
            min_undesired_support=10,
            min_undesired_confidence=0.5,
            min_desired_support=10,
            min_desired_confidence=0.5,
        )
        ar.cross_validate(
            _make_dataset(),
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
        assert ar.output is None  # main instance was never fitted
