#!/usr/bin/env python
"""Tests for `action_rules` package."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from action_rules.action_rules import ActionRules
from action_rules.output import Output


@pytest.fixture
def action_rules():
    """
    Fixture to initialize an ActionRules object with preset parameters.

    Returns
    -------
    ActionRules
        An instance of the ActionRules class.
    """
    return ActionRules(
        min_stable_attributes=1,
        min_flexible_attributes=1,
        min_undesired_support=1,
        min_undesired_confidence=0.5,
        min_desired_support=1,
        min_desired_confidence=0.5,
        verbose=False,
    )


def test_init(action_rules):
    """
    Test the initialization of the ActionRules class.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the initialization parameters are correctly set.
    """
    assert action_rules.min_stable_attributes == 1
    assert action_rules.min_flexible_attributes == 1
    assert action_rules.min_undesired_support == 1
    assert action_rules.min_undesired_confidence == 0.5
    assert action_rules.min_desired_support == 1
    assert action_rules.min_desired_confidence == 0.5
    assert not action_rules.verbose


def test_count_max_nodes(action_rules):
    """
    Test the count_max_nodes method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the count_max_nodes method calculates the correct number of nodes.
    """
    stable_items_binding = {'attr1': [1, 2, 3]}
    flexible_items_binding = {'attr2': [4, 5]}
    result = action_rules.count_max_nodes(stable_items_binding, flexible_items_binding)
    assert result == 11


def test_set_array_library(action_rules):
    """
    Test the set_array_library method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the array library is correctly set based on the use_gpu flag.
    """
    # Test with GPU - it can not be done because the GPU library is optional

    # Test without GPU
    action_rules.set_array_library(use_gpu=False, df=pd.DataFrame())
    assert not action_rules.is_gpu_np
    assert not action_rules.is_gpu_pd


def test_df_to_array(action_rules):
    """
    Test the df_to_array method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the DataFrame is correctly converted to a NumPy array.
    """
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    action_rules.set_array_library(use_gpu=False, df=df)
    data, columns = action_rules.df_to_array(df)
    np.testing.assert_array_equal(data, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8))
    assert columns == ['A', 'B']


def test_one_hot_encode(action_rules):
    """
    Test the one_hot_encode method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the one-hot encoding is correctly applied to the specified attributes.
    """
    df = pd.DataFrame({'stable': ['a', 'b', 'a'], 'flexible': ['x', 'y', 'z'], 'target': ['yes', 'no', 'yes']})
    action_rules.set_array_library(use_gpu=False, df=df)
    encoded_df = action_rules.one_hot_encode(df, ['stable'], ['flexible'], 'target')
    expected_columns = [
        'stable_<item_stable>_a',
        'stable_<item_stable>_b',
        'flexible_<item_flexible>_x',
        'flexible_<item_flexible>_y',
        'flexible_<item_flexible>_z',
        'target_<item_target>_yes',
        'target_<item_target>_no',
    ]
    assert set(encoded_df.columns) == set(expected_columns)


def test_one_hot_encode_empty_stable(action_rules):
    """
    Test the one_hot_encode method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the one-hot encoding is correctly applied to the specified attributes.
    """
    df = pd.DataFrame({'stable': ['a', 'b', 'a'], 'flexible': ['x', 'y', 'z'], 'target': ['yes', 'no', 'yes']})
    action_rules.set_array_library(use_gpu=False, df=df)
    encoded_df = action_rules.one_hot_encode(df, [], [], 'target')
    expected_columns = [
        'target_<item_target>_yes',
        'target_<item_target>_no',
    ]
    assert set(encoded_df.columns) == set(expected_columns)


def test_one_hot_encode_excludes_missing_stable(action_rules):
    """NaN in a stable column must not produce a ``stable_<item_stable>_nan`` one-hot column.

    Documents the pessimistic interpretation of null values for antecedents (Dardzinska 2013,
    Section 2.3.2) — a missing stable attribute does not match any value-specific itemset.
    """
    df = pd.DataFrame(
        {
            'stable': ['a', np.nan],
            'flexible': ['x', 'y'],
            'target': ['yes', 'no'],
        }
    )
    action_rules.set_array_library(use_gpu=False, df=df)
    encoded_df = action_rules.one_hot_encode(df, ['stable'], ['flexible'], 'target')
    assert 'stable_<item_stable>_a' in encoded_df.columns
    assert 'stable_<item_stable>_nan' not in encoded_df.columns


def test_one_hot_encode_excludes_missing_flexible(action_rules):
    """NaN in a flexible column must not produce a ``flexible_<item_flexible>_nan`` one-hot column."""
    df = pd.DataFrame(
        {
            'stable': ['a', 'b'],
            'flexible': ['x', np.nan],
            'target': ['yes', 'no'],
        }
    )
    action_rules.set_array_library(use_gpu=False, df=df)
    encoded_df = action_rules.one_hot_encode(df, ['stable'], ['flexible'], 'target')
    assert 'flexible_<item_flexible>_x' in encoded_df.columns
    assert 'flexible_<item_flexible>_nan' not in encoded_df.columns


def test_one_hot_encode_keeps_target_missing_as_category(action_rules):
    """NaN in the target column is preserved as its own explicit category.

    Asymmetric to antecedent handling on purpose: downstream ``get_split_tables`` will then
    cleanly exclude unlabelled rows from both the undesired and desired splits rather than
    silently misassigning them.
    """
    df = pd.DataFrame(
        {
            'stable': ['a', 'b'],
            'flexible': ['x', 'y'],
            'target': ['yes', np.nan],
        }
    )
    action_rules.set_array_library(use_gpu=False, df=df)
    encoded_df = action_rules.one_hot_encode(df, ['stable'], ['flexible'], 'target')
    assert 'target_<item_target>_yes' in encoded_df.columns
    assert 'target_<item_target>_nan' in encoded_df.columns


def test_get_bindings(action_rules):
    """
    Test the get_bindings method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that attributes are correctly bound to their respective columns.
    """
    columns = [
        'stable_<item_stable>_a',
        'stable_<item_stable>_b',
        'flexible_<item_flexible>_x',
        'flexible_<item_flexible>_y',
        'flexible_<item_flexible>_z',
        'target_<item_target>_yes',
        'target_<item_target>_no',
    ]
    stable_attributes = ['stable']
    flexible_attributes = ['flexible']
    target = 'target'
    stable_items_binding, flexible_items_binding, target_items_binding, column_values = action_rules.get_bindings(
        columns, stable_attributes, flexible_attributes, target
    )
    assert stable_items_binding == {'stable': [0, 1]}
    assert flexible_items_binding == {'flexible': [2, 3, 4]}
    assert target_items_binding == {'target': [5, 6]}
    assert column_values == {
        0: ('stable', 'a'),
        1: ('stable', 'b'),
        2: ('flexible', 'x'),
        3: ('flexible', 'y'),
        4: ('flexible', 'z'),
        5: ('target', 'yes'),
        6: ('target', 'no'),
    }


def test_get_stop_list(action_rules):
    """
    Test the get_stop_list method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the stop list is correctly generated.
    """
    stable_items_binding = {'attr1': [1, 2]}
    flexible_items_binding = {'attr2': [3]}
    stop_list = action_rules.get_stop_list(stable_items_binding, flexible_items_binding)
    expected_stop_list = [(1, 1), (1, 2), (2, 1), (2, 2), ('attr2', 'attr2')]
    assert stop_list == expected_stop_list




@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
        True,
    ],
)
def test_fit(action_rules, use_gpu):
    """
    Test the fit method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.
    use_gpu : bool
        Use sparse array.

    Asserts
    -------
    Asserts that the full workflow of generating action rules works correctly.
    """
    df = pd.DataFrame({'stable': ['a', 'b', 'a'], 'flexible': ['x', 'y', 'z'], 'target': ['yes', 'no', 'no']})
    action_rules.fit(
        df,
        stable_attributes=['stable'],
        flexible_attributes=['flexible'],
        target='target',
        target_undesired_state='no',
        target_desired_state='yes',
        use_gpu=use_gpu,
    )
    rules = action_rules.get_rules()
    assert rules is not None
    assert len(rules.action_rules) == 1
    assert isinstance(rules, Output)


def test_fit_raises_error_when_already_fit(action_rules):
    """
    Test the fit method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the initialized model can not be fit again.
    """
    df = pd.DataFrame({'stable': ['a', 'b', 'a'], 'flexible': ['x', 'y', 'z'], 'target': ['yes', 'no', 'yes']})
    action_rules.fit(
        df,
        stable_attributes=['stable'],
        flexible_attributes=['flexible'],
        target='target',
        target_undesired_state='no',
        target_desired_state='yes',
    )
    with pytest.raises(RuntimeError, match="The model is already fit."):
        action_rules.fit(
            df,
            stable_attributes=['stable'],
            flexible_attributes=['flexible'],
            target='target',
            target_undesired_state='no',
            target_desired_state='yes',
        )


def test_fit_onehot(action_rules):
    """
    Test the fit_onehot method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the fit_onehot method processes the data correctly and fits the model.
    """
    df = pd.DataFrame(
        {
            'young': [0, 1, 0, 0],
            'old': [1, 0, 1, 1],
            'high': [1, 1, 0, 0],
            'low': [0, 0, 1, 1],
            'animals': [1, 1, 1, 0],
            'toys': [0, 0, 1, 1],
            'no': [0, 0, 1, 1],
            'yes': [1, 1, 0, 0],
        }
    )

    stable_attributes = {'age': ['young', 'old']}
    flexible_attributes = {'income': ['high', 'low'], 'hobby': ['animals', 'toys']}
    target = {'target': ['yes', 'no']}

    action_rules.fit_onehot(
        data=df,
        stable_attributes=stable_attributes,
        flexible_attributes=flexible_attributes,
        target=target,
        target_undesired_state='no',
        target_desired_state='yes',
        use_sparse_matrix=False,
        use_gpu=False,
    )

    # Check that the model has been fitted
    assert action_rules.output is not None
    assert isinstance(action_rules.output, Output)

    # Check if the columns were renamed correctly and irrelevant columns removed
    expected_columns = [
        'age_<item_stable>_young',
        'age_<item_stable>_old',
        'income_<item_flexible>_high',
        'income_<item_flexible>_low',
        'hobby_<item_flexible>_animals',
        'hobby_<item_flexible>_toys',
        'target_<item_target>_yes',
        'target_<item_target>_no',
    ]
    assert set(action_rules.rules.columns) == set(expected_columns)

    # Check if the correct attributes were passed to the fit method
    assert action_rules.rules is not None
    assert len(action_rules.rules.action_rules) > 0  # Rules should have been generated


def test_get_rules(action_rules):
    """
    Test the get_rules method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the generated rules are correctly returned.
    """
    with pytest.raises(RuntimeError, match="The model is not fit."):
        assert action_rules.get_rules() is None
    action_rules.output = MagicMock()
    assert action_rules.get_rules() is not None
    assert action_rules.get_rules() == action_rules.output


def test_predict(action_rules):
    """
    Test the predict method of the ActionRules class.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the prediction works correctly and returns the expected DataFrame.
    """
    frame_row = pd.Series({'stable': 'a', 'flexible': 'z'})
    with pytest.raises(RuntimeError, match="The model is not fit."):
        action_rules.predict(frame_row)
    df = pd.DataFrame({'stable': ['a', 'b', 'a'], 'flexible': ['x', 'y', 'z'], 'target': ['yes', 'no', 'no']})
    action_rules.fit(
        df,
        stable_attributes=['stable'],
        flexible_attributes=['flexible'],
        target='target',
        target_undesired_state='no',
        target_desired_state='yes',
    )
    result = action_rules.predict(frame_row)
    assert not result.empty
    assert 'flexible (Recommended)' in result.columns
    assert 'ActionRules_RuleIndex' in result.columns
    assert 'ActionRules_UndesiredSupport' in result.columns
    assert 'ActionRules_DesiredSupport' in result.columns
    assert 'ActionRules_UndesiredConfidence' in result.columns
    assert 'ActionRules_DesiredConfidence' in result.columns
    assert 'ActionRules_Uplift' in result.columns

    assert result.iloc[0]['flexible (Recommended)'] == 'x'
    assert result.iloc[0]['ActionRules_RuleIndex'] == 0
    assert result.iloc[0]['ActionRules_UndesiredSupport'] == 1
    assert result.iloc[0]['ActionRules_DesiredSupport'] == 1
    assert result.iloc[0]['ActionRules_UndesiredConfidence'] == 1.0
    assert result.iloc[0]['ActionRules_DesiredConfidence'] == 1.0
    assert result.iloc[0]['ActionRules_Uplift'] == 1 / 3  # one is changed, 3 transactions


def test_remap_utility_tables(action_rules):
    """
    Test the remap_utility_tables method.

    The intrinsic utility table keys are tuples in the format (Attribute, Value), and the transition
    utility table keys are tuples in the format (Attribute, from_value, to_value). Given a column_values
    mapping that maps internal column indices to (Attribute, value) pairs, this test verifies that the utility
    tables are remapped to use the corresponding column indices.

    For example, given:
      intrinsic_table = {
          ('Salary', 'Low'): -300.0,
          ('Salary', 'Medium'): -500.0,
          ('Salary', 'High'): -1000.0,
          ('Attrition', 'False'): 700.0,
          ('Attrition', 'True'): 0.0,
      }
      transition_table = {
          ('Salary', 'Low', 'Medium'): -1.5,
          ('Salary', 'Low', 'High'): -3.5,
          ('Salary', 'Medium', 'High'): -1.3,
      }
      column_values = {
          0: ('Salary', 'low'),
          1: ('Salary', 'medium'),
          2: ('Salary', 'high'),
          3: ('Attrition', 'false'),
          4: ('Attrition', 'true'),
      }
    The expected remapped utility tables are:
      expected_intrinsic = {0: -300.0, 1: -500.0, 2: -1000.0, 3: 700.0, 4: 0.0}
      expected_transition = {(0, 1): -1.5, (0, 2): -3.5, (1, 2): -1.3}
    """
    intrinsic_table = {
        ('Salary', 'Low'): -300.0,
        ('Salary', 'Medium'): -500.0,
        ('Salary', 'High'): -1000.0,
        ('Attrition', 'False'): 700.0,
        ('Attrition', 'True'): 0.0,
    }
    transition_table = {
        ('Salary', 'Low', 'Medium'): -1.5,
        ('Salary', 'Low', 'High'): -3.5,
        ('Salary', 'Medium', 'High'): -1.3,
    }
    column_values = {
        0: ('Salary', 'low'),
        1: ('Salary', 'medium'),
        2: ('Salary', 'high'),
        3: ('Attrition', 'false'),
        4: ('Attrition', 'true'),
    }
    # Overwrite the instance's utility tables with the new tables.
    action_rules.intrinsic_utility_table = intrinsic_table
    action_rules.transition_utility_table = transition_table

    remapped_intrinsic, remapped_transition = action_rules.remap_utility_tables(column_values)

    expected_intrinsic = {
        0: -300.0,
        1: -500.0,
        2: -1000.0,
        3: 700.0,
        4: 0.0,
    }
    expected_transition = {
        (0, 1): -1.5,
        (0, 2): -3.5,
        (1, 2): -1.3,
    }
    assert remapped_intrinsic == expected_intrinsic
    assert remapped_transition == expected_transition


def test_build_bit_masks_single_word(action_rules):
    """
    Verify that a small binary matrix is packed into a single 64-bit word per attribute.
    """
    action_rules.set_array_library(use_gpu=False, df=pd.DataFrame({'dummy': [0]}))
    data = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    bit_masks = action_rules.build_bit_masks(data)

    assert bit_masks.shape == (2, 1)
    assert bit_masks.dtype == np.uint64
    assert bit_masks[0, 0] == np.uint64(0b00000000000000000000000000000101)
    assert bit_masks[1, 0] == np.uint64(0b00000000000000000000000000000010)


def test_build_bit_masks_multiple_words(action_rules):
    """
    Verify that packing spans multiple 64-bit words when transactions exceed 64 entries.
    """
    action_rules.set_array_library(use_gpu=False, df=pd.DataFrame({'dummy': [0]}))
    data = np.zeros((2, 130), dtype=np.uint8)
    # attribute #0 hits several boundary positions
    data[0, 0] = 1
    data[0, 63] = 1
    data[0, 64] = 1
    data[0, 129] = 1
    # attribute #1 lights up different offsets
    data[1, 1] = 1
    data[1, 62] = 1
    data[1, 65] = 1
    data[1, 100] = 1
    data[1, 128] = 1

    bit_masks = action_rules.build_bit_masks(data)

    assert bit_masks.shape == (2, 3)

    attr0_word0 = (np.uint64(1) << np.uint64(0)) | (np.uint64(1) << np.uint64(63))
    attr0_word1 = np.uint64(1) << np.uint64(0)
    attr0_word2 = np.uint64(1) << np.uint64(1)
    assert bit_masks[0, 0] == attr0_word0
    assert bit_masks[0, 1] == attr0_word1
    assert bit_masks[0, 2] == attr0_word2

    attr1_word0 = (np.uint64(1) << np.uint64(1)) | (np.uint64(1) << np.uint64(62))
    attr1_word1 = (np.uint64(1) << np.uint64(1)) | (np.uint64(1) << np.uint64(36))
    attr1_word2 = np.uint64(1) << np.uint64(0)
    assert bit_masks[1, 0] == attr1_word0
    assert bit_masks[1, 1] == attr1_word1
    assert bit_masks[1, 2] == attr1_word2


def test_fit_uses_bfs_candidate_expansion(action_rules, monkeypatch):
    """
    Candidate expansion should follow queue order so earlier siblings are processed first.
    """
    visited_prefixes = []

    class DummyRules:
        def __init__(self, *args, **kwargs):
            self.action_rules = []

        def prune_classification_rules(self, depth, stop_list):
            return None

        def generate_action_rules(self):
            return None

    class DummyCandidateGenerator:
        def __init__(self, **kwargs):
            return None

        def generate_candidates(self, **candidate):
            child_candidate = {
                key: value
                for key, value in candidate.items()
                if key
                in {
                    'stable_items_binding',
                    'flexible_items_binding',
                    'actionable_attributes',
                }
            }
            prefix = tuple(candidate['ar_prefix'])
            visited_prefixes.append(prefix)
            if prefix == tuple():
                return [
                    {
                        **child_candidate,
                        'ar_prefix': ('a',),
                        'itemset_prefix': ('a',),
                    },
                    {
                        **child_candidate,
                        'ar_prefix': ('b',),
                        'itemset_prefix': ('b',),
                    },
                ]
            if prefix == ('a',):
                return [
                    {
                        **child_candidate,
                        'ar_prefix': ('a', 'a1'),
                        'itemset_prefix': ('a', 'a1'),
                    }
                ]
            return []

    monkeypatch.setattr('action_rules.action_rules.CandidateGenerator', DummyCandidateGenerator)
    monkeypatch.setattr('action_rules.action_rules.Rules', DummyRules)

    df = pd.DataFrame({'stable': ['a', 'b'], 'flexible': ['x', 'y'], 'target': ['yes', 'no']})
    action_rules.fit(
        df,
        stable_attributes=['stable'],
        flexible_attributes=['flexible'],
        target='target',
        target_undesired_state='no',
        target_desired_state='yes',
    )

    assert visited_prefixes == [tuple(), ('a',), ('b',), ('a', 'a1')]
