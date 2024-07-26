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


def test_get_split_tables(action_rules):
    """
    Test the get_split_tables method.

    Parameters
    ----------
    action_rules : ActionRules
        The ActionRules instance to test.

    Asserts
    -------
    Asserts that the dataset is correctly split into tables based on target item bindings.
    """
    data = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1]])
    target_items_binding = {'target': [2, 3]}
    target = 'target'
    split_tables = action_rules.get_split_tables(data, target_items_binding, target)
    np.testing.assert_array_equal(split_tables[2], data[:, [1]])
    np.testing.assert_array_equal(split_tables[3], data[:, [0, 2]])


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
