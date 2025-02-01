#!/usr/bin/env python
"""Tests for `action_rules` package."""

import pytest

from action_rules.output import Output


@pytest.fixture
def sample_action_rules():
    """
    Fixture for sample action rules to be used in tests.

    Returns
    -------
    list
        List containing the action rules.
    """
    return [
        {
            'undesired': {
                'itemset': [0, 2],
                'support': 10,
                'confidence': 0.8,
                'target': 0,
            },
            'desired': {
                'itemset': [1, 2],
                'support': 5,
                'confidence': 0.6,
                'target': 1,
            },
            'uplift': 0.2,
        }
    ]


@pytest.fixture
def column_values():
    """
    Fixture for column values.

    Returns
    -------
    dict
        Dictionary containing column values.
    """
    return {0: ('status', 'default'), 1: ('status', 'paid'), 2: ('age', '30'), 3: ('age', '31')}


@pytest.fixture
def stable_items_binding():
    """
    Fixture for stable items binding.

    Returns
    -------
    dict
        Dictionary containing bindings for stable items.
    """
    return {'age': [2, 3]}


@pytest.fixture
def flexible_items_binding():
    """
    Fixture for flexible items binding.

    Returns
    -------
    dict
        Dictionary containing bindings for flexible items.
    """
    return {'status': [0, 1]}


@pytest.fixture
def output_instance(sample_action_rules, column_values, stable_items_binding, flexible_items_binding):
    """
    Fixture for Output instance.

    Returns
    -------
    Output
        Instance of the Output class.
    """
    return Output(sample_action_rules, 'status', stable_items_binding, flexible_items_binding, column_values)


def test_get_ar_notation(output_instance):
    """
    Test the get_ar_notation method of the Output class.

    Parameters
    ----------
    output_instance : Output
        Instance of the Output class.
    """
    ar_notation = output_instance.get_ar_notation()
    assert len(ar_notation) == 1
    assert 'status: default → paid' in ar_notation[0]
    assert 'support of undesired part: 10' in ar_notation[0]
    assert 'confidence of undesired part: 0.8' in ar_notation[0]


def test_get_export_notation(output_instance):
    """
    Test the get_export_notation method of the Output class.

    Parameters
    ----------
    output_instance : Output
        Instance of the Output class.
    """
    export_notation = output_instance.get_export_notation()
    assert len(export_notation) > 0
    assert '"attribute": "status"' in export_notation
    assert '"undesired": "default"' in export_notation
    assert '"desired": "paid"' in export_notation


def test_get_pretty_ar_notation(output_instance):
    """
    Test the get_pretty_ar_notation method of the Output class.

    Parameters
    ----------
    output_instance : Output
        Instance of the Output class.
    """
    pretty_ar_notation = output_instance.get_pretty_ar_notation()
    assert len(pretty_ar_notation) == 1
    assert "'age' is '30'" in pretty_ar_notation[0]
    assert "then 'status' value 'default' is changed to 'paid" in pretty_ar_notation[0]


@pytest.fixture
def sample_action_rules_dominant_test():
    """
    Provide multiple action rules for testing the 'get_dominant_rules' method.

    This fixture returns a list of dictionaries, each representing an action rule
    with an 'undesired' and 'desired' part, plus an 'uplift' value. These rules
    vary in the size of their 'itemset' and in their 'uplift', which allows
    testing different dominance scenarios (supersets, subsets, etc.).

    Returns
    -------
    list
        A list of action rules. Each rule is a dictionary with the following keys:
        'undesired' : dict
            Contains an 'itemset', 'support', 'confidence', and 'target'.
        'desired' : dict
            Contains an 'itemset', 'support', 'confidence', and 'target'.
        'uplift' : float
            The uplift value for the rule.
    """
    return [
        {
            # rule_index = 0
            'undesired': {
                'itemset': [0],
                'support': 10,
                'confidence': 0.8,
                'target': 0,
            },
            'desired': {
                'itemset': [0],
                'support': 5,
                'confidence': 0.6,
                'target': 1,
            },
            'uplift': 0.2,
        },
        {
            # rule_index = 1 (a superset of rule_index=0, but with lower uplift)
            'undesired': {
                'itemset': [0, 2],
                'support': 8,
                'confidence': 0.7,
                'target': 0,
            },
            'desired': {
                'itemset': [0, 2],
                'support': 4,
                'confidence': 0.5,
                'target': 1,
            },
            'uplift': 0.1,
        },
        {
            # rule_index = 2 (a subset of rule_index=1, but with higher uplift)
            'undesired': {
                'itemset': [2],
                'support': 15,
                'confidence': 0.9,
                'target': 0,
            },
            'desired': {
                'itemset': [2],
                'support': 10,
                'confidence': 0.66,
                'target': 1,
            },
            'uplift': 0.3,
        },
    ]


@pytest.fixture
def output_instance_dominant_test(
    sample_action_rules_dominant_test, column_values, stable_items_binding, flexible_items_binding
):
    """
    Create an Output instance for testing the 'get_dominant_rules' method.

    This fixture initializes an Output object with a list of sample action rules
    (including undesired and desired parts, plus uplift), as well as relevant
    mappings such as stable items, flexible items, and column values.

    Parameters
    ----------
    sample_action_rules_dominant_test : list of dict
        A list of action rules to test dominance behavior.
    column_values : dict
        A dictionary mapping column indices to attribute-value pairs.
    stable_items_binding : dict
        A dictionary specifying stable attribute bindings.
    flexible_items_binding : dict
        A dictionary specifying flexible attribute bindings.

    Returns
    -------
    Output
        An instance of the Output class, configured with the provided rules
        and metadata for testing.
    """
    return Output(
        sample_action_rules_dominant_test, 'status', stable_items_binding, flexible_items_binding, column_values
    )


def test_get_dominant_rules(output_instance_dominant_test):
    """
    Validate that 'get_dominant_rules' correctly identifies and orders dominant rules.

    This test checks whether the rules returned by 'get_dominant_rules' are those
    that remain after applying a Pareto dominance comparison based on the size
    of their 'itemsets' and their 'uplift'. It also verifies that they are sorted
    by 'uplift' in descending order.

    Specifically:
      - A rule that is a superset of another with equal or lower uplift
        should be excluded.
      - A rule that is a subset of another with higher or equal uplift
        should replace the other.
      - The final dominant rules are then sorted by descending uplift.

    Parameters
    ----------
    output_instance_dominant_test : Output
        An instance of the Output class initialized with sample action rules
        tailored to test dominance logic.
    """
    dominant_indices = output_instance_dominant_test.get_dominant_rules()

    # Explanation:
    #   - Rule #1 (index=1) is a superset of #0 but with lower uplift, so it should be excluded.
    #   - Rule #2 (index=2) is a subset of #1 but has higher uplift, so #1 is dominated.
    #   - Rule #0 remains because it's not dominated by #1 (it has a smaller set, but #1's uplift is lower).
    #
    # After sorting by uplift in descending order:
    #   -> index=2 (uplift=0.3) should be first, then index=0 (uplift=0.2).

    assert dominant_indices == [2, 0], f"Expected dominant indices [2, 0], but got {dominant_indices}"
