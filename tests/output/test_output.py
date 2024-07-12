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
