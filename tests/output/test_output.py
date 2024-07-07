#!/usr/bin/env python
"""Tests for `action_rules` package."""

import json

import pytest

from action_rules.output import Output


@pytest.fixture
def output_instance():
    """
    Fixture to initialize an Output object with preset action rules.

    Returns
    -------
    Output
        An instance of the Output class.
    """
    action_rules = [
        {
            'undesired': {
                'itemset': ['attr1_<item_stable>_val1', 'attr2_<item_flexible>_undesired_val2'],
                'support': 10,
                'confidence': 0.8,
                'target': 'target_<item_target>_undesired',
            },
            'desired': {
                'itemset': ['attr1_<item_stable>_val1', 'attr2_<item_flexible>_desired_val2'],
                'support': 15,
                'confidence': 0.9,
                'target': 'target_<item_target>_desired',
            },
            'uplift': 1.5,
        }
    ]
    return Output(action_rules, 'target')


def test_init(output_instance):
    """
    Test the initialization of the Output class.

    Parameters
    ----------
    output_instance : Output
        The Output instance to test.

    Asserts
    -------
    Asserts that the initialization parameters are correctly set.
    """
    assert output_instance.action_rules == [
        {
            'undesired': {
                'itemset': ['attr1_<item_stable>_val1', 'attr2_<item_flexible>_undesired_val2'],
                'support': 10,
                'confidence': 0.8,
                'target': 'target_<item_target>_undesired',
            },
            'desired': {
                'itemset': ['attr1_<item_stable>_val1', 'attr2_<item_flexible>_desired_val2'],
                'support': 15,
                'confidence': 0.9,
                'target': 'target_<item_target>_desired',
            },
            'uplift': 1.5,
        }
    ]
    assert output_instance.target == 'target'


def test_get_ar_notation(output_instance):
    """
    Test the get_ar_notation method.

    Parameters
    ----------
    output_instance : Output
        The Output instance to test.

    Asserts
    -------
    Asserts that the action rules are correctly represented in a human-readable format.
    """
    ar_notation = output_instance.get_ar_notation()
    assert ar_notation == [
        {
            'undesired': {
                'itemset': ['attr1_<item_stable>_val1', 'attr2_<item_flexible>_undesired_val2'],
                'support': 10,
                'confidence': 0.8,
                'target': 'target_<item_target>_undesired',
            },
            'desired': {
                'itemset': ['attr1_<item_stable>_val1', 'attr2_<item_flexible>_desired_val2'],
                'support': 15,
                'confidence': 0.9,
                'target': 'target_<item_target>_desired',
            },
            'uplift': 1.5,
        }
    ]


def test_get_export_notation(output_instance):
    """
    Test the get_export_notation method.

    Parameters
    ----------
    output_instance : Output
        The Output instance to test.

    Asserts
    -------
    Asserts that the action rules are correctly represented for export.
    """
    export_notation = output_instance.get_export_notation()
    expected = json.dumps(
        [
            {
                'stable': [{'attribute': 'attr1', 'value': 'val1'}],
                'flexible': [{'attribute': 'attr2', 'undesired': 'undesired_val2', 'desired': 'desired_val2'}],
                'target': {'attribute': 'target', 'undesired': 'undesired', 'desired': 'desired'},
                'support of undesired part': 10,
                'confidence of undesired part': 0.8,
                'support of desired part': 15,
                'confidence of desired part': 0.9,
                'uplift': 1.5,
            }
        ]
    )
    assert export_notation == expected


def test_get_pretty_ar_notation(output_instance):
    """
    Test the get_pretty_ar_notation method.

    Parameters
    ----------
    output_instance : Output
        The Output instance to test.

    Asserts
    -------
    Asserts that the action rules are correctly represented as text strings.
    """
    pretty_ar_notation = output_instance.get_pretty_ar_notation()
    expected = [
        "If attribute 'attr1' is 'val1', attribute 'attr2' value 'undesired_val2' is changed to 'desired_val2', then"
        + " 'target' value 'target_<item_target>_undesired' is changed to 'target_<item_target>_desired with uplift:"
        + " 1.5."
    ]
    assert pretty_ar_notation == expected
