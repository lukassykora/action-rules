#!/usr/bin/env python
"""Tests for `action_rules` package."""

import json

import pytest

from action_rules.input import Input
from action_rules.output import Output


@pytest.fixture
def input_instance():
    """
    Fixture to initialize an Input object.

    Returns
    -------
    Input
        An instance of the Input class.
    """
    return Input()


def test_import_action_rules(input_instance):
    """
    Test the import_action_rules method.

    Parameters
    ----------
    input_instance : Input
        The Input instance to test.

    Asserts
    -------
    Asserts that the action rules are correctly imported from a JSON string.
    """
    json_data = json.dumps(
        [
            {
                "support of undesired part": 0.6,
                "confidence of undesired part": 0.7,
                "support of desired part": 0.8,
                "confidence of desired part": 0.9,
                "uplift": 1.2,
                "target": {"attribute": "attr", "undesired": "undesired_value", "desired": "desired_value"},
                "stable": [{"attribute": "stable_attr", "value": "stable_value"}],
                "flexible": [{"attribute": "flexible_attr", "undesired": "flex_undesired", "desired": "flex_desired"}],
            }
        ]
    )

    output = input_instance.import_action_rules(json_data)

    assert isinstance(output, Output)
    assert len(output.action_rules) == 1
    rule = output.action_rules[0]

    assert rule['undesired']['support'] == 0.6
    assert rule['undesired']['confidence'] == 0.7
    assert rule['undesired']['target'] == "attr_<item_target>_undesired_value"
    assert rule['desired']['support'] == 0.8
    assert rule['desired']['confidence'] == 0.9
    assert rule['desired']['target'] == "attr_<item_target>_desired_value"
    assert rule['uplift'] == 1.2
    assert rule['undesired']['itemset'] == [
        "stable_attr_<item_stable>_stable_value",
        "flexible_attr_<item_flexible>_flex_undesired",
    ]
    assert rule['desired']['itemset'] == [
        "stable_attr_<item_stable>_stable_value",
        "flexible_attr_<item_flexible>_flex_desired",
    ]


def test_import_action_rules_multiple_rules(input_instance):
    """
    Test the import_action_rules method with multiple rules.

    Parameters
    ----------
    input_instance : Input
        The Input instance to test.

    Asserts
    -------
    Asserts that multiple action rules are correctly imported from a JSON string.
    """
    json_data = json.dumps(
        [
            {
                "support of undesired part": 0.6,
                "confidence of undesired part": 0.7,
                "support of desired part": 0.8,
                "confidence of desired part": 0.9,
                "uplift": 1.2,
                "target": {"attribute": "attr", "undesired": "undesired_value", "desired": "desired_value"},
                "stable": [{"attribute": "stable_attr", "value": "stable_value"}],
                "flexible": [{"attribute": "flexible_attr", "undesired": "flex_undesired", "desired": "flex_desired"}],
            },
            {
                "support of undesired part": 0.5,
                "confidence of undesired part": 0.6,
                "support of desired part": 0.7,
                "confidence of desired part": 0.8,
                "uplift": 1.1,
                "target": {"attribute": "attr2", "undesired": "undesired_value2", "desired": "desired_value2"},
                "stable": [{"attribute": "stable_attr2", "value": "stable_value2"}],
                "flexible": [
                    {"attribute": "flexible_attr2", "undesired": "flex_undesired2", "desired": "flex_desired2"}
                ],
            },
        ]
    )

    output = input_instance.import_action_rules(json_data)

    assert isinstance(output, Output)
    assert len(output.action_rules) == 2

    # Test first rule
    rule1 = output.action_rules[0]
    assert rule1['undesired']['support'] == 0.6
    assert rule1['undesired']['confidence'] == 0.7
    assert rule1['undesired']['target'] == "attr_<item_target>_undesired_value"
    assert rule1['desired']['support'] == 0.8
    assert rule1['desired']['confidence'] == 0.9
    assert rule1['desired']['target'] == "attr_<item_target>_desired_value"
    assert rule1['uplift'] == 1.2
    assert rule1['undesired']['itemset'] == [
        "stable_attr_<item_stable>_stable_value",
        "flexible_attr_<item_flexible>_flex_undesired",
    ]
    assert rule1['desired']['itemset'] == [
        "stable_attr_<item_stable>_stable_value",
        "flexible_attr_<item_flexible>_flex_desired",
    ]

    # Test second rule
    rule2 = output.action_rules[1]
    assert rule2['undesired']['support'] == 0.5
    assert rule2['undesired']['confidence'] == 0.6
    assert rule2['undesired']['target'] == "attr2_<item_target>_undesired_value2"
    assert rule2['desired']['support'] == 0.7
    assert rule2['desired']['confidence'] == 0.8
    assert rule2['desired']['target'] == "attr2_<item_target>_desired_value2"
    assert rule2['uplift'] == 1.1
    assert rule2['undesired']['itemset'] == [
        "stable_attr2_<item_stable>_stable_value2",
        "flexible_attr2_<item_flexible>_flex_undesired2",
    ]
    assert rule2['desired']['itemset'] == [
        "stable_attr2_<item_stable>_stable_value2",
        "flexible_attr2_<item_flexible>_flex_desired2",
    ]
