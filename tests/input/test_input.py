#!/usr/bin/env python
"""Tests for `action_rules` package."""

import json

import pytest

from action_rules.input import Input
from action_rules.output import Output


@pytest.fixture
def sample_json_data():
    """
    Fixture for sample JSON data to be used in tests.

    Returns
    -------
    str
        JSON string representing the action rules.
    """
    return json.dumps(
        [
            {
                "support of undesired part": 10,
                "confidence of undesired part": 0.8,
                "support of desired part": 5,
                "confidence of desired part": 0.6,
                "uplift": 0.2,
                "target": {"attribute": "status", "undesired": "default", "desired": "paid"},
                "stable": [
                    {"attribute": "age", "value": "30"},
                    {"attribute": "income", "value": "low", "flexible_as_stable": True},
                ],
                "flexible": [{"attribute": "income", "undesired": "low", "desired": "medium"}],
            }
        ]
    )


@pytest.fixture
def input_instance():
    """
    Fixture for Input instance.

    Returns
    -------
    Input
        Instance of the Input class.
    """
    return Input()


def test_import_action_rules(input_instance, sample_json_data):
    """
    Test the import_action_rules method of the Input class.

    Parameters
    ----------
    input_instance : Input
        Instance of the Input class.
    sample_json_data : str
        JSON string representing the action rules.
    """
    output = input_instance.import_action_rules(sample_json_data)
    assert isinstance(output, Output)
    assert len(output.action_rules) == 1
    assert output.target == "status"
    assert output.action_rules[0]['undesired']['support'] == 10
    assert output.action_rules[0]['desired']['support'] == 5
    assert output.action_rules[0]['undesired']['confidence'] == 0.8
    assert output.action_rules[0]['desired']['confidence'] == 0.6
    assert output.action_rules[0]['uplift'] == 0.2
    assert len(output.stable_cols) > 0
    assert len(output.flexible_cols) > 0
    assert len(output.column_values) > 0
