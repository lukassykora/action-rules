#!/usr/bin/env python
"""Tests for `action_rules` package."""

import json

import pytest

from action_rules.input import Input
from action_rules.output import Output


@pytest.fixture
def sample_json_data():
    """Fixture for sample JSON data to be used in tests."""
    return json.dumps(
        [
            {
                'support of undesired part': 10,
                'confidence of undesired part': 0.8,
                'support of desired part': 5,
                'confidence of desired part': 0.6,
                'uplift': 0.2,
                'target': {'attribute': 'status', 'undesired': 'default', 'desired': 'paid'},
                'stable': [{'attribute': 'age', 'value': 30}],
                'flexible': [{'attribute': 'income', 'undesired': 'low', 'desired': 'medium'}],
            }
        ]
    )


@pytest.fixture
def input_instance():
    """Fixture for Input instance."""
    return Input()


def test_import_action_rules(input_instance, sample_json_data):
    """Test the import_action_rules method of Input."""
    output = input_instance.import_action_rules(sample_json_data)
    assert isinstance(output, Output)
    assert len(output.action_rules) == 1

    action_rule = output.action_rules[0]

    assert 'undesired' in action_rule
    assert 'desired' in action_rule
    assert action_rule['undesired']['support'] == 10
    assert action_rule['undesired']['confidence'] == 0.8
    assert action_rule['undesired']['target'] == 'status_<item_target>_default'
    assert action_rule['desired']['support'] == 5
    assert action_rule['desired']['confidence'] == 0.6
    assert action_rule['desired']['target'] == 'status_<item_target>_paid'
    assert action_rule['uplift'] == 0.2

    undesired_itemset = action_rule['undesired']['itemset']
    desired_itemset = action_rule['desired']['itemset']

    assert 'age_<item_stable>_30' in undesired_itemset
    assert 'income_<item_flexible>_low' in undesired_itemset
    assert 'age_<item_stable>_30' in desired_itemset
    assert 'income_<item_flexible>_medium' in desired_itemset
