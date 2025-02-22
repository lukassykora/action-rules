#!/usr/bin/env python
"""Tests for `action_rules` package."""

import pytest

from action_rules.rules.rules import Rules


@pytest.fixture
def rules():
    """Fixture for Rules instance."""
    return Rules(
        'status_<item_target>_default',
        'status_<item_target>_paid',
        ['age_<item_stable>_30', 'age_<item_stable>_40'],
        20,
    )


@pytest.fixture
def rules_with_utilities():
    """
    Fixture for a Rules instance with preset utility tables.

    The intrinsic utility table maps:
        0 -> 1.0   (e.g., representing 'age_<item_stable>_30')
        1 -> 2.0   (e.g., representing 'age_<item_stable>_40')

    The transition utility table maps:
        (0, 1) -> 0.5  (transition gain from item 0 to item 1)
    """
    intrinsic_table = {0: 1.0, 1: 2.0}
    transition_table = {(0, 1): 0.5}
    return Rules(
        'status_<item_target>_default',
        'status_<item_target>_paid',
        ['age_<item_stable>_30', 'age_<item_stable>_40'],
        20,
        intrinsic_utility_table=intrinsic_table,
        transition_utility_table=transition_table,
    )


def test_add_classification_rules(rules):
    """Test the add_classification_rules method of Rules."""
    new_ar_prefix = tuple()
    itemset_prefix = tuple()
    undesired_states = [{'item': 'age_<item_stable>_30', 'support': 10, 'confidence': 0.8}]
    desired_states = [{'item': 'age_<item_stable>_30', 'support': 5, 'confidence': 0.6}]
    rules.add_classification_rules(new_ar_prefix, itemset_prefix, undesired_states, desired_states)
    assert len(rules.classification_rules[new_ar_prefix]['undesired']) > 0
    assert len(rules.classification_rules[new_ar_prefix]['desired']) > 0


def test_generate_action_rules(rules):
    """Test the generate_action_rules method of Rules."""
    new_ar_prefix = tuple()
    itemset_prefix = tuple()
    undesired_states = [{'item': 'age_<item_stable>_30', 'support': 10, 'confidence': 0.8}]
    desired_states = [{'item': 'age_<item_stable>_30', 'support': 5, 'confidence': 0.6}]
    rules.add_classification_rules(new_ar_prefix, itemset_prefix, undesired_states, desired_states)
    rules.generate_action_rules()
    assert len(rules.action_rules) > 0


def test_prune_classification_rules(rules):
    """Test the prune_classification_rules method of Rules."""
    new_ar_prefix = tuple()
    itemset_prefix = tuple()
    undesired_states = [{'item': 'age_<item_stable>_30', 'support': 10, 'confidence': 0.8}]
    desired_states = []
    rules.add_classification_rules(new_ar_prefix, itemset_prefix, undesired_states, desired_states)
    stop_list = []
    rules.prune_classification_rules(0, stop_list)
    assert len(stop_list) > 0


def test_calculate_confidence(rules):
    """Test the calculate_confidence method of Rules."""
    confidence = rules.calculate_confidence(10, 5)
    assert confidence == 0.6666666666666666


def test_calculate_uplift(rules):
    """Test the calculate_uplift method of Rules."""
    uplift = rules.calculate_uplift(10, 0.8, 0.6)
    assert uplift == 0.25


def test_compute_rule_utilities(rules_with_utilities):
    """
    Test the compute_rule_utilities method of Rules.

    Using:
      - undesired_rule with itemset [0] (intrinsic utility = 1.0)
      - desired_rule with itemset [1] (intrinsic utility = 2.0)
      - transition utility for (0, 1) = 0.5
    Expected:
      - u_undesired = 1.0
      - u_desired = 2.0
      - rule_utility_difference = 2.0 - 1.0 = 1.0
      - transition_gain = 0.5 (since items differ)
      - rule_utility_gain = 1.0 + 0.5 = 1.5
    """
    undesired_rule = {'itemset': [0]}
    desired_rule = {'itemset': [1]}
    u_undesired, u_desired, diff, trans_gain, rule_gain = rules_with_utilities.compute_rule_utilities(
        undesired_rule, desired_rule
    )
    assert u_undesired == 1.0
    assert u_desired == 2.0
    assert diff == 1.0
    assert trans_gain == 0.5
    assert rule_gain == 1.5


def test_compute_realistic_rule_utilities(rules_with_utilities):
    """
    Test the compute_realistic_rule_utilities method of Rules.

    Using the same base values from test_compute_rule_utilities:
      - undesired_rule_utility = 1.0, desired_rule_utility = 2.0, transition_gain = 0.5.
    Additionally, set:
      - undesired_rule with confidence 0.8 and support 10.
      - desired_rule with confidence 0.6.
    Expected calculations:
      - realistic_undesired_utility = 0.8*1.0 + 0.2*2.0 = 0.8 + 0.4 = 1.2.
      - realistic_desired_utility = 0.4*1.0 + 0.6*2.0 = 0.4 + 1.2 = 1.6.
      - realistic_rule_difference = 1.6 - 1.2 = 0.4.
      - effective_transactions = support / 0.8 = 10 / 0.8 = 12.5.
      - realistic_rule_gain_dataset = 12.5 * (realistic_rule_difference + transition_gain)
                                      = 12.5 * (0.4 + 0.5) = 12.5 * 0.9 = 11.25.
      - transition_gain_dataset = 12.5 * 0.5 = 6.25.
    """
    undesired_rule = {'itemset': [0], 'support': 10, 'confidence': 0.8}
    desired_rule = {'itemset': [1], 'confidence': 0.6}
    # Base values from compute_rule_utilities
    base_u_undesired, base_u_desired, _, base_trans_gain, _ = rules_with_utilities.compute_rule_utilities(
        undesired_rule, desired_rule
    )
    (realistic_undesired, realistic_desired, realistic_diff, trans_gain_dataset, realistic_gain_dataset) = (
        rules_with_utilities.compute_realistic_rule_utilities(
            undesired_rule, desired_rule, base_u_undesired, base_u_desired, base_trans_gain
        )
    )
    # Check expected values:
    assert pytest.approx(realistic_undesired, rel=1e-5) == 1.2
    assert pytest.approx(realistic_desired, rel=1e-5) == 1.6
    assert pytest.approx(realistic_diff, rel=1e-5) == 0.4
    assert pytest.approx(trans_gain_dataset, rel=1e-5) == 6.25
    assert pytest.approx(realistic_gain_dataset, rel=1e-5) == 11.25
