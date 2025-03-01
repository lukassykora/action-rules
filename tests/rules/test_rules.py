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
        'status_<item_target>_default' -> 0.5  (intrinsic utility for the undesired target)
        'status_<item_target>_paid'    -> 1.0  (intrinsic utility for the desired target)

    The transition utility table maps:
        (0, 1) -> 0.5  (transition gain for a flexible attribute change)
        ('status_<item_target>_default', 'status_<item_target>_paid') -> 0.3
            (transition gain for changing the target state)
    """
    intrinsic_table = {0: 1.0, 1: 2.0, 'status_<item_target>_default': 0.5, 'status_<item_target>_paid': 1.0}
    transition_table = {(0, 1): 0.5, ('status_<item_target>_default', 'status_<item_target>_paid'): 0.3}
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
      - Target intrinsic utilities: 0.5 for undesired, 1.0 for desired.
      - Transition utility for (0, 1) = 0.5 and for target transition = 0.3.
    Expected:
      - u_undesired = 1.0 + 0.5 = 1.5
      - u_desired = 2.0 + 1.0 = 3.0
      - rule_utility_difference = 3.0 - 1.5 = 1.5
      - transition_gain = 0.5 (flexible) + 0.3 (target) = 0.8
      - rule_utility_gain = 1.5 + 0.8 = 2.3
    """
    undesired_rule = {'itemset': [0]}
    desired_rule = {'itemset': [1]}
    u_undesired, u_desired, diff, trans_gain, rule_gain = rules_with_utilities.compute_rule_utilities(
        undesired_rule, desired_rule
    )
    assert u_undesired == 1.5
    assert u_desired == 3.0
    assert diff == 1.5
    assert trans_gain == 0.8
    assert rule_gain == 2.3


def test_compute_realistic_rule_utilities(rules_with_utilities):
    """
    Test the compute_realistic_rule_utilities method of Rules.

    Using the same base values from test_compute_rule_utilities:
      - undesired_rule_utility = 1.5, desired_rule_utility = 3.0, transition_gain = 0.8.
    Additionally, set:
      - undesired_rule with confidence 0.8 and support 10.
      - desired_rule with confidence 0.6.
    Expected calculations:
      - realistic_undesired_utility = 0.8*1.5 + 0.2*3.0 = 1.2 + 0.6 = 1.8.
      - realistic_desired_utility = 0.4*1.5 + 0.6*3.0 = 0.6 + 1.8 = 2.4.
      - realistic_rule_difference = 2.4 - 1.8 = 0.6.
      - effective_transactions = support / 0.8 = 10 / 0.8 = 12.5.
      - realistic_rule_gain_dataset = 12.5 * (0.6 + 0.8) = 12.5 * 1.4 = 17.5.
      - transition_gain_dataset = 12.5 * 0.8 = 10.0.
    """
    undesired_rule = {'itemset': [0], 'support': 10, 'confidence': 0.8}
    desired_rule = {'itemset': [1], 'confidence': 0.6}
    # Compute base utilities from compute_rule_utilities.
    base_u_undesired, base_u_desired, _, base_trans_gain, _ = rules_with_utilities.compute_rule_utilities(
        undesired_rule, desired_rule
    )
    (realistic_undesired, realistic_desired, realistic_diff, trans_gain_dataset, realistic_gain_dataset) = (
        rules_with_utilities.compute_realistic_rule_utilities(
            undesired_rule, desired_rule, base_u_undesired, base_u_desired, base_trans_gain
        )
    )
    assert pytest.approx(realistic_undesired, rel=1e-5) == 1.8
    assert pytest.approx(realistic_desired, rel=1e-5) == 2.4
    assert pytest.approx(realistic_diff, rel=1e-5) == 0.6
    assert pytest.approx(trans_gain_dataset, rel=1e-5) == 10.0
    assert pytest.approx(realistic_gain_dataset, rel=1e-5) == 17.5


def test_compute_action_rule_measures(rules):
    """
    Test the compute_action_rule_measures method of Rules.

    Using:
      - support_undesired = 10
      - confidence_undesired = 0.8
      - support_desired = 5
      - confidence_desired = 0.6

    Expected:
      - action_support = min(10, 5) = 5
      - action_confidence = 0.8 * 0.6 = 0.48
    """
    action_support, action_confidence = rules.compute_action_rule_measures(10, 0.8, 5, 0.6)
    assert action_support == 5
    assert action_confidence == 0.48


def test_add_prefix_without_conf(rules):
    """
    Test the add_prefix_without_conf method of Rules.

    This test checks that prefixes are correctly added to both the desired and undesired sets.
    """
    prefix = ('age_<item_stable>_30',)
    rules.add_prefix_without_conf(prefix, is_desired=True)
    rules.add_prefix_without_conf(prefix, is_desired=False)
    assert prefix in rules.desired_prefixes_without_conf
    assert prefix in rules.undesired_prefixes_without_conf
