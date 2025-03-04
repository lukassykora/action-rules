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


def test_compute_rule_utility(rules):
    """
    Test the compute_rule_utility method of Rules.

    This test sets up custom intrinsic and transition utility tables and defines
    sample undesired and desired rules to verify that compute_rule_utility returns
    the expected tuple of (max_rule_gain, realistic_rule_gain, realistic_rule_gain_dataset).

    Test scenario:
      - intrinsic_utility_table:
          * 0: 1.0
          * 1: 2.0
          * 2: 3.0
          * rules.undesired_state: 2.0
          * rules.desired_state: 5.0
      - transition_utility_table:
          * (0, 1): 1.5
          * (rules.undesired_state, rules.desired_state): 3.0
      - undesired_rule:
          * itemset: [0, 2]
          * confidence: 0.8
          * support: 10
      - desired_rule:
          * itemset: [1, 2]
          * confidence: 0.6

    Expected computation:
      1. u_undesired = 1.0 (for 0) + 3.0 (for 2) = 4.0
      2. u_desired   = 2.0 (for 1) + 3.0 (for 2) = 5.0
      3. Transition gain from (0,1): 1.5 (since 0 != 1), and no gain for (2,2)
         => rule_gain = (5.0 - 4.0 + 1.5) = 2.5
      4. Target utilities:
         - u_undesired_target = 2.0, u_desired_target = 5.0,
         - transition_gain_target = 3.0,
         => target_gain = (5.0 - 2.0 + 3.0) = 6.0
      5. Realistic target gain:
         = (0.6 - (1 - 0.8)) * 6.0 = (0.6 - 0.2) * 6.0 = 2.4
      6. max_rule_gain = 2.5 + 6.0 = 8.5
         realistic_rule_gain = 2.5 + 2.4 = 4.9
      7. Transactions estimated as: support / confidence = 10 / 0.8 = 12.5
         => realistic_rule_gain_dataset = 12.5 * 4.9 = 61.25

    Returns
    -------
    tuple of (float, float, float)
        The computed gains: (max_rule_gain, realistic_rule_gain, realistic_rule_gain_dataset)
    """
    # Setup custom utility tables for testing.
    rules.intrinsic_utility_table = {
        0: 1.0,
        1: 2.0,
        2: 3.0,
        rules.undesired_state: 2.0,
        rules.desired_state: 5.0,
    }
    rules.transition_utility_table = {
        (0, 1): 1.5,
        (rules.undesired_state, rules.desired_state): 3.0,
    }

    # Define test rules.
    undesired_rule = {'itemset': [0, 2], 'confidence': 0.8, 'support': 10}
    desired_rule = {'itemset': [1, 2], 'confidence': 0.6}

    # Compute the utility gains.
    result = rules.compute_rule_utility(undesired_rule, desired_rule)
    expected = (8.5, 4.9, 61.25)
    tol = 1e-6  # tolerance for floating point comparisons

    assert abs(result[0] - expected[0]) < tol, f"max_rule_gain: expected {expected[0]}, got {result[0]}"
    assert abs(result[1] - expected[1]) < tol, f"realistic_rule_gain: expected {expected[1]}, got {result[1]}"
    assert abs(result[2] - expected[2]) < tol, f"realistic_rule_gain_dataset: expected {expected[2]}, got {result[2]}"
