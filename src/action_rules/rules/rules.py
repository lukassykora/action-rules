"""Class Rules."""

from collections import defaultdict
from typing import Optional  # noqa


class Rules:
    """
    A class used to manage and generate classification and action rules.

    Attributes
    ----------
    classification_rules : defaultdict
        Default dictionary to store classification rules for undesired and desired states.
    undesired_state : str
        The undesired state of the target attribute.
    desired_state : str
        The desired state of the target attribute.
    columns : list
        List of columns in the dataset.
    action_rules : list
        List to store generated action rules.
    undesired_prefixes_without_conf : set
        Set to store prefixes of undesired states without conflicts.
    desired_prefixes_without_conf : set
        Set to store prefixes of desired states without conflicts.
    count_transactions : int
        The number of transactions in the data.
    intrinsic_utility_table : dict, optional
        (attribute, value) -> float
        A lookup table for the intrinsic utility of each attribute-value pair.
        If None, no intrinsic utility is considered.
    transition_utility_table : dict, optional
        (attribute, from_value, to_value) -> float
        A lookup table for cost/gain of transitions between values.
        If None, no transition utility is considered.

    Methods
    -------
    add_prefix_without_conf(prefix, is_desired)
        Add a prefix to the set of prefixes without conflicts.
    add_classification_rules(new_ar_prefix, itemset_prefix, undesired_states, desired_states)
        Add classification rules for undesired and desired states.
    generate_action_rules()
        Generate action rules from classification rules.
    prune_classification_rules(k, stop_list)
        Prune classification rules based on their length and update the stop list.
    calculate_confidence(support, opposite_support)
        Calculate the confidence of a rule.
    calculate_uplift(undesired_support, undesired_confidence, desired_confidence)
        Calculate the uplift of an action rule.
    """

    def __init__(
        self,
        undesired_state: str,
        desired_state: str,
        columns: list,
        count_transactions: int,
        intrinsic_utility_table: Optional[dict] = None,
        transition_utility_table: Optional[dict] = None,
    ):
        """
        Initialize the Rules class with the specified undesired and desired states, columns, and transaction count.

        Parameters
        ----------
        undesired_state : str
            The undesired state of the target attribute.
        desired_state : str
            The desired state of the target attribute.
        columns : list
            List of columns in the dataset.
        count_transactions : int
            The number of transactions in the data.
        intrinsic_utility_table : dict, optional
            (attribute, value) -> float
            A lookup table for the intrinsic utility of each attribute-value pair.
            If None, no intrinsic utility is considered.
        transition_utility_table : dict, optional
            (attribute, from_value, to_value) -> float
            A lookup table for cost/gain of transitions between values.
            If None, no transition utility is considered.

        Notes
        -----
        The classification_rules attribute is initialized as a defaultdict with a lambda function that creates
        dictionaries for 'desired' and 'undesired' states.
        """
        self.classification_rules = defaultdict(lambda: {'desired': [], 'undesired': []})  # type: defaultdict
        self.undesired_state = undesired_state
        self.columns = columns
        self.desired_state = desired_state
        self.action_rules = []  # type: list
        self.undesired_prefixes_without_conf = set()  # type: set
        self.desired_prefixes_without_conf = set()  # type: set
        self.count_transactions = count_transactions
        self.intrinsic_utility_table = intrinsic_utility_table or {}
        self.transition_utility_table = transition_utility_table or {}

    def add_prefix_without_conf(self, prefix: tuple, is_desired: bool):
        """
        Add a prefix to the set of prefixes without conflicts.

        Parameters
        ----------
        prefix : tuple
            The prefix to be added.
        is_desired : bool
            If True, add the prefix to the desired prefixes set; otherwise, add it to the undesired prefixes set.

        Notes
        -----
        This method is useful for keeping track of prefixes that have no conflicting rules and can be
        used directly in rule generation.
        """
        if is_desired:
            self.desired_prefixes_without_conf.add(prefix)
        else:
            self.undesired_prefixes_without_conf.add(prefix)

    def add_classification_rules(self, new_ar_prefix, itemset_prefix, undesired_states, desired_states):
        """
        Add classification rules for undesired and desired states.

        Parameters
        ----------
        new_ar_prefix : tuple
            Prefix of the action rule.
        itemset_prefix : tuple
            Prefix of the itemset.
        undesired_states : list
            List of dictionaries containing undesired state information.
        desired_states : list
            List of dictionaries containing desired state information.

        Notes
        -----
        This method updates the classification_rules attribute with new rules based on the provided
        undesired and desired states. Each state is represented as a dictionary containing item, support,
        confidence, and target information.
        """
        for undesired_item in undesired_states:
            new_itemset_prefix = itemset_prefix + (undesired_item['item'],)
            self.classification_rules[new_ar_prefix]['undesired'].append(
                {
                    'itemset': new_itemset_prefix,
                    'support': undesired_item['support'],
                    'confidence': undesired_item['confidence'],
                    'target': self.undesired_state,
                }
            )
        for desired_item in desired_states:
            new_itemset_prefix = itemset_prefix + (desired_item['item'],)
            self.classification_rules[new_ar_prefix]['desired'].append(
                {
                    'itemset': new_itemset_prefix,
                    'support': desired_item['support'],
                    'confidence': desired_item['confidence'],
                    'target': self.desired_state,
                }
            )

    def generate_action_rules(self):
        """
        Generate action rules from classification rules.

        Notes
        -----
        This method creates action rules by combining classification rules for undesired and desired states.
        The uplift for each action rule is calculated using the `calculate_uplift` method and the result is
        stored in the action_rules attribute.
        """
        for attribute_prefix, rules in self.classification_rules.items():
            for desired_rule in rules['desired']:
                for undesired_rule in rules['undesired']:
                    # Uplift
                    uplift = self.calculate_uplift(
                        undesired_rule['support'],
                        undesired_rule['confidence'],
                        desired_rule['confidence'],
                    )
                    # Utility
                    utility = None
                    if self.intrinsic_utility_table is not None or self.transition_utility_table is not None:
                        (
                            max_rule_gain,
                            realistic_rule_gain,
                            realistic_rule_gain_dataset
                        ) = self.compute_rule_utility(undesired_rule, desired_rule)
                        utility = {
                            'max_rule_gain': max_rule_gain,
                            'realistic_rule_gain': realistic_rule_gain,
                            'realistic_dataset_gain': realistic_rule_gain_dataset,
                        }
                    # Action rule measures
                    ar_support, ar_confidence = self.compute_action_rule_measures(
                        undesired_rule.get('support', 0.0),
                        undesired_rule.get('confidence', 0.0),
                        desired_rule.get('support', 0.0),
                        desired_rule.get('confidence', 0.0),
                    )
                    self.action_rules.append(
                        {
                            'undesired': undesired_rule,
                            'desired': desired_rule,
                            'uplift': uplift,
                            'support': ar_support,
                            'confidence': ar_confidence,
                            **utility,
                        }
                    )

    def prune_classification_rules(self, k: int, stop_list: list):
        """
        Prune classification rules based on their length and update the stop list.

        Parameters
        ----------
        k : int
            Length of the attribute prefix.
        stop_list : list
            List of prefixes to stop generating rules for.

        Notes
        -----
        This method removes classification rules whose prefix length equals k and either desired or undesired
        states are empty. The corresponding prefixes are also added to the stop_list to avoid further rule generation.
        """
        del_prefixes = []
        for attribute_prefix, rules in self.classification_rules.items():
            if k == len(attribute_prefix):
                len_desired = len(rules['desired'])
                len_undesired = len(rules['undesired'])
                if len_desired == 0 or len_undesired == 0:
                    if (len_desired == 0 and attribute_prefix not in self.desired_prefixes_without_conf) or (
                        len_undesired == 0 and attribute_prefix not in self.undesired_prefixes_without_conf
                    ):
                        stop_list.append(attribute_prefix)
                    del_prefixes.append(attribute_prefix)
        for attribute_prefix in del_prefixes:
            del self.classification_rules[attribute_prefix]

    def calculate_confidence(self, support, opposite_support):
        """
        Calculate the confidence of a rule.

        Parameters
        ----------
        support : int
            The support value for the desired or undesired state.
        opposite_support : int
            The support value for the opposite state.

        Returns
        -------
        float
            The confidence value calculated as support / (support + opposite_support).
            Returns 0 if the sum of support and opposite_support is 0.

        Notes
        -----
        Confidence is a measure of the reliability of a rule. A higher confidence indicates a stronger
        association between the conditions of the rule and the target state.
        """
        if support + opposite_support == 0:
            return 0
        return support / (support + opposite_support)

    def calculate_uplift(self, undesired_support: int, undesired_confidence: float, desired_confidence: float) -> float:
        """
        Calculate the uplift of an action rule.

        Parameters
        ----------
        undesired_support : int
            The support value for the undesired state.
        undesired_confidence : float
            The confidence value for the undesired state.
        desired_confidence : float
            The confidence value for the desired state.

        Returns
        -------
        float
            The uplift value calculated as:
            ((desired_confidence - (1 - undesired_confidence)) * (undesired_support / undesired_confidence))
            / self.count_transactions.

        Notes
        -----
        Uplift measures the increase in the probability of achieving the desired state when applying the action rule
        compared to not applying it. It is used to assess the effectiveness of the rule.
        """
        return (
            (desired_confidence - (1 - undesired_confidence)) * (undesired_support / undesired_confidence)
        ) / self.count_transactions

    def compute_rule_utility(self, undesired_rule: dict, desired_rule: dict):
        u_undesired = 0.0
        # Sum intrinsic utilities for each item index in the undesired rule's itemset.
        for idx in undesired_rule.get('itemset', []):
            intrinsic_value = self.intrinsic_utility_table.get(idx, 0.0)
            u_undesired += intrinsic_value

        # Initialize the desired rule utility.
        u_desired = 0.0
        # Sum intrinsic utilities for each item index in the desired rule's itemset.
        for idx in desired_rule.get('itemset', []):
            intrinsic_value = self.intrinsic_utility_table.get(idx, 0.0)
            u_desired += intrinsic_value

        # Initialize additional transition gain.
        transition_gain = 0.0
        # Iterate over corresponding item indices from undesired and desired itemsets.
        for u_idx, d_idx in zip(undesired_rule.get('itemset', []), desired_rule.get('itemset', [])):
            # Only add transition gain if the indices differ (indicating a change in a flexible attribute).
            if u_idx != d_idx:
                trans_value = self.transition_utility_table.get((u_idx, d_idx), 0.0)
                transition_gain += trans_value

        rule_gain = u_desired - u_undesired + transition_gain

        # Target utility
        u_undesired_target = self.intrinsic_utility_table.get(self.undesired_state, 0.0)
        u_desired_target = self.intrinsic_utility_table.get(self.desired_state, 0.0)
        transition_gain_target = self.transition_utility_table.get((self.undesired_state, self.desired_state), 0.0)

        target_gain = u_desired_target - u_undesired_target + transition_gain_target

        # Realistic target gain
        undesired_rule_confidence = undesired_rule.get('confidence', 0.0)
        desired_rule_confidence = desired_rule.get('confidence', 0.0)
        target_gain_realistic = (desired_rule_confidence - (1 - undesired_rule_confidence)) * target_gain

        # Rule gain
        max_rule_gain = rule_gain + target_gain
        realistic_rule_gain = rule_gain + target_gain_realistic

        # Compute dataset-level realistic gain.
        support = undesired_rule.get('support', 0)
        if undesired_rule_confidence > 0:
            transactions = support / undesired_rule_confidence
        else:
            transactions = 0.0
        realistic_rule_gain_dataset = transactions * realistic_rule_gain

        return max_rule_gain, realistic_rule_gain, realistic_rule_gain_dataset


    def compute_action_rule_measures(
        self, support_undesired, confidence_undesired, support_desired, confidence_desired
    ):
        """
        Compute the support and confidence for an action rule formed from an undesired rule and a desired rule.

        The action rule is derived by pairing a classification rule that leads to an undesired outcome
        with a classification rule that leads to a desired outcome. In this formulation, the support
        of the action rule is defined as the minimum of the supports of the two component rules, and
        the confidence of the action rule is defined as the product of their confidences.

        Parameters
        ----------
        support_undesired : float
            The support of the undesired rule (e.g., count or relative frequency).
        confidence_undesired : float
            The confidence of the undesired rule (a value between 0 and 1).
        support_desired : float
            The support of the desired rule.
        confidence_desired : float
            The confidence of the desired rule (a value between 0 and 1).

        Returns
        -------
        tuple of (float, float)
            A tuple containing:
                - action_support : float
                    The support of the action rule, computed as min(support_undesired, support_desired).
                - action_confidence : float
                    The confidence of the action rule, computed as confidence_undesired * confidence_desired.

        """
        action_support = min(support_undesired, support_desired)
        action_confidence = confidence_undesired * confidence_desired
        return action_support, action_confidence
