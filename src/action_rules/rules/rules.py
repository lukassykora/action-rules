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
                            undesired_rule_utility,
                            desired_rule_utility,
                            rule_utility_difference,
                            transition_gain,
                            rule_utility_gain,
                        ) = self.compute_rule_utilities(undesired_rule, desired_rule)
                        (
                            realistic_undesired_utility,
                            realistic_desired_utility,
                            realistic_rule_difference,
                            transition_gain_dataset,
                            realistic_rule_gain_dataset,
                        ) = self.compute_realistic_rule_utilities(
                            undesired_rule, desired_rule, undesired_rule_utility, desired_rule_utility, transition_gain
                        )
                        utility = {
                            'undesired_rule_utility': undesired_rule_utility,
                            'desired_rule_utility': desired_rule_utility,
                            'rule_utility_difference': rule_utility_difference,
                            'transition_gain': transition_gain,
                            'rule_utility_gain': rule_utility_gain,
                            'realistic_undesired_utility': realistic_undesired_utility,
                            'realistic_desired_utility': realistic_desired_utility,
                            'realistic_rule_difference': realistic_rule_difference,
                            'transition_gain_dataset': transition_gain_dataset,
                            'realistic_rule_gain_dataset': realistic_rule_gain_dataset,
                        }
                    self.action_rules.append(
                        {'undesired': undesired_rule, 'desired': desired_rule, 'uplift': uplift, **utility}
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

    def compute_rule_utilities(self, undesired_rule: dict, desired_rule: dict) -> tuple:
        """
        Compute the base (intrinsic) rule-level utility measures for a candidate action rule.

        This method calculates:
          - undesired_rule_utility: The sum of intrinsic utility values for all items in the undesired rule's itemset.
          - desired_rule_utility: The sum of intrinsic utility values for all items in the desired rule's itemset.
          - rule_utility_difference: The difference (u_desired - u_undesired) based solely on intrinsic utilities.
          - transition_gain: The additional utility gained from flexible attribute changes.
          - rule_utility_gain: The overall net gain computed as (rule_utility_difference + transition_gain).

        Parameters
        ----------
        undesired_rule : dict
            A dictionary representing the undesired classification rule.
            Expected to have an 'itemset' key containing a list or tuple of internal column indices.
        desired_rule : dict
            A dictionary representing the desired classification rule.
            Expected to have an 'itemset' key containing a list or tuple of internal column indices.

        Returns
        -------
        tuple
            A tuple of five floats:
            (undesired_rule_utility, desired_rule_utility, rule_utility_difference,
             transition_gain, rule_utility_gain).

        Notes
        -----
        - It is assumed that self.intrinsic_utility_table maps internal column indices to intrinsic utility values.
        - self.transition_utility_table is assumed to map (from_index, to_index) pairs to transition utility values.
        - Only items present in the utility tables contribute to the computed utilities.
        """
        # Initialize the undesired rule utility.
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

        # Compute the intrinsic difference between desired and undesired utilities.
        rule_utility_difference = u_desired - u_undesired

        # Initialize additional transition gain.
        transition_gain = 0.0
        # Iterate over corresponding item indices from undesired and desired itemsets.
        for u_idx, d_idx in zip(undesired_rule.get('itemset', []), desired_rule.get('itemset', [])):
            # Only add transition gain if the indices differ (indicating a change in a flexible attribute).
            if u_idx != d_idx:
                trans_value = self.transition_utility_table.get((u_idx, d_idx), 0.0)
                transition_gain += trans_value

        # The overall rule utility gain includes the intrinsic difference plus the transition gain.
        rule_utility_gain = rule_utility_difference + transition_gain

        return u_undesired, u_desired, rule_utility_difference, transition_gain, rule_utility_gain

    def compute_realistic_rule_utilities(
        self,
        undesired_rule: dict,
        desired_rule: dict,
        undesired_rule_utility: float,
        desired_rule_utility: float,
        transition_gain: float,
    ) -> tuple:
        """
        Compute the confidence-based (realistic) rule-level utility measures for a candidate action rule.

        This method calculates realistic (i.e., confidence-scaled) utilities based on the provided base measures.
        In particular, it computes:

          - Realistic Undesired Utility: A weighted combination representing the effective undesired utility,

            U_undesired,realistic = c_u * U_undesired + (1 - c_u) * U_desired,

            where c_u is the confidence value from the undesired rule.

          - Realistic Desired Utility: A weighted combination representing the effective desired utility,

            U_desired,realistic = (1 - c_d) * U_undesired + c_d * U_desired,

            where c_d is the confidence value from the desired rule.

          - Realistic Rule Difference: The difference between the realistic desired and undesired utilities,

            ΔU_realistic = U_desired,realistic - U_undesired,realistic.

          - Dataset-Level Realistic Gain: The overall realistic rule gain is computed as
            (ΔU_realistic + G_trans), where G_trans is the base transition gain.
            The effective number of transactions is estimated as

              N_eff = support / c_u   (if c_u > 0),

            and the dataset-level realistic gain is then given by

              ΔU_dataset,realistic = N_eff * (ΔU_realistic + G_trans).

          - Dataset-Level Transition Gain: G_trans,dataset = N_eff * G_trans.

        Parameters
        ----------
        undesired_rule : dict
            A dictionary representing the undesired classification rule.
            Expected keys include:
                'itemset': list or tuple of internal column indices,
                'support': int, and
                'confidence': float (optional; defaults to 0.0).
        desired_rule : dict
            A dictionary representing the desired classification rule.
            Expected keys include:
                'itemset': list or tuple of internal column indices,
                'confidence': float (used as the fraction c_d; defaults to 0.0).
        undesired_rule_utility : float
            The base intrinsic utility sum for the undesired rule (from compute_rule_utilities).
        desired_rule_utility : float
            The base intrinsic utility sum for the desired rule (from compute_rule_utilities).
        transition_gain : float
            The base transition gain (from compute_rule_utilities).

        Returns
        -------
        tuple
            A tuple of five floats:
            (realistic_undesired_utility, realistic_desired_utility, realistic_rule_difference,
             transition_gain_dataset, realistic_rule_gain_dataset).

        Notes
        -----
        - The confidence value c is taken from undesired_rule['confidence'] (and desired_rule['confidence']
          for c_d), defaulting to 0.0 if not present.
        - The realistic intrinsic utilities are computed as weighted combinations of the base utilities:
              U_undesired,realistic = c_u * U_undesired + (1 - c_u) * U_desired
              U_desired,realistic   = (1 - c_d) * U_undesired + c_d * U_desired.
        - The realistic rule difference is ΔU_realistic = U_desired,realistic - U_undesired,realistic.
        - The dataset-level realistic gain scales the net realistic gain by the estimated number of transactions,
          where N_eff is approximated as support / c_u (if c_u > 0).
        """
        # Retrieve the confidence (c) from the desired rule; default to 0 if not present.
        undesired_rule_confidence = undesired_rule.get('confidence', 0.0)
        desired_rule_confidence = desired_rule.get('confidence', 0.0)
        realistic_undesired_utility = (
            undesired_rule_confidence * undesired_rule_utility + (1 - undesired_rule_confidence) * desired_rule_utility
        )
        realistic_desired_utility = (
            1 - desired_rule_confidence
        ) * undesired_rule_utility + desired_rule_confidence * desired_rule_utility

        # Compute the realistic intrinsic difference.
        realistic_rule_difference = realistic_desired_utility - realistic_undesired_utility

        # Overall realistic rule utility gain.
        realistic_rule_gain = realistic_rule_difference + transition_gain

        # Compute dataset-level realistic gain.
        support = undesired_rule.get('support', 0)
        if undesired_rule_confidence > 0:
            transactions = support / undesired_rule_confidence
        else:
            transactions = 0.0

        if transactions > 0:
            realistic_rule_gain_dataset = transactions * realistic_rule_gain
            transition_gain_dataset = transactions * transition_gain
        else:
            realistic_rule_gain_dataset = 0.0
            transition_gain_dataset = 0.0

        return (
            realistic_undesired_utility,
            realistic_desired_utility,
            realistic_rule_difference,
            transition_gain_dataset,
            realistic_rule_gain_dataset,
        )
