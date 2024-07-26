"""Class Rules."""

from collections import defaultdict


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

    def __init__(self, undesired_state: str, desired_state: str, columns: list, count_transactions: int):
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
                    uplift = self.calculate_uplift(
                        undesired_rule['support'],
                        undesired_rule['confidence'],
                        desired_rule['confidence'],
                    )
                    self.action_rules.append({'undesired': undesired_rule, 'desired': desired_rule, 'uplift': uplift})

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
            ((desired_confidence - (1 - undesired_confidence)) * undesired_support) / self.count_transactions.

        Notes
        -----
        Uplift measures the increase in the probability of achieving the desired state when applying the action rule
        compared to not applying it. It is used to assess the effectiveness of the rule.
        """
        return ((desired_confidence - (1 - undesired_confidence)) * undesired_support) / self.count_transactions
