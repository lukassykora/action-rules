"""Class Output."""

import json


class Output:
    """
    A class used to format and export action rules.

    Attributes
    ----------
    action_rules : list
        List containing the action rules.
    target : str
        The target attribute for the action rules.
    stable_cols : list
        List of indices for stable columns.
    flexible_cols : list
        List of indices for flexible columns.
    column_values : dict
        Dictionary containing the values of the columns.

    Methods
    -------
    get_ar_notation()
        Generate a string representation of the action rules in a human-readable format.
    get_export_notation()
        Generate a JSON string of dictionaries representing the action rules for export.
    get_pretty_ar_notation()
        Generate a list of text strings representing the action rules.
    """

    def __init__(
        self,
        action_rules: list,
        target: str,
        stable_items_binding: dict,
        flexible_items_binding: dict,
        column_values: dict,
    ):
        """
        Initialize the Output class with the specified action rules and target attribute.

        Parameters
        ----------
        action_rules : list
            List containing the action rules.
        target : str
            The target attribute for the action rules.
        stable_items_binding : dict
            Dictionary containing bindings for stable items.
        flexible_items_binding : dict
            Dictionary containing bindings for flexible items.
        column_values : dict
            Dictionary containing the values of the columns.

        Notes
        -----
        The constructor initializes the Output object by setting the provided action rules, target attribute,
        stable items, flexible items, and column values. It flattens the stable and flexible items bindings to
        create lists of indices for stable and flexible columns.
        """
        self.action_rules = action_rules
        self.target = target
        self.stable_cols = [item for sublist in stable_items_binding.values() for item in sublist]
        self.flexible_cols = [item for sublist in flexible_items_binding.values() for item in sublist]
        self.column_values = column_values

    def get_ar_notation(self):
        """
        Generate a string representation of the action rules in a human-readable format.

        Returns
        -------
        str
            String representation of the action rules.

        Notes
        -----
        This method constructs a human-readable string representation of the action rules. Each rule is
        formatted to show the attribute-value conditions and transitions. The representation includes
        the support and confidence values for both the undesired and desired parts, as well as the uplift.
        """
        ar_notation = []
        for action_rule in self.action_rules:
            rule = '['
            for i, item in enumerate(action_rule['undesired']['itemset']):
                if i > 0:
                    rule += ' ∧ '
                rule += '('
                if item == action_rule['desired']['itemset'][i]:
                    if item in self.stable_cols:
                        val = self.column_values[item]
                        rule += str(val[0]) + ': ' + str(val[1])
                    else:
                        val = self.column_values[item]
                        rule += str(val[0]) + '*: ' + str(val[1])
                else:
                    val = self.column_values[item]
                    val_desired = self.column_values[action_rule['desired']['itemset'][i]]
                    rule += str(val[0]) + ': ' + str(val[1]) + ' → ' + str(val_desired[1])
                rule += ')'
            rule += (
                '] ⇒ ['
                + str(self.target)
                + ': '
                + str(self.column_values[action_rule['undesired']['target']][1])
                + ' → '
                + str(self.column_values[action_rule['desired']['target']][1])
                + ']'
            )
            rule += (
                ', support of undesired part: '
                + str(action_rule['undesired']['support'])
                + ', confidence of undesired part: '
                + str(action_rule['undesired']['confidence'])
            )
            rule += (
                ', support of desired part: '
                + str(action_rule['desired']['support'])
                + ', confidence of desired part: '
                + str(action_rule['desired']['confidence'])
            )
            rule += ', uplift: ' + str(action_rule['uplift'])
            ar_notation.append(rule)
        return ar_notation

    def get_export_notation(self):
        """
        Generate a JSON string of dictionaries representing the action rules for export.

        Returns
        -------
        str
            JSON string of dictionaries representing the action rules.

        Notes
        -----
        This method constructs a list of dictionaries where each dictionary represents an action rule.
        The dictionaries include attributes for stable and flexible items, as well as the target attribute,
        support, confidence, and uplift values. The list is then converted to a JSON string for export.
        """
        rules = []
        for ar_dict in self.action_rules:
            rule = {'stable': [], 'flexible': []}
            for i, item in enumerate(ar_dict['undesired']['itemset']):
                if item == ar_dict['desired']['itemset'][i]:
                    if item in self.stable_cols:
                        val = self.column_values[item]
                        rule['stable'].append({'attribute': val[0], 'value': val[1]})
                    else:
                        val = self.column_values[item]
                        rule['stable'].append({'attribute': val[0], 'value': val[1], 'flexible_as_stable': True})
                else:
                    val = self.column_values[item]
                    val_desired = self.column_values[ar_dict['desired']['itemset'][i]]
                    rule['flexible'].append({'attribute': val[0], 'undesired': val[1], 'desired': val_desired[1]})
            rule['target'] = {
                'attribute': self.target,
                'undesired': str(self.column_values[ar_dict['undesired']['target']][1]),
                'desired': str(self.column_values[ar_dict['desired']['target']][1]),
            }
            rule['support of undesired part'] = int(ar_dict['undesired']['support'])
            rule['confidence of undesired part'] = float(ar_dict['undesired']['confidence'])
            rule['support of desired part'] = int(ar_dict['desired']['support'])
            rule['confidence of desired part'] = float(ar_dict['desired']['confidence'])
            rule['uplift'] = float(ar_dict['uplift'])
            rules.append(rule)
        return json.dumps(rules)

    def get_pretty_ar_notation(self):
        """
        Generate a list of text strings representing the action rules.

        Returns
        -------
        list
            List of text strings representing the action rules.

        Notes
        -----
        This method constructs a list of text strings where each string represents an action rule in a
        readable format. The format includes conditions and transitions for each attribute, along with
        the target attribute change, support, confidence, and uplift values.
        """
        rules = []
        for ar_dict in self.action_rules:
            text = "If "
            for i, item in enumerate(ar_dict['undesired']['itemset']):
                if item == ar_dict['desired']['itemset'][i]:
                    if item in self.stable_cols:
                        val = self.column_values[item]
                        text += "attribute '" + val[0] + "' is '" + val[1] + "', "
                    else:
                        val = self.column_values[item]
                        text += "attribute (flexible is used as stable) '" + val[0] + "' is '" + val[1] + "', "
                else:
                    val = self.column_values[item]
                    val_desired = self.column_values[ar_dict['desired']['itemset'][i]]
                    text += "attribute '" + val[0] + "' value '" + val[1] + "' is changed to '" + val_desired[1] + "', "
            text += (
                "then '"
                + self.target
                + "' value '"
                + self.column_values[ar_dict['undesired']['target']][1]
                + "' is changed to '"
                + self.column_values[ar_dict['desired']['target']][1]
                + " with uplift: "
                + str(ar_dict['uplift'])
                + ", support of undesired part: "
                + str(ar_dict['undesired']['support'])
                + ", confidence of undesired part: "
                + str(ar_dict['undesired']['confidence'])
                + ", support of desired part: "
                + str(ar_dict['desired']['support'])
                + ", confidence of desired part: "
                + str(ar_dict['desired']['confidence'])
                + "."
            )
            rules.append(text)
        return rules
