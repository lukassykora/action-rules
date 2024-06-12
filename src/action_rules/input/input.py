"""Class Input."""

import json

from action_rules.output import Output


class Input:
    """
    A class used to import action rules.

    Methods
    -------
    import_action_rules()
        Import action rules from a JSON string and set the action_rules attribute.
    """

    def __init__(self):
        """Initialize the Output class with the specified action rules and target attribute."""

    def import_action_rules(self, json_data: str) -> Output:
        """
        Import action rules from a JSON string and set the action_rules attribute.

        Parameters
        ----------
        json_data : str
            JSON string representing the action rules.

        Returns
        -------
        Output
            Output object representing the action rules.
        """
        rules = json.loads(json_data)
        action_rules = []
        target = rules[0]['target']['attribute']
        for rule in rules:
            ar_dict = {
                'undesired': {
                    'itemset': [],
                    'support': rule['support of undesired part'],
                    'confidence': rule['confidence of undesired part'],
                    'target': f"{rule['target']['attribute']}_<item_target>_{rule['target']['undesired']}",
                },
                'desired': {
                    'itemset': [],
                    'support': rule['support of desired part'],
                    'confidence': rule['confidence of desired part'],
                    'target': f"{rule['target']['attribute']}_<item_target>_{rule['target']['desired']}",
                },
                'uplift': rule['uplift'],
            }

            for item in rule['stable']:
                if 'flexible_as_stable' in item:
                    ar_dict['undesired']['itemset'].append(f"{item['attribute']}_<item_flexible>_{item['value']}")
                    ar_dict['desired']['itemset'].append(f"{item['attribute']}_<item_flexible>_{item['value']}")
                else:
                    ar_dict['undesired']['itemset'].append(f"{item['attribute']}_<item_stable>_{item['value']}")
                    ar_dict['desired']['itemset'].append(f"{item['attribute']}_<item_stable>_{item['value']}")

            for item in rule['flexible']:
                ar_dict['undesired']['itemset'].append(f"{item['attribute']}_<item_flexible>_{item['undesired']}")
                ar_dict['desired']['itemset'].append(f"{item['attribute']}_<item_flexible>_{item['desired']}")

            action_rules.append(ar_dict)

        return Output(action_rules, target)
