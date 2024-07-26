"""Class Input."""

import json

from action_rules.output import Output


class Input:
    """
    A class used to import action rules.

    Methods
    -------
    import_action_rules(json_data)
        Import action rules from a JSON string and set the action_rules attribute.
    """

    def __init__(self):
        """
        Initialize the Input class.

        Notes
        -----
        This class is used to import action rules from a JSON string and convert them into an Output object.
        """

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

        Notes
        -----
        This method parses a JSON string containing action rules, extracts relevant information,
        and constructs an Output object. The method initializes the target attribute, stable items,
        flexible items, and column values. It processes both stable and flexible items for each rule
        and updates the corresponding dictionaries.

        The JSON structure is expected to have the following format:
        [
            {
                "target": {
                    "attribute": "target_attribute",
                    "undesired": "undesired_value",
                    "desired": "desired_value"
                },
                "support of undesired part": int,
                "confidence of undesired part": float,
                "support of desired part": int,
                "confidence of desired part": float,
                "uplift": float,
                "stable": [
                    {
                        "attribute": "attribute_name",
                        "value": "attribute_value"
                    },
                    ...
                ],
                "flexible": [
                    {
                        "attribute": "attribute_name",
                        "undesired": "undesired_value",
                        "desired": "desired_value"
                    },
                    ...
                ]
            },
            ...
        ]

        The method ensures that each attribute-value pair is assigned a unique index and maintains
        the mappings in the column_values dictionary. The stable_items_binding and flexible_items_binding
        dictionaries are updated accordingly.

        Example
        -------
        json_data = '''
        [
            {
                "target": {
                    "attribute": "target",
                    "undesired": "no",
                    "desired": "yes"
                },
                "support of undesired part": 10,
                "confidence of undesired part": 0.5,
                "support of desired part": 20,
                "confidence of desired part": 0.8,
                "uplift": 0.3,
                "stable": [
                    {"attribute": "age", "value": "young"},
                    {"attribute": "income", "value": "high"}
                ],
                "flexible": [
                    {"attribute": "education", "undesired": "low", "desired": "high"}
                ]
            }
        ]
        '''
        input_obj = Input()
        output = input_obj.import_action_rules(json_data)
        """
        rules = json.loads(json_data)
        action_rules = []
        target = rules[0]['target']['attribute']
        stable_items_binding = {}  # type: dict
        flexible_items_binding = {}  # type: dict
        column_values = {}
        highest_index = 0
        for rule in rules:
            if highest_index == 0:
                column_values[highest_index] = (rule['target']['attribute'], rule['target']['undesired'])
                highest_index += 1
                column_values[highest_index] = (rule['target']['attribute'], rule['target']['desired'])
                highest_index += 1
            ar_dict = {
                'undesired': {
                    'itemset': [],
                    'support': rule['support of undesired part'],
                    'confidence': rule['confidence of undesired part'],
                    'target': 0,
                },
                'desired': {
                    'itemset': [],
                    'support': rule['support of desired part'],
                    'confidence': rule['confidence of desired part'],
                    'target': 1,
                },
                'uplift': rule['uplift'],
            }
            for item in rule['stable']:
                if (item['attribute'], item['value']) not in column_values.values():
                    column_values[highest_index] = (item['attribute'], item['value'])
                    if 'flexible_as_stable' in item:
                        if item['attribute'] not in flexible_items_binding.keys():
                            flexible_items_binding.update({item['attribute']: []})
                        flexible_items_binding[item['attribute']].append(highest_index)
                    else:
                        if item['attribute'] not in stable_items_binding.keys():
                            stable_items_binding.update({item['attribute']: []})
                        stable_items_binding[item['attribute']].append(highest_index)
                    highest_index += 1
                value = [
                    key
                    for key, (attr, value) in column_values.items()
                    if value == item['value'] and attr == item['attribute']
                ][0]
                ar_dict['undesired']['itemset'].append(value)
                ar_dict['desired']['itemset'].append(value)

            for item in rule['flexible']:
                if item['attribute'] not in flexible_items_binding.keys():
                    flexible_items_binding.update({item['attribute']: []})
                if (item['attribute'], item['undesired']) not in column_values.values():
                    column_values[highest_index] = (item['attribute'], item['undesired'])
                    flexible_items_binding[item['attribute']].append(highest_index)
                    highest_index += 1
                if (item['attribute'], item['desired']) not in column_values.values():
                    column_values[highest_index] = (item['attribute'], item['desired'])
                    flexible_items_binding[item['attribute']].append(highest_index)
                    highest_index += 1
                value_0 = [
                    key
                    for key, (attr, value) in column_values.items()
                    if value == item['undesired'] and attr == item['attribute']
                ][0]
                value_1 = [
                    key
                    for key, (attr, value) in column_values.items()
                    if value == item['desired'] and attr == item['attribute']
                ][0]
                ar_dict['undesired']['itemset'].append(value_0)
                ar_dict['desired']['itemset'].append(value_1)

            action_rules.append(ar_dict)

        return Output(action_rules, target, stable_items_binding, flexible_items_binding, column_values)
