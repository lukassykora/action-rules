# Usage

To use Action Rules in a project

```python
# Import Module
from action_rules import ActionRules
import pandas as pd

# Get Data
transactions = {'Sex': ['M', 'F', 'M', 'M', 'F', 'M', 'F'],
                'Age': ['Y', 'Y', 'O', 'Y', 'Y', 'O', 'Y'],
                'Class': [1, 1, 2, 2, 1, 1, 2],
                'Embarked': ['S', 'C', 'S', 'C', 'S', 'C', 'C'],
                'Survived': [1, 1, 0, 0, 1, 1, 0],
                }
data = pd.DataFrame.from_dict(transactions)
# Initialize ActionRules Miner with Parameters
stable_attributes = ['Age', 'Sex']
flexible_attributes = ['Embarked', 'Class']
target = 'Survived'
min_stable_attributes = 2
min_flexible_attributes = 1  # min 1
min_undesired_support = 1
min_undesired_confidence = 0.5
min_desired_support = 1
min_desired_confidence = 0.5
undesired_state = '0'
desired_state = '1'
# Action Rules Mining
action_rules = ActionRules(min_stable_attributes, min_flexible_attributes, min_undesired_support,
                           min_undesired_confidence, min_desired_support, min_desired_confidence, verbose=False)
# Fit
action_rules.fit(
    data,
    stable_attributes,
    flexible_attributes,
    target,
    undesired_state,
    desired_state,
)
# Print rules
for action_rule in action_rules.get_rules().get_ar_notation():
    print(action_rule)
# Print rules (pretty notation)
for action_rule in action_rules.get_rules().get_pretty_ar_notation():
    print(action_rule)
# JSON export
print(action_rules.get_rules().get_export_notation())
```

## Confidence intervals

After fitting, `confidence_intervals()` attaches an interval to every rule. All methods and options
(`analytic_type`, `bootstrap_type`, `resample_mode`, `prior`, `simultaneous_m`, `discovery_fraction`,
`estimation_data`) are described in the README section *Confidence Intervals* and in the API reference.

```python
# Bootstrap with a Bonferroni adjustment over all reported rules
n_rules = len(action_rules.get_rules().action_rules)
results = action_rules.confidence_intervals(data, method="bootstrap", simultaneous_m=n_rules, random_state=42)

# Bayesian intervals with the Jeffreys prior
results = action_rules.confidence_intervals(data, method="bayesian", prior="jeffreys", random_state=42)

# Honest intervals: rules re-mined on a discovery split, intervals on the disjoint estimation split
results = action_rules.confidence_intervals(data, method="honest_split", discovery_fraction=0.5, random_state=42)
```
