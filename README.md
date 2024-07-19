# Action Rules

[![pypi](https://img.shields.io/pypi/v/action-rules.svg)](https://pypi.org/project/action-rules/)
[![python](https://img.shields.io/pypi/pyversions/action-rules.svg)](https://pypi.org/project/action-rules/)
[![Build Status](https://github.com/lukassykora/action-rules/actions/workflows/dev.yml/badge.svg)](https://github.com/lukassykora/action-rules/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/lukassykora/action-rules/branch/main/graphs/badge.svg)](https://codecov.io/github/lukassykora/action-rules)

The package for action rules mining using Action-Apriori (Apriori Modified for Action Rules Mining).

* Documentation: <https://lukassykora.github.io/action-rules>
* GitHub: <https://github.com/lukassykora/action-rules>
* PyPI: <https://pypi.org/project/action-rules/>
* Free software: MIT

## Installation

``` console
$ pip install action-rules
```

## Features

### Action Rules API

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
min_undesired_confidence = 0.5  # min 0.5
min_desired_support = 1
min_desired_confidence = 0.5  # min 0.5
undesired_state = '0'
desired_state = '1'
# Action Rules Mining
action_rules = ActionRules(
    min_stable_attributes=min_stable_attributes,
    min_flexible_attributes=min_flexible_attributes,
    min_undesired_support=min_undesired_support,
    min_undesired_confidence=min_undesired_confidence,
    min_desired_support=min_desired_support,
    min_desired_confidence=min_desired_confidence,
    verbose=True
)
# Fit
action_rules.fit(
    data=data,  # cuDF or Pandas Dataframe
    stable_attributes=stable_attributes,
    flexible_attributes=flexible_attributes,
    target=target,
    target_undesired_state=undesired_state,
    target_desired_state=desired_state,
    use_sparse_matrix=True,  # needs SciPy or Cupyx (if use_gpu is True) installed
    use_gpu=False,  # needs Cupy installed
)
# Print rules
# Example: [(Age: O) ∧ (Sex: M) ∧ (Embarked: S → C)] ⇒ [Survived: 0 → 1], support of undesired part: 1, confidence of undesired part: 1.0, support of desired part: 1, confidence of desired part: 1.0, uplift: 1.0
for action_rule in action_rules.get_rules().get_ar_notation():
    print(action_rule)
# Print rules (pretty notation)
# Example: If attribute 'Age' is 'O', attribute 'Sex' is 'M', attribute 'Embarked' value 'S' is changed to 'C', then 'Survived' value '0' is changed to '1 with uplift: 1.0.
for action_rule in action_rules.get_rules().get_pretty_ar_notation():
    print(action_rule)
# JSON export
print(action_rules.get_rules().get_export_notation())
```

### Action Rules CLI

``` console
$ action-rules --min_stable_attributes 2 --min_flexible_attributes 1 --min_undesired_support 1 --min_undesired_confidence 0.5 --min_desired_support 1 --min_desired_confidence 0.5 --csv_path 'data.csv' --stable_attributes 'Sex, Age' --flexible_attributes 'Class, Embarked' --target 'Survived' --undesired_state '0' --desired_state '1' --output_json_path 'output.json'
```

## Jupyter Notebook Example

<https://github.com/lukassykora/action-rules/blob/main/notebooks/Example.ipynb>

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and
the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
