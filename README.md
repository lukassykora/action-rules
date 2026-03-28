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

For command-line interface (CLI) usage, the action-rules package must be installed using pipx:
``` console
$ pipx install action-rules
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

### Confidence Intervals

Compute confidence intervals for uplift and realistic rule gain using one of three methods: **bootstrap**, **analytic** (Wald), or **Bayesian**.

```console
$ pip install action-rules[inference]   # adds scipy
$ pip install action-rules[viz]         # adds matplotlib + scipy
```

```python
from action_rules import ActionRules

# After fitting action rules...
action_rules = ActionRules(
    min_stable_attributes=2, min_flexible_attributes=1,
    min_undesired_support=1, min_undesired_confidence=0.5,
    min_desired_support=1, min_desired_confidence=0.5,
)
action_rules.fit(data, stable_attributes, flexible_attributes, target, '0', '1')

# Compute bootstrap confidence intervals
results = action_rules.confidence_intervals(
    data,
    method="bootstrap",      # "bootstrap", "analytic", "wald", or "bayesian"
    confidence_level=0.95,
    threshold=0.0,           # categorize rules as Accept/Reject/Uncertain
    n_bootstrap=1000,        # bootstrap resamples (bootstrap only)
    random_state=42,
)

# Each result contains: uplift_point, uplift_ci_lower, uplift_ci_upper, uplift_se,
# realistic_rule_gain_point/ci_lower/ci_upper/se (if utility tables provided),
# category (RuleCategory.ACCEPT / REJECT / UNCERTAIN)
for r in results:
    print(f"Rule {r.rule_index}: uplift = {r.uplift_point:.4f} "
          f"[{r.uplift_ci_lower:.4f}, {r.uplift_ci_upper:.4f}] → {r.category.value}")

# JSON export now includes CI data
print(action_rules.get_rules().get_export_notation())
```

#### Visualization

```python
from action_rules.visualization import bootstrap_histogram, forest_plot, grouped_forest_plot

# Single-rule distribution (bootstrap or Bayesian)
fig = bootstrap_histogram(results[0], metric="uplift", threshold=0.0)
fig.savefig("distribution.png")

# Forest plot: all rules with CI bars
fig = forest_plot(results, metric="uplift", threshold=0.0)
fig.savefig("forest.png")

# Compare methods side-by-side
results_boot = action_rules.confidence_intervals(data, method="bootstrap", random_state=42)
results_anal = action_rules.confidence_intervals(data, method="analytic")
results_bayes = action_rules.confidence_intervals(data, method="bayesian", random_state=42)

fig = grouped_forest_plot(
    {"bootstrap": results_boot, "analytic": results_anal, "bayesian": results_bayes},
    metric="uplift",
    threshold=0.0,
)
fig.savefig("comparison.png")
```

#### CLI with Confidence Intervals

``` console
$ action-rules --min_stable_attributes 2 --min_flexible_attributes 1 \
    --min_undesired_support 1 --min_undesired_confidence 0.5 \
    --min_desired_support 1 --min_desired_confidence 0.5 \
    --csv_path data.csv --stable_attributes 'Sex, Age' \
    --flexible_attributes 'Class, Embarked' --target Survived \
    --undesired_state 0 --desired_state 1 \
    --ci_method bootstrap --confidence_level 0.95 --n_bootstrap 1000 \
    --ci_threshold 0.0 --random_state 42 \
    --output_json_path output.json
```

Available CI options:
- `--ci_method` — `bootstrap`, `analytic`, `wald`, or `bayesian`
- `--confidence_level` — confidence level (default: 0.95)
- `--ci_threshold` — threshold for Accept/Reject/Uncertain categorization
- `--n_bootstrap` — number of bootstrap resamples (default: 1000)
- `--n_mc` — number of Monte Carlo draws for Bayesian (default: 10000)
- `--random_state` — random seed for reproducibility

## Jupyter Notebook Examples

* [Confidence Intervals (Bootstrap, Analytic, Bayesian)](https://github.com/lukassykora/action-rules/blob/main/notebooks/ConfidenceIntervals.ipynb)
* [Titanic Dataset (GPU accelerated)](https://github.com/lukassykora/action-rules/blob/main/notebooks/Example.ipynb)
* [Customer Churn (easy workflow)](https://github.com/lukassykora/action-rules/blob/main/notebooks/ExampleCustomerChurn.ipynb)
* [High-Utility Action Rules Mining Example](https://github.com/lukassykora/action-rules/blob/main/notebooks/Utility.ipynb)

## Performance

* [Customer Churn (GPU vs. CPU, sparse matrix vs. dense matrix, Pandas vs. cuDF)](https://github.com/lukassykora/action-rules/blob/main/notebooks/Performance.ipynb)
* [Scalene Profiling](https://github.com/lukassykora/action-rules/blob/main/notebooks/profiling/plot.ipynb)
* [GPU Memory Usage - Sparse vs. Dense Matrix](https://github.com/lukassykora/action-rules/blob/main/notebooks/gpu_sparse_vs_dense/process_logs.ipynb)
* [CPU Usage](https://github.com/lukassykora/action-rules/blob/main/notebooks/cpu_cores/cpu_usage.ipynb)
* [Compare the Action-Rules package with ActionRulesDiscovery package](https://github.com/lukassykora/action-rules/blob/main/notebooks/Comparison.ipynb) - link to ActionRulesDiscovery package: <https://github.com/lukassykora/actionrules>

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and
the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
