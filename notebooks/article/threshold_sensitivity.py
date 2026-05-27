"""Threshold sensitivity sweep for the article.

Replicates the rule-categorisation analysis (Table 5) on each of the four
benchmark datasets under three threshold configurations:

- ``base`` — the calibrated thresholds used in the main results (Table 5).
- ``tight`` — higher per-class support thresholds, yielding fewer rules.
- ``loose`` — lower per-class support thresholds, yielding more rules.

For each (dataset, configuration) we mine action rules with the same
hyperparameters as ``05_categorization_breakdown.ipynb`` (bootstrap
percentile CIs at 95\,\%, B=500, seed=42), then categorise rules at
threshold = 0 separately for the uplift CI and the realistic rule gain
CI.  Results are persisted under
``article/results/threshold_sensitivity.csv``.

This script is the canonical source for the *Threshold sensitivity*
appendix referenced from the Limitations section of ``sn-article.tex``.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd


def _add_repo_to_path() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / 'pyproject.toml').exists():
            sys.path.insert(0, str(parent / 'src'))
            sys.path.insert(0, str(parent / 'notebooks' / 'article'))
            return parent
    raise FileNotFoundError('repository root with pyproject.toml not found')


REPO = _add_repo_to_path()

# pylint: disable=wrong-import-position
from _datasets import ALL_LOADERS, DatasetConfig  # type: ignore  # noqa: E402

from action_rules import ActionRules  # noqa: E402


# Match 05_categorization_breakdown.ipynb defaults so the base config in this
# sweep is identical to the values that drive Table 5.
N_BOOTSTRAP = 500
SEED = 42
THRESHOLD = 0.0


_PERTURBATIONS = {
    'Telco Customer Churn': {
        'tight': {'min_undesired_support': 270, 'min_desired_support': 135},
        'loose': {'min_undesired_support': 170, 'min_desired_support': 85},
    },
    'UCI Bank Marketing': {
        'tight': {'min_undesired_support': 220, 'min_desired_support': 220},
        'loose': {'min_undesired_support': 150, 'min_desired_support': 150},
    },
    'IBM Employee Attrition': {
        'tight': {'min_undesired_support': 22, 'min_desired_support': 22},
        'loose': {'min_undesired_support': 14, 'min_desired_support': 14},
    },
    'Taiwan Credit Card Default': {
        'tight': {'min_undesired_support': 575, 'min_desired_support': 575},
        'loose': {'min_undesired_support': 350, 'min_desired_support': 350},
    },
}


def _mine_and_categorise(cfg: DatasetConfig) -> dict:
    ar = ActionRules(
        min_stable_attributes=cfg.min_stable_attributes,
        min_flexible_attributes=cfg.min_flexible_attributes,
        min_undesired_support=cfg.min_undesired_support,
        min_desired_support=cfg.min_desired_support,
        min_undesired_confidence=cfg.min_undesired_confidence,
        min_desired_confidence=cfg.min_desired_confidence,
        intrinsic_utility_table=cfg.intrinsic_utility_table,
        transition_utility_table=cfg.transition_utility_table,
    )
    ar.fit(
        cfg.df,
        stable_attributes=cfg.stable_attributes,
        flexible_attributes=cfg.flexible_attributes,
        target=cfg.target,
        target_undesired_state=cfg.target_undesired_state,
        target_desired_state=cfg.target_desired_state,
        use_sparse_matrix=cfg.use_sparse_matrix,
    )
    n_rules = len(ar.output.action_rules) if ar.output is not None else 0
    if n_rules == 0:
        return {
            'n_rules': 0,
            'uplift_accept': 0, 'uplift_uncertain': 0, 'uplift_reject': 0,
            'gain_accept': 0, 'gain_uncertain': 0, 'gain_reject': 0,
        }

    counts: dict[str, Counter] = {}
    for metric in ('uplift', 'realistic_rule_gain'):
        results = ar.confidence_intervals(
            cfg.df,
            method='bootstrap',
            confidence_level=0.95,
            threshold=THRESHOLD,
            metric=metric,
            n_bootstrap=N_BOOTSTRAP,
            random_state=SEED,
        )
        counts[metric] = Counter(
            r.category.value for r in results if r.category is not None
        )

    return {
        'n_rules': n_rules,
        'uplift_accept': counts['uplift'].get('accept', 0),
        'uplift_uncertain': counts['uplift'].get('uncertain', 0),
        'uplift_reject': counts['uplift'].get('reject', 0),
        'gain_accept': counts['realistic_rule_gain'].get('accept', 0),
        'gain_uncertain': counts['realistic_rule_gain'].get('uncertain', 0),
        'gain_reject': counts['realistic_rule_gain'].get('reject', 0),
    }


def main() -> None:
    warnings.filterwarnings('ignore')

    rows: list[dict] = []
    for ds_name, loader in ALL_LOADERS.items():
        print(f'=== {ds_name} ===')
        base_cfg = loader()
        configs = {'base': base_cfg}
        for label, overrides in _PERTURBATIONS[ds_name].items():
            configs[label] = replace(base_cfg, **overrides)

        for cfg_label, cfg in configs.items():
            print(
                f'  [{cfg_label}] support={cfg.min_undesired_support}/'
                f'{cfg.min_desired_support} conf={cfg.min_undesired_confidence}'
            )
            res = _mine_and_categorise(cfg)
            res.update(
                {
                    'dataset': ds_name,
                    'config': cfg_label,
                    'min_undesired_support': cfg.min_undesired_support,
                    'min_desired_support': cfg.min_desired_support,
                    'min_undesired_confidence': cfg.min_undesired_confidence,
                    'min_desired_confidence': cfg.min_desired_confidence,
                }
            )
            rows.append(res)
            print(
                f'    -> n={res["n_rules"]}, uplift A/U/R='
                f'{res["uplift_accept"]}/{res["uplift_uncertain"]}/{res["uplift_reject"]},'
                f' gain A/U/R={res["gain_accept"]}/{res["gain_uncertain"]}/{res["gain_reject"]}'
            )

    df = pd.DataFrame(rows)
    cols = [
        'dataset', 'config',
        'min_undesired_support', 'min_desired_support',
        'min_undesired_confidence', 'min_desired_confidence',
        'n_rules',
        'uplift_accept', 'uplift_uncertain', 'uplift_reject',
        'gain_accept', 'gain_uncertain', 'gain_reject',
    ]
    df = df[cols]

    out_csv = REPO / 'article' / 'results' / 'threshold_sensitivity.csv'
    df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv.relative_to(REPO)}')


if __name__ == '__main__':
    main()
