"""Utility-table sensitivity sweep for the article.

The headline economic-rejection finding (Table~5) depends on the chosen
intrinsic and transition utility tables.  This sweep stresses how the
realistic-rule-gain categorisation reacts when the *cost* side of the
utility model is scaled uniformly: every negative entry in the intrinsic
utility table and every negative entry in the transition utility table
is multiplied by a factor ``s``.  Positive entries (in particular the
target benefit assigned to the desired class) are left unchanged so that
``s`` represents a pure cost-to-benefit ratio shift, not an arbitrary
rescaling of the whole gain.

Three configurations are reported for every dataset:

- ``cost_x0.5`` — intervention costs are half the baseline.
- ``base`` — utility tables as published.
- ``cost_x1.5`` — intervention costs are 50\,\% higher than the baseline.

For every (dataset, configuration) we re-mine the action rules using the
same support/confidence thresholds as the main results, recompute
bootstrap percentile 95\,\% CIs (B=500, seed=42), and categorise rules at
:math:`\tau_G = 0` for the realistic rule gain CI.  The uplift CI is
invariant to the utility tables so we report it only at the base
configuration.

Output: ``article/results/utility_table_sensitivity.csv``.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Mapping

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


N_BOOTSTRAP = 500
SEED = 42
THRESHOLD = 0.0


# A configuration is identified by a label and a cost-scaling factor.
_SCALINGS: dict[str, float] = {
    'cost_x0.5': 0.5,
    'base': 1.0,
    'cost_x1.5': 1.5,
}


def _scale_negatives(table: Mapping | None, factor: float) -> dict | None:
    """Return a shallow copy of *table* with every negative value scaled by *factor*."""
    if table is None:
        return None
    return {k: (v * factor if v < 0 else v) for k, v in table.items()}


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
        for cfg_label, factor in _SCALINGS.items():
            scaled = replace(
                base_cfg,
                intrinsic_utility_table=_scale_negatives(base_cfg.intrinsic_utility_table, factor),
                transition_utility_table=_scale_negatives(base_cfg.transition_utility_table, factor),
            )
            print(f'  [{cfg_label}] cost scaling factor={factor}')
            res = _mine_and_categorise(scaled)
            res.update(
                {
                    'dataset': ds_name,
                    'config': cfg_label,
                    'cost_scaling': factor,
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
        'dataset', 'config', 'cost_scaling',
        'n_rules',
        'uplift_accept', 'uplift_uncertain', 'uplift_reject',
        'gain_accept', 'gain_uncertain', 'gain_reject',
    ]
    df = df[cols]

    out_csv = REPO / 'article' / 'results' / 'utility_table_sensitivity.csv'
    df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv.relative_to(REPO)}')


if __name__ == '__main__':
    main()
