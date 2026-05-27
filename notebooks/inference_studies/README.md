# Inference and validation studies

Replicability notebooks for the action-rules confidence-interval and
cross-validation workflows. Each notebook is self-contained: load datasets via
`notebooks/article/_datasets.py`, mine rules, compute statistics, persist a
CSV under `results/`.

## Notebooks (run in any order)

| Notebook | Topic |
| --- | --- |
| `01_rule_level_confidence_intervals.ipynb` | Per-rule 95% CIs from all five CI methods (bootstrap percentile/BCa, Wald, Wilson, Bayesian) across four datasets. |
| `02_coverage_simulation.ipynb` | Empirical coverage of each CI method on a synthetic DGP with known true uplift. |
| `03_runtime_benchmark.ipynb` | Wall-clock cost of every CI method on every dataset. |
| `04_rule_categorization.ipynb` | Accept / Reject / Uncertain breakdown at threshold = 0 for uplift and realistic rule gain. |
| `05_cross_validation.ipynb` | Stratified K-fold CV with four targeting strategies and three reliability views (in-sample / fold range / CV aggregate with cluster bootstrap CI). |
| `06_threshold_sensitivity.ipynb` | Sensitivity of the categorisation to support-threshold perturbations. |
| `07_utility_table_sensitivity.ipynb` | Sensitivity of the realistic-gain categorisation to a uniform cost-side rescaling of the utility tables. |

## Inputs

- Four benchmark datasets, loaded from `notebooks/data/telco.csv` and
  `notebooks/ci/data/*.csv`. All preprocessing is centralised in
  `notebooks/article/_datasets.py`.
- A synthetic data-generating process from `tests/simulation/coverage_simulation.py`.

## Outputs

Every notebook writes a CSV (and occasionally an NPZ) into
`notebooks/inference_studies/results/`. Outputs are byte-stable across machines
when the random seed (`SEED = 42`) is kept.

## Dependencies

```bash
pip install action-rules[viz,inference]
```

(`scipy` for analytic / Bayesian intervals, `matplotlib` for the optional
plotting cell at the end of `02_coverage_simulation.ipynb`.)

## Runtime budget

End-to-end, all seven notebooks complete in roughly 10–15 minutes on a modern
laptop at the article-grade parameter defaults.
