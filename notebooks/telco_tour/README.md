# Telco hands-on tour

A two-notebook walkthrough of the action-rules workflow on the Telco Customer
Churn dataset (7,043 customers, `notebooks/data/telco.csv`).

## Notebooks (run in order)

| Notebook | Topic |
| --- | --- |
| `01_end_to_end_telco.ipynb` | Load data, define utility tables, fit action rules, compute analytic confidence intervals, export to JSON. |
| `02_visual_diagnostics.ipynb` | Four publication-grade diagnostic figures (churn-rate panel, bootstrap distribution + analytic overlay, forest plot with category markers, bootstrap-vs-analytic CI-width scatter). |

## Inputs

- `notebooks/data/telco.csv` (ships with the repository, semicolon-separated).

## Outputs

Under `notebooks/telco_tour/outputs/`:

- `telco_rules.json` — exported rule set (round-trippable via
  `action_rules.input.Input`).
- `fig1_churn_rate_panel.pdf`
- `fig2_bootstrap_hist.pdf`
- `fig3_forest_plot.pdf`
- `fig4_bootstrap_vs_analytic.pdf`

## Dependencies

```bash
pip install action-rules[viz,inference]
```

`matplotlib` is required only for the visual diagnostics notebook; the
end-to-end notebook is matplotlib-free.

## Runtime budget

Both notebooks together complete in well under one minute on a modern laptop.
