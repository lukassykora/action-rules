# Changelog

## [0.0.1] - 2024-05-16

* First release on PyPI.

## [0.0.2] - 2024-05-16

* Fixed installation issues.

## [0.0.3] - 2024-05-16

* Working release on PyPI.

## [0.0.4] - 2024-06-10

* New repository.
* Tests.

## [0.0.5] - 2024-06-10

* Readme.
* GitHub Actions.

## [0.0.6] - 2024-06-10

* Fix Codecov.

## [0.0.7] - 2024-06-11

* Fix Codecov for GitHub Actions.

## [0.0.8] - 2024-06-11

* Publish documentation.

## [0.0.9] - 2024-06-11

* Publish documentation - next try.

## [0.0.10] - 2024-06-11

* Publish documentation - next try.

## [0.0.11] - 2024-06-11

* Publish documentation - next try.

## [0.0.12] - 2024-06-11

* Publish documentation - next try.

## [0.0.13] - 2024-06-11

* Publish documentation - next try.

## [0.0.14] - 2024-06-11

* Publish documentation - next try.

## [0.0.15] - 2024-06-12

* Publish documentation - next try.

## [0.0.16] - 2024-06-12

* Publish documentation - next try.

## [0.0.17] - 2024-06-12

* Publish documentation - next try.

## [0.0.18] - 2024-06-12

* Publish documentation - next try.

## [0.0.19] - 2024-06-12

* Publish documentation - next try.

## [0.0.20] - 2024-06-12

* Publish documentation - next try.

## [0.0.21] - 2024-06-12

* Numpy style documentation.

## [0.0.22] - 2024-06-12

* Import rules.

## [0.0.23] - 2024-07-12

* GPU Acceleration.

## [0.0.24] - 2024-07-15

* cuDF improved import.

## [0.0.25] - 2024-07-17

* Clear GPU memory pool.

## [0.0.26] - 2024-07-17

* Eliminate zeros for sparse matrix.

## [0.0.27] - 2024-07-19

* Readme improved.

## [0.0.28] - 2024-07-23

* Predict function.

## [0.0.29] - 2024-07-23

* Improved pretty notation.

## [1.0.0] - 2024-07-27

* Uplift fixed.
* Docstrings and notations.
* Completely finished.

## [1.0.1] - 2024-07-31

* Uplift fixed.

## [1.0.2] - 2024-07-31

* Profiling.

## [1.0.3] - 2024-07-31

* Comparison with another package.

## [1.0.4] - 2024-08-04

* Fix predict method for cupy

## [1.0.5] - 2024-08-05

* Fix empty list of stable attributes

## [1.0.6] - 2024-08-06

* Feature: New fit_onehot

## [1.0.7] - 2024-08-06

* Fix: Onehot data copy

## [1.0.8] - 2025-01-01

* Feature: Dominant action rules

## [1.0.9] - 2025-02-23

* Feature: High-Utility Action Rules Mining

## [1.0.10] - 2025-03-01

* Feature: Measures - support and confidence

## [1.0.11] - 2025-03-04

* Fix: Utility

## [1.1.0] - 2026-05-27

* Feature: Confidence intervals for action rules via `ActionRules.confidence_intervals()` with three engines — bootstrap (percentile, BCa), analytic (Wald, Newcombe-Wilson, auto), and Bayesian (Beta-Binomial Monte Carlo).
* Feature: Rule categorisation (`accept` / `reject` / `uncertain`) based on a user-supplied decision threshold over `uplift` or `realistic_rule_gain`.
* Feature: Cross-validation via `ActionRules.cross_validate()` with stratified folds, configurable selection strategies, and out-of-fold metric estimation.
* Feature: New `action_rules.visualization` module with forest plots, bootstrap distributions, coverage calibration, and CI-width diagnostics (matplotlib lazy-imported).
* Feature: CI results are surfaced in `Output.get_ar_notation()`, `Output.get_pretty_ar_notation()`, and JSON export (`get_export_notation()`); NaN/Inf are serialised as `null`.
* Feature: New CLI options for confidence intervals (`--ci-method`, `--ci-level`, `--ci-threshold`, `--ci-metric`, …).
* Fix: `df_to_array` no longer creates spurious `<attr>_<item_*>_nan` one-hot columns for missing antecedent values; `NaN` is now preserved through `get_dummies` per the pessimistic null-value semantics of Dardzinska (2013, §2.3.2). Target column behaviour unchanged.
* Docs: Notebooks for the Telco churn end-to-end tour, rule-level CIs across three datasets (Bank Marketing, Credit Card Default, Employee Attrition), and the inference studies / article figures.

## [2.0.0] - 2026-06-06


* Feature: Packed-bitset rewrite of the Action-Apriori mining core. CPU and GPU CUDA path. 
* Compatibility: Public API preserved so existing 1.x call sites keep working unchanged.

## [2.0.1] - 2026-08-10

* Docs: Removed the stale `# min 0.5` comments from the README and usage docs — the confidence thresholds have no enforced lower bound.
* Tests: Added a regression test covering mining with confidence thresholds below 0.5.
* Examples: The Telco utility tables now price every flexible attribute state (InternetService and PhoneService margins, explicit `No internet service` levels), so cancelling a service line is no longer treated as free; the `telco_tour` notebooks and outputs were regenerated (seeded, reproducible).

## [2.1.0] - 2026-10-26

* Feature: Random-design delta-method Wald confidence intervals (`analytic_type='wald_rd'` / `method='wald_rd'`) accounting for random antecedent prevalence alongside conditional rule confidences.
* Feature: Exact Clopper-Pearson single-proportion intervals combined via Newcombe's square-and-add formula (`analytic_type='clopper_pearson'`) and Bonferroni simultaneous multiplicity adjustment (`simultaneous_m`).
* Feature: Named conjugate priors (`prior='jeffreys'`, `'uniform'`, `'haldane'`) in `BayesianEngine` and `method='jeffreys'` shortcut in `ActionRules.confidence_intervals()`.
* Feature: Honest two-stage sample-splitting inference (`method='honest_split'`) and fast vectorised 5-cell multinomial bootstrap resampling (`resample_mode='multinomial'`).
* Feature: Added `RuleCategory.UNSUPPORTED` and `ConfidenceIntervalResult.decision_status` for four-way rule categorisation (`accept`, `reject`, `uncertain`, `unsupported`) when antecedent support is zero or CI bounds are `NaN`.
* Fix: Guarded `Rules.calculate_uplift` against zero division when `undesired_confidence == 0` or `count_transactions == 0`, and prevented self-pairing of identical classification rule itemsets in `Rules.generate_action_rules` when mining with low confidence thresholds (`< 0.5`).
* Notebooks: Added automatic dataset download and caching fallbacks in `notebooks/article/_datasets.py` for UCI Bank Marketing, IBM Employee Attrition, and Taiwan Credit Card Default.

## [2.1.1] - 2026-10-26

* Fix: `method='honest_split'` is now a genuine two-stage procedure. The rows are split into disjoint discovery and estimation parts before mining; rules are re-mined on the discovery part only (support thresholds scaled by its share of the rows) and their intervals are computed on the estimation part. In 2.1.0 the rules were selected on the full data and only the intervals were recomputed on a random half, which left the selection bias in place. The fitted model is replaced by the discovery-part model and the split is exposed as `ActionRules.honest_split_indices`; a pre-split holdout can be passed as `estimation_data`.
* Fix: `honest_split` respects `analytic_type` (an explicit `'wald'` is no longer switched to `'newcombe_wilson'`).
* Fix: `simultaneous_m` (Bonferroni) now applies to every method — bootstrap, BCa, Bayesian and Jeffreys as well as the analytic intervals. `ConfidenceIntervalResult.confidence_level` keeps reporting the nominal family-wise level.
* Fix: `confidence_intervals()` validates method-specific options instead of silently ignoring them: an unknown `resample_mode`, `resample_mode != 'rows'` with a non-bootstrap method, `prior` with a non-Bayesian method, and `discovery_fraction` / `estimation_data` without `honest_split` raise `ValueError`.
* Fix: `prior='haldane'` is the exact Beta(0, 0) prior; a zero posterior parameter is floored at `POSTERIOR_EPSILON`. Negative prior parameters raise `ValueError`.
* Fix: The multinomial bootstrap falls back to row resampling for rules whose antecedents overlap, where the 5-cell model does not apply.
* Docs: Documented all v2.1 methods and options in the `confidence_intervals()` docstring, README and usage docs; the CLI gains `--prior`, `--simultaneous_m`, `--resample_mode`, `--discovery_fraction` and the new `--ci_method` / `--analytic_type` values.
* CI: Replaced the removed `pymdownx.slugs.uslugify` in `mkdocs.yml`, capped `pymdown-extensions < 12` in the `doc` extra (version 12 breaks `mkdocstrings`), the deprecated `::set-output` and `skip_existing` in the release workflows, and split `BootstrapEngine.compute` to satisfy the flake8 complexity limit.
