"""Cross-validation for action-rule discovery and targeting.

The :class:`CrossValidator` orchestrates K-fold cross-validation around the
``ActionRules.fit`` pipeline.  For every fold it:

1. Splits the dataset (stratified by target value by default).
2. Mines action rules on the *train* fold via a fresh ``ActionRules`` instance.
3. Computes confidence intervals on the *train* fold (used as ranking
   scores).
4. Re-scores each discovered rule on the *test* fold (``test_uplift``,
   ``test_realistic_rule_gain``, ``test_support``).
5. Applies a set of *targeting strategies* (point estimate, lower CI bound,
   ``lower>0`` filter, risk-adjusted score) and computes targeting metrics
   (``uplift@k``, Qini, AUUC, profit@k) on the test fold.

Across folds the result reports both:

- **stability**: mean ± std of every (strategy, metric) pair, used as a
  qualitative summary of fold-to-fold variance, **not** as a formal 95 %
  CI (the latter is unreliable for K-fold CV — see Bates, Hastie &
  Tibshirani, "Cross-Validation: What does it estimate and how well does
  it do it?", arXiv:2104.00673, 2021).
- **bootstrap CI**: a *cluster bootstrap* by fold (the default) — for each
  bootstrap replicate, rule records are resampled with replacement within
  each fold, the targeting metric is computed per fold, then averaged
  across folds.  The resulting interval estimates the same fold-mean
  quantity as the ``mean`` column.  A legacy ``oof_pool`` design (resample
  within fold then concatenate into one pool before computing the metric)
  is also available; it estimates a *pool-level* statistic that differs
  from the fold mean by roughly a factor of K, so ``mean`` and ``[CI]``
  will not align under that design — kept only for backwards
  compatibility.

Utility tables are supported: when present, every metric is also reported
for the realistic rule gain in addition to plain uplift.
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..inference.base import (
    compute_group_counts,
    compute_realistic_gain,
    compute_uplift_from_counts,
    extract_rule_masks,
)
from . import metrics as M

# Recognised strategy and metric names.  Kept module-level for easy
# import/test access and to drive validation in :func:`cross_validate`.
STRATEGIES = ('point', 'lower', 'lower_positive', 'risk_adjusted')
METRICS = ('uplift_at_k', 'qini', 'auuc', 'profit_at_k')


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CrossValidationResult:
    r"""Container for the output of :class:`CrossValidator.run`.

    Attributes
    ----------
    n_splits : int
        Number of folds used.
    strategies : tuple of str
        Targeting strategies evaluated.
    metrics : tuple of str
        Metric names evaluated.
    k_fraction : float
        Top-k cutoff used by the ``*_at_k`` metrics.
    confidence_level : float
        Nominal confidence level used for CI computations.
    has_utility : bool
        ``True`` when utility tables were supplied and gain-aware metrics
        were computed.
    rule_records : pd.DataFrame
        One row per (fold, rule discovered in that fold).  Columns:
        ``fold``, ``rule_index_in_fold``, ``support_train``, ``support_test``,
        ``train_uplift``, ``train_uplift_lower``, ``train_uplift_upper``,
        ``train_uplift_se``, ``test_uplift``, and the ``*_gain`` analogues
        when utility was supplied.
    fold_summary : pd.DataFrame
        One row per (fold, strategy, metric), with the metric value on
        that fold's test set.
    strategy_summary : pd.DataFrame
        One row per (strategy, metric) with ``mean`` and ``std`` over folds
        (stability), and ``ci_lower`` / ``ci_upper`` columns when bootstrap
        CIs were computed.
    insample_summary : pd.DataFrame, optional
        One row per (strategy, metric, metric_target) with the metric value
        computed by mining and scoring on the same full dataset (apparent
        / in-sample performance, optimistic by construction).  Populated
        only when ``compute_insample_baseline=True`` is passed; ``None``
        otherwise so existing consumers see no schema change.  Used by the
        article to quantify the optimism gap between in-sample and OOF
        performance (Hastie, Tibshirani \\& Friedman, *Elements of
        Statistical Learning*, Ch. 7).
    rule_stability : pd.DataFrame, optional
        Pairwise Jaccard overlap of discovered rule signatures across folds
        (one row per ordered fold pair).  Populated when ``track_stability``
        is enabled.
    n_rules_per_fold : list of int
        Convenience list of rule counts per fold.
    """

    n_splits: int
    strategies: Tuple[str, ...]
    metrics: Tuple[str, ...]
    k_fraction: float
    confidence_level: float
    has_utility: bool
    rule_records: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    insample_summary: Optional[pd.DataFrame] = None
    rule_stability: Optional[pd.DataFrame] = None
    n_rules_per_fold: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def stratified_kfold_indices(
    y: Sequence,
    n_splits: int,
    random_state: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) tuples for stratified K-fold splitting.

    Hand-rolled to avoid a hard dependency on scikit-learn.  Within each
    class the indices are shuffled with a NumPy generator seeded by
    *random_state* and then evenly divided into ``n_splits`` chunks.

    Parameters
    ----------
    y : sequence
        Stratification labels (any hashable values).
    n_splits : int
        Number of folds; must be at least 2.
    random_state : int, optional
        Seed for the NumPy generator.  ``None`` uses fresh entropy.

    Returns
    -------
    list of (ndarray, ndarray)
        ``[(train_idx_0, test_idx_0), ..., (train_idx_{K-1}, test_idx_{K-1})]``
        with each index array sorted ascending.

    Raises
    ------
    ValueError
        When *n_splits* is less than 2 or any class has fewer than
        *n_splits* members.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")

    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(list(y))
    unique = np.unique(y_arr)
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for c in unique:
        class_idx = np.where(y_arr == c)[0]
        if class_idx.shape[0] < n_splits:
            raise ValueError(
                f"Stratified K-fold needs at least {n_splits} members per class; "
                f"class {c!r} has only {class_idx.shape[0]}."
            )
        rng.shuffle(class_idx)
        chunks = np.array_split(class_idx, n_splits)
        for i, chunk in enumerate(chunks):
            folds[i].extend(chunk.tolist())

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(y_arr.shape[0])
    for i in range(n_splits):
        test_idx = np.array(sorted(folds[i]), dtype=np.intp)
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
        splits.append((train_idx, test_idx))
    return splits


def _rule_signature(rule_masks) -> Tuple:
    """Return a hashable identifier for a rule based on its itemset indices."""
    return (
        tuple(sorted(rule_masks.undesired_itemset)),
        tuple(sorted(rule_masks.desired_itemset)),
        rule_masks.target_undesired,
        rule_masks.target_desired,
    )


def _attribute_value_signature(rule_masks) -> Tuple:
    """Return a hashable identifier based on the (attribute, value) pairs.

    Unlike :func:`_rule_signature` (which uses integer column indices
    specific to the train fold's encoding), this representation survives
    differences in one-hot column ordering between folds, so it is the
    right key for cross-fold Jaccard stability.
    """
    return (
        tuple(sorted(rule_masks.mask_undesired.items())),
        tuple(sorted(rule_masks.mask_desired.items())),
        rule_masks.target_attribute,
        rule_masks.target_undesired,
        rule_masks.target_desired,
    )


def _score_rule_on_data(
    rule_masks,
    data: pd.DataFrame,
    intrinsic_utility_table: Optional[dict],
    transition_utility_table: Optional[dict],
    column_values: dict,
) -> dict:
    """Compute uplift / support / gain for a single rule against *data*.

    Returns a dict with keys ``support``, ``confidence_u``, ``confidence_d``,
    ``uplift``, and ``realistic_rule_gain`` (the last is ``None`` when both
    utility tables are empty).
    """
    n_u_ante, n_u_match, n_d_ante, n_d_match, n_total = compute_group_counts(data, rule_masks)
    uplift = compute_uplift_from_counts(n_u_ante, n_u_match, n_d_ante, n_d_match, n_total)

    conf_u = n_u_match / n_u_ante if n_u_ante > 0 else 0.0
    conf_d = n_d_match / n_d_ante if n_d_ante > 0 else 0.0

    gain = None
    if intrinsic_utility_table or transition_utility_table:
        gain = compute_realistic_gain(
            rule_masks,
            conf_u=conf_u,
            conf_d=conf_d,
            intrinsic_utility_table=intrinsic_utility_table,
            transition_utility_table=transition_utility_table,
            column_values=column_values,
        )

    # "Support" on the test fold is the union antecedent count — i.e. how
    # many rows match either side.  The maximum of the two is a useful
    # weight for coverage-weighted Qini.
    support = max(n_u_ante, n_d_ante)
    return {
        'support': support,
        'support_undesired': n_u_ante,
        'support_desired': n_d_ante,
        'confidence_undesired': conf_u,
        'confidence_desired': conf_d,
        'uplift': uplift,
        'realistic_rule_gain': gain,
    }


def _strategy_score(record: dict, strategy: str, risk_lambda: float = 1.96) -> float:
    """Return the ranking score for a rule record under a given strategy.

    Returns ``-inf`` when the rule should be excluded (``lower_positive``
    rejects rules whose lower CI bound is non-positive).
    """
    point = record['train_uplift']
    lo = record['train_uplift_lower']
    se = record['train_uplift_se']
    if strategy == 'point':
        return float(point)
    if strategy == 'lower':
        return float(lo)
    if strategy == 'lower_positive':
        return float(lo) if lo > 0.0 else float('-inf')
    if strategy == 'risk_adjusted':
        return float(point - risk_lambda * se)
    raise ValueError(f"Unknown strategy '{strategy}'.")


def _metric_value(
    name: str, scores: np.ndarray, outcomes: np.ndarray, supports: np.ndarray, k_fraction: float
) -> float:
    if name == 'uplift_at_k':
        return M.uplift_at_k(scores, outcomes, k_fraction=k_fraction)
    if name == 'qini':
        return M.qini_coefficient(scores, outcomes, supports=supports)
    if name == 'auuc':
        return M.auuc(scores, outcomes, supports=supports)
    if name == 'profit_at_k':
        return M.incremental_profit_at_k(scores, outcomes, supports=supports, k_fraction=k_fraction)
    raise ValueError(f"Unknown metric '{name}'.")


# ---------------------------------------------------------------------------
# CrossValidator
# ---------------------------------------------------------------------------


class CrossValidator:
    """Run K-fold cross-validation around the ``ActionRules.fit`` pipeline.

    Parameters
    ----------
    action_rules_factory : callable
        Zero-argument callable returning a fresh ``ActionRules`` instance
        configured with the desired hyperparameters and utility tables.
        Called once per fold so each fold gets its own pristine instance.
    stable_attributes, flexible_attributes : list of str
    target : str
    target_undesired_state, target_desired_state : str
    n_splits : int
        Number of CV folds; must be ≥ 2.
    stratify : bool
        When ``True`` (default), folds are stratified by ``target`` value.
    intrinsic_utility_table, transition_utility_table : dict, optional
        Utility tables.  When both are ``None``/empty, gain-aware columns
        are omitted from the output.
    strategies : tuple of str
        Subset of :data:`STRATEGIES` to evaluate.
    metrics : tuple of str
        Subset of :data:`METRICS` to evaluate.
    k_fraction : float
        Top-k cutoff for the ``*_at_k`` metrics.
    ci_method : str
        Inference engine for train-fold confidence intervals.  One of
        ``'bootstrap'``, ``'analytic'``, ``'wald'``, ``'bayesian'``.
    n_bootstrap : int
        Number of bootstrap replicates used by the train-fold CI engine
        (only when ``ci_method='bootstrap'``).
    risk_lambda : float
        λ in the ``risk_adjusted`` strategy: ``score = point - λ·SE``.
    confidence_level : float
        Nominal coverage for CI computations.
    random_state : int, optional
        Seed for reproducibility (governs both the fold split and the
        train-fold bootstrap engine).
    n_bootstrap_oof : int
        Bootstrap replicates used for the across-fold rule-resampling CI on
        the final summary table.  Set to ``0`` to skip bootstrap CIs.
    track_stability : bool
        When ``True``, populate :attr:`CrossValidationResult.rule_stability`
        with pairwise Jaccard overlap of discovered rule sets across folds.
    compute_insample_baseline : bool
        When ``True``, additionally mine action rules on the full dataset
        and score every (strategy, metric) on the same full dataset to
        obtain the apparent (in-sample) performance baseline.  Result is
        stored on :attr:`CrossValidationResult.insample_summary`.  Default
        ``False`` preserves backward compatibility — the existing aggregated
        outputs are unchanged.
    """

    def __init__(
        self,
        action_rules_factory: Callable,
        *,
        stable_attributes: List[str],
        flexible_attributes: List[str],
        target: str,
        target_undesired_state: str,
        target_desired_state: str,
        n_splits: int = 5,
        stratify: bool = True,
        intrinsic_utility_table: Optional[dict] = None,
        transition_utility_table: Optional[dict] = None,
        strategies: Sequence[str] = STRATEGIES,
        metrics: Sequence[str] = METRICS,
        k_fraction: float = 0.2,
        ci_method: str = 'bootstrap',
        n_bootstrap: int = 500,
        risk_lambda: float = 1.96,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
        n_bootstrap_oof: int = 1000,
        bootstrap_design: str = 'cluster_fold',
        track_stability: bool = True,
        use_sparse_matrix: bool = False,
        compute_insample_baseline: bool = False,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2; got {n_splits}.")
        if not 0.0 < k_fraction <= 1.0:
            raise ValueError(f"k_fraction must be in (0, 1]; got {k_fraction}.")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(f"confidence_level must be in (0, 1); got {confidence_level}.")
        if n_bootstrap_oof < 0:
            raise ValueError(f"n_bootstrap_oof must be >= 0; got {n_bootstrap_oof}.")
        if bootstrap_design not in ('cluster_fold', 'oof_pool'):
            raise ValueError(
                f"bootstrap_design must be 'cluster_fold' or 'oof_pool'; got {bootstrap_design!r}."
            )
        unknown_s = set(strategies) - set(STRATEGIES)
        if unknown_s:
            raise ValueError(f"Unknown strategies: {sorted(unknown_s)}. Supported: {STRATEGIES}.")
        unknown_m = set(metrics) - set(METRICS)
        if unknown_m:
            raise ValueError(f"Unknown metrics: {sorted(unknown_m)}. Supported: {METRICS}.")

        self.factory = action_rules_factory
        self.stable_attributes = list(stable_attributes)
        self.flexible_attributes = list(flexible_attributes)
        self.target = target
        self.target_undesired_state = target_undesired_state
        self.target_desired_state = target_desired_state
        self.n_splits = int(n_splits)
        self.stratify = bool(stratify)
        self.intrinsic_utility_table = intrinsic_utility_table or {}
        self.transition_utility_table = transition_utility_table or {}
        self.strategies = tuple(strategies)
        self.metrics = tuple(metrics)
        self.k_fraction = float(k_fraction)
        self.ci_method = ci_method
        self.n_bootstrap = int(n_bootstrap)
        self.risk_lambda = float(risk_lambda)
        self.confidence_level = float(confidence_level)
        self.random_state = random_state
        self.n_bootstrap_oof = int(n_bootstrap_oof)
        self.bootstrap_design = bootstrap_design
        self.track_stability = bool(track_stability)
        self.use_sparse_matrix = bool(use_sparse_matrix)
        self.compute_insample_baseline = bool(compute_insample_baseline)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame) -> CrossValidationResult:
        """Execute the full K-fold pipeline and return the aggregated result."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("CrossValidator.run currently requires a pandas DataFrame.")
        data = data.reset_index(drop=True)

        strat_y = data[self.target] if self.stratify else pd.Series(np.zeros(len(data)))
        splits = stratified_kfold_indices(strat_y, self.n_splits, random_state=self.random_state)

        all_rule_records: List[dict] = []
        n_rules_per_fold: List[int] = []
        fold_signatures: List[set] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_df = data.iloc[train_idx].reset_index(drop=True)
            test_df = data.iloc[test_idx].reset_index(drop=True)

            fold_records, signatures = self._process_fold(fold_idx, train_df, test_df)
            all_rule_records.extend(fold_records)
            n_rules_per_fold.append(len(fold_records))
            fold_signatures.append(signatures)

        rule_records_df = pd.DataFrame(all_rule_records)
        fold_summary = self._compute_fold_summary(rule_records_df)
        strategy_summary = self._compute_strategy_summary(rule_records_df, fold_summary)

        rule_stability = self._compute_rule_stability(fold_signatures) if self.track_stability else None

        insample_summary: Optional[pd.DataFrame] = None
        if self.compute_insample_baseline:
            insample_summary = self._compute_insample_summary(data)

        return CrossValidationResult(
            n_splits=self.n_splits,
            strategies=self.strategies,
            metrics=self.metrics,
            k_fraction=self.k_fraction,
            confidence_level=self.confidence_level,
            has_utility=bool(self.intrinsic_utility_table or self.transition_utility_table),
            rule_records=rule_records_df,
            fold_summary=fold_summary,
            strategy_summary=strategy_summary,
            insample_summary=insample_summary,
            rule_stability=rule_stability,
            n_rules_per_fold=n_rules_per_fold,
        )

    # ------------------------------------------------------------------
    # Per-fold processing
    # ------------------------------------------------------------------

    def _process_fold(
        self,
        fold_idx: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[List[dict], set]:
        """Run fit + CI on the train fold; re-score every rule on the test fold."""
        # Filter to rows that actually have at least one of the target values
        # we care about, then verify both states are present.  Otherwise the
        # rule mining would crash on a one-class fold (rare with stratified CV
        # but can happen for tiny datasets).
        target_col = self.target
        if str(self.target_undesired_state) not in set(train_df[target_col].astype(str)):
            return [], set()
        if str(self.target_desired_state) not in set(train_df[target_col].astype(str)):
            return [], set()

        ar = self.factory()
        try:
            ar.fit(
                train_df,
                stable_attributes=self.stable_attributes,
                flexible_attributes=self.flexible_attributes,
                target=target_col,
                target_undesired_state=self.target_undesired_state,
                target_desired_state=self.target_desired_state,
                use_sparse_matrix=self.use_sparse_matrix,
            )
        except Exception as exc:
            warnings.warn(
                f"CrossValidator: fold {fold_idx} fit failed ({type(exc).__name__}: {exc}); "
                "skipping fold and continuing.",
                RuntimeWarning,
                stacklevel=2,
            )
            return [], set()

        # No rules discovered on this fold — return an empty record set so the
        # downstream summary just reports empty / zero values for the fold.
        if ar.output is None or not ar.output.action_rules:
            return [], set()

        # Train-fold confidence intervals provide the per-rule CI bounds and
        # SE used by the lower / risk-adjusted strategies.
        try:
            ci_results = ar.confidence_intervals(
                train_df,
                method=self.ci_method,
                confidence_level=self.confidence_level,
                metric='uplift',
                n_bootstrap=self.n_bootstrap,
                random_state=self.random_state,
            )
        except Exception as exc:
            # CI engines occasionally fail on degenerate folds (e.g. zero variance).
            # Warn rather than swallow silently; the lower-CI strategies will collapse
            # to point estimates for this fold.
            warnings.warn(
                f"CrossValidator: fold {fold_idx} CI computation failed "
                f"({type(exc).__name__}: {exc}); falling back to point estimates.",
                RuntimeWarning,
                stacklevel=2,
            )
            ci_results = []

        ci_by_rule_idx = {r.rule_index: r for r in ci_results}
        rule_masks_list = extract_rule_masks(ar.output)
        column_values = ar.output.column_values

        intrinsic = ar._original_intrinsic_utility_table or self.intrinsic_utility_table
        transition = ar._original_transition_utility_table or self.transition_utility_table

        records: List[dict] = []
        signatures: set = set()

        for rule_idx, rule_masks in enumerate(rule_masks_list):
            train_score = _score_rule_on_data(rule_masks, train_df, intrinsic, transition, column_values)
            test_score = _score_rule_on_data(rule_masks, test_df, intrinsic, transition, column_values)
            ci = ci_by_rule_idx.get(rule_idx)

            record = {
                'fold': fold_idx,
                'rule_index_in_fold': rule_idx,
                'support_train': train_score['support'],
                'support_test': test_score['support'],
                'train_uplift': train_score['uplift'],
                'train_uplift_lower': ci.uplift_ci_lower if ci is not None else train_score['uplift'],
                'train_uplift_upper': ci.uplift_ci_upper if ci is not None else train_score['uplift'],
                'train_uplift_se': ci.uplift_se if ci is not None else 0.0,
                'test_uplift': test_score['uplift'],
            }
            if self.intrinsic_utility_table or self.transition_utility_table:
                record['train_gain'] = train_score['realistic_rule_gain']
                record['test_gain'] = test_score['realistic_rule_gain']
            records.append(record)
            signatures.add(_attribute_value_signature(rule_masks))

        return records, signatures

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def _compute_fold_summary(self, rule_records: pd.DataFrame) -> pd.DataFrame:
        """One row per (fold, strategy, metric) with the metric value on that fold."""
        rows: List[dict] = []
        if rule_records.empty or 'fold' not in rule_records.columns:
            return pd.DataFrame(columns=['fold', 'strategy', 'metric', 'metric_target', 'value'])
        has_gain = 'test_gain' in rule_records.columns
        for fold_idx, fold_df in rule_records.groupby('fold'):
            records = fold_df.to_dict(orient='records')
            for strategy in self.strategies:
                scores = np.array([_strategy_score(r, strategy, self.risk_lambda) for r in records])
                # Drop excluded rules (lower_positive may exclude all).
                valid_mask = np.isfinite(scores)
                if not valid_mask.any():
                    for metric in self.metrics:
                        rows.append(
                            {
                                'fold': fold_idx,
                                'strategy': strategy,
                                'metric': metric,
                                'metric_target': 'uplift',
                                'value': 0.0,
                            }
                        )
                        if has_gain:
                            rows.append(
                                {
                                    'fold': fold_idx,
                                    'strategy': strategy,
                                    'metric': metric,
                                    'metric_target': 'gain',
                                    'value': 0.0,
                                }
                            )
                    continue

                scores_v = scores[valid_mask]
                test_uplift_v = fold_df['test_uplift'].to_numpy()[valid_mask]
                support_test_v = fold_df['support_test'].to_numpy()[valid_mask]
                for metric in self.metrics:
                    rows.append(
                        {
                            'fold': fold_idx,
                            'strategy': strategy,
                            'metric': metric,
                            'metric_target': 'uplift',
                            'value': _metric_value(
                                metric,
                                scores_v,
                                test_uplift_v,
                                support_test_v.astype(float),
                                self.k_fraction,
                            ),
                        }
                    )
                if has_gain:
                    test_gain_raw = fold_df['test_gain'].fillna(0.0).to_numpy()[valid_mask].astype(float)
                    for metric in self.metrics:
                        rows.append(
                            {
                                'fold': fold_idx,
                                'strategy': strategy,
                                'metric': metric,
                                'metric_target': 'gain',
                                'value': _metric_value(
                                    metric,
                                    scores_v,
                                    test_gain_raw,
                                    support_test_v.astype(float),
                                    self.k_fraction,
                                ),
                            }
                        )
        return pd.DataFrame(rows)

    def _compute_strategy_summary(
        self,
        rule_records: pd.DataFrame,
        fold_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate over folds: mean ± std (stability) plus optional bootstrap CI."""
        if fold_summary.empty:
            return fold_summary
        agg = (
            fold_summary.groupby(['strategy', 'metric', 'metric_target'])['value']
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        agg.rename(columns={'count': 'n_folds'}, inplace=True)
        agg['std'] = agg['std'].fillna(0.0)

        if self.n_bootstrap_oof > 0 and not rule_records.empty:
            ci_low, ci_high = self._bootstrap_oof_cis(rule_records)
            agg['ci_lower'] = [
                ci_low.get((row['strategy'], row['metric'], row['metric_target']), np.nan) for _, row in agg.iterrows()
            ]
            agg['ci_upper'] = [
                ci_high.get((row['strategy'], row['metric'], row['metric_target']), np.nan) for _, row in agg.iterrows()
            ]
        return agg

    def _bootstrap_oof_cis(self, rule_records: pd.DataFrame) -> Tuple[dict, dict]:
        """Bootstrap CI under the configured design.

        ``cluster_fold`` (default): for each replicate, resample rules within
        each fold with replacement, compute the metric *per fold*, then
        average across folds.  Estimates the same fold-mean quantity as the
        ``mean`` column of the strategy summary.

        ``oof_pool`` (legacy): resample within fold, concatenate into a
        single pool, compute the metric on the union.  Kept for backwards
        compatibility; estimates a pool-level statistic ~K× the fold mean.
        """
        rng = np.random.default_rng(self.random_state)
        has_gain = 'test_gain' in rule_records.columns

        folds = sorted(rule_records['fold'].unique())
        per_fold: dict = {}
        for fold_idx in folds:
            fold_df = rule_records[rule_records['fold'] == fold_idx]
            per_fold[fold_idx] = {
                'records': fold_df.to_dict(orient='records'),
                'test_uplift': fold_df['test_uplift'].to_numpy(),
                'support_test': fold_df['support_test'].to_numpy().astype(float),
                'test_gain': fold_df['test_gain'].fillna(0.0).to_numpy().astype(float) if has_gain else None,
            }

        samples: dict = {}
        targets = ('uplift', 'gain') if has_gain else ('uplift',)
        for strategy in self.strategies:
            for metric in self.metrics:
                for tgt in targets:
                    samples[(strategy, metric, tgt)] = np.empty(self.n_bootstrap_oof, dtype=float)

        if self.bootstrap_design == 'cluster_fold':
            for b in range(self.n_bootstrap_oof):
                self._fill_cluster_bootstrap_row(samples, per_fold, folds, rng, has_gain, targets, b)
        else:  # 'oof_pool'
            for b in range(self.n_bootstrap_oof):
                pool = self._build_oof_pool(per_fold, folds, rng, has_gain)
                if pool is None:
                    for key in samples:
                        samples[key][b] = 0.0
                    continue
                self._fill_bootstrap_row(samples, pool, has_gain, targets, b)

        alpha = (1.0 - self.confidence_level) / 2.0
        ci_low: dict = {}
        ci_high: dict = {}
        for key, arr in samples.items():
            ci_low[key] = float(np.quantile(arr, alpha))
            ci_high[key] = float(np.quantile(arr, 1.0 - alpha))
        return ci_low, ci_high

    def _fill_cluster_bootstrap_row(
        self, samples: dict, per_fold: dict, folds: list, rng, has_gain: bool, targets: tuple, b: int
    ) -> None:
        """Cluster bootstrap by fold: per-fold metric, then mean across folds.

        Resamples rule records within each fold independently, computes the
        targeting metric on that fold's resample, and averages across folds.
        The resulting bootstrap distribution estimates the same fold-mean
        quantity as ``CrossValidationResult.strategy_summary['mean']``.
        """
        per_fold_values_uplift: dict = {(s, m): [] for s in self.strategies for m in self.metrics}
        per_fold_values_gain: dict = (
            {(s, m): [] for s in self.strategies for m in self.metrics} if has_gain else {}
        )

        for fold_idx in folds:
            info = per_fold[fold_idx]
            records = info['records']
            if not records:
                continue
            n = len(records)
            resample_idx = rng.integers(0, n, size=n)
            rec_resample = [records[i] for i in resample_idx]
            uplift_all = info['test_uplift'][resample_idx]
            support_all = info['support_test'][resample_idx]
            gain_all = (
                info['test_gain'][resample_idx] if (has_gain and info['test_gain'] is not None) else None
            )

            for strategy in self.strategies:
                scores = np.array(
                    [_strategy_score(r, strategy, self.risk_lambda) for r in rec_resample]
                )
                valid = np.isfinite(scores)
                if not valid.any():
                    for metric in self.metrics:
                        per_fold_values_uplift[(strategy, metric)].append(0.0)
                        if has_gain:
                            per_fold_values_gain[(strategy, metric)].append(0.0)
                    continue
                scores_v = scores[valid]
                uplift_v = uplift_all[valid]
                support_v = support_all[valid]
                for metric in self.metrics:
                    per_fold_values_uplift[(strategy, metric)].append(
                        _metric_value(metric, scores_v, uplift_v, support_v, self.k_fraction)
                    )
                if has_gain and gain_all is not None:
                    gain_v = gain_all[valid]
                    for metric in self.metrics:
                        per_fold_values_gain[(strategy, metric)].append(
                            _metric_value(metric, scores_v, gain_v, support_v, self.k_fraction)
                        )

        for (strategy, metric), vals in per_fold_values_uplift.items():
            samples[(strategy, metric, 'uplift')][b] = float(np.mean(vals)) if vals else 0.0
        if has_gain:
            for (strategy, metric), vals in per_fold_values_gain.items():
                samples[(strategy, metric, 'gain')][b] = float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _build_oof_pool(per_fold: dict, folds: list, rng, has_gain: bool):
        """Stratified resample within each fold, return concatenated OOF pool.

        Returns ``None`` when every fold has zero rules.  Otherwise returns a
        dict with parallel arrays ``records``, ``test_uplift``, ``support`` and
        (when utility tables are used) ``test_gain``.
        """
        pool_records: List[dict] = []
        pool_test_uplift: List[float] = []
        pool_support: List[float] = []
        pool_test_gain: List[float] = []
        for fold_idx in folds:
            info = per_fold[fold_idx]
            records = info['records']
            if not records:
                continue
            n = len(records)
            resample_idx = rng.integers(0, n, size=n)
            pool_records.extend(records[i] for i in resample_idx)
            pool_test_uplift.extend(info['test_uplift'][resample_idx].tolist())
            pool_support.extend(info['support_test'][resample_idx].tolist())
            if has_gain and info['test_gain'] is not None:
                pool_test_gain.extend(info['test_gain'][resample_idx].tolist())
        if not pool_records:
            return None
        return {
            'records': pool_records,
            'test_uplift': np.asarray(pool_test_uplift),
            'support': np.asarray(pool_support),
            'test_gain': np.asarray(pool_test_gain) if has_gain else None,
        }

    def _fill_bootstrap_row(self, samples: dict, pool: dict, has_gain: bool, targets: tuple, b: int) -> None:
        """Compute every (strategy, metric, target) value on a single OOF pool."""
        for strategy in self.strategies:
            scores = np.array([_strategy_score(r, strategy, self.risk_lambda) for r in pool['records']])
            valid = np.isfinite(scores)
            if not valid.any():
                for metric in self.metrics:
                    for tgt in targets:
                        samples[(strategy, metric, tgt)][b] = 0.0
                continue
            scores_v = scores[valid]
            uplift_v = pool['test_uplift'][valid]
            support_v = pool['support'][valid]
            for metric in self.metrics:
                samples[(strategy, metric, 'uplift')][b] = _metric_value(
                    metric, scores_v, uplift_v, support_v, self.k_fraction
                )
            if has_gain and pool['test_gain'] is not None:
                gain_v = pool['test_gain'][valid]
                for metric in self.metrics:
                    samples[(strategy, metric, 'gain')][b] = _metric_value(
                        metric, scores_v, gain_v, support_v, self.k_fraction
                    )

    def _compute_insample_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the apparent (in-sample) baseline by mining and scoring on the full dataset.

        Reuses :meth:`_process_fold` with ``train_df == test_df == data`` so the
        in-sample numbers come from exactly the same scoring code path as the
        per-fold ones; no duplication of metric or strategy logic.  The returned
        frame has the same ``(strategy, metric, metric_target, value)`` schema as
        :attr:`CrossValidationResult.fold_summary`, with a single synthetic
        ``fold`` index of ``-1``.

        The result is intentionally optimistic — mining selected rules that fit
        this exact dataset, and we score them on the same data.  It is reported
        to expose the optimism gap relative to the CV mean (Hastie, Tibshirani &
        Friedman, *Elements of Statistical Learning*, Ch. 7).
        """
        records, _ = self._process_fold(-1, data, data)
        if not records:
            return pd.DataFrame(columns=['strategy', 'metric', 'metric_target', 'value'])
        rule_records_df = pd.DataFrame(records)
        fold_summary = self._compute_fold_summary(rule_records_df)
        # Drop the synthetic ``fold`` column so callers see a clean (strategy,
        # metric, metric_target, value) frame analogous to a per-rule baseline.
        if 'fold' in fold_summary.columns:
            fold_summary = fold_summary.drop(columns=['fold']).reset_index(drop=True)
        return fold_summary

    def _compute_rule_stability(self, fold_signatures: List[set]) -> pd.DataFrame:
        r"""Pairwise Jaccard overlap of rule signatures across folds.

        Both empty → 1.0 (trivially identical empty sets); exactly one empty →
        0.0 (no overlap); otherwise standard \|A∩B\|/\|A∪B\|.
        """
        rows = []
        for i in range(len(fold_signatures)):
            for j in range(i + 1, len(fold_signatures)):
                a = fold_signatures[i]
                b = fold_signatures[j]
                if not a and not b:
                    jaccard = 1.0
                elif not a or not b:
                    jaccard = 0.0
                else:
                    jaccard = len(a & b) / len(a | b)
                rows.append({'fold_a': i, 'fold_b': j, 'jaccard': float(jaccard)})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def cross_validate(
    action_rules_factory: Callable,
    data: pd.DataFrame,
    *,
    stable_attributes: List[str],
    flexible_attributes: List[str],
    target: str,
    target_undesired_state: str,
    target_desired_state: str,
    n_splits: int = 5,
    stratify: bool = True,
    intrinsic_utility_table: Optional[dict] = None,
    transition_utility_table: Optional[dict] = None,
    strategies: Sequence[str] = STRATEGIES,
    metrics: Sequence[str] = METRICS,
    k_fraction: float = 0.2,
    ci_method: str = 'bootstrap',
    n_bootstrap: int = 500,
    risk_lambda: float = 1.96,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    n_bootstrap_oof: int = 1000,
    bootstrap_design: str = 'cluster_fold',
    track_stability: bool = True,
    use_sparse_matrix: bool = False,
    compute_insample_baseline: bool = False,
) -> CrossValidationResult:
    """Run :class:`CrossValidator` and return its :class:`CrossValidationResult`.

    See :class:`CrossValidator` for the full parameter documentation.
    """
    validator = CrossValidator(
        action_rules_factory,
        stable_attributes=stable_attributes,
        flexible_attributes=flexible_attributes,
        target=target,
        target_undesired_state=target_undesired_state,
        target_desired_state=target_desired_state,
        n_splits=n_splits,
        stratify=stratify,
        intrinsic_utility_table=intrinsic_utility_table,
        transition_utility_table=transition_utility_table,
        strategies=strategies,
        metrics=metrics,
        k_fraction=k_fraction,
        ci_method=ci_method,
        n_bootstrap=n_bootstrap,
        risk_lambda=risk_lambda,
        confidence_level=confidence_level,
        random_state=random_state,
        n_bootstrap_oof=n_bootstrap_oof,
        bootstrap_design=bootstrap_design,
        track_stability=track_stability,
        use_sparse_matrix=use_sparse_matrix,
        compute_insample_baseline=compute_insample_baseline,
    )
    return validator.run(data)
