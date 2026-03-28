"""Bootstrap confidence interval engine for action rules."""

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import (
    ConfidenceIntervalResult,
    InferenceEngine,
    categorize_rule,
    compute_group_counts,
    compute_realistic_gain,
    compute_uplift_from_counts,
)


def _precompute_rule_masks(
    data: pd.DataFrame,
    rule: object,
) -> tuple:
    """Pre-compute boolean numpy arrays for a rule's conditions.

    Building these masks once before the bootstrap loop avoids repeated
    DataFrame.iloc slicing and string-comparison work inside the inner loop,
    yielding an order-of-magnitude speedup on large datasets.

    Parameters
    ----------
    data : pd.DataFrame
        Full original dataset.
    rule : RuleMasks
        Structured rule object with mask_undesired, mask_desired,
        target_attribute, target_undesired, and target_desired fields.

    Returns
    -------
    tuple
        Four boolean arrays ``(u_ante, u_match, d_ante, d_match)`` each of
        shape ``(n,)`` where ``n = len(data)``.

        * ``u_ante`` — rows satisfying the undesired antecedent conditions.
        * ``u_match`` — ``u_ante`` rows whose target equals ``target_undesired``.
        * ``d_ante`` — rows satisfying the desired antecedent conditions.
        * ``d_match`` — ``d_ante`` rows whose target equals ``target_desired``.
    """
    n = len(data)

    # Undesired antecedent: all conditions in mask_undesired must hold.
    u_ante = np.ones(n, dtype=bool)
    for col, val in rule.mask_undesired.items():
        u_ante &= data[col].astype(str).values == str(val)

    # Desired antecedent: all conditions in mask_desired must hold.
    d_ante = np.ones(n, dtype=bool)
    for col, val in rule.mask_desired.items():
        d_ante &= data[col].astype(str).values == str(val)

    # Target column as a string array — computed once, shared by both sides.
    target_arr = data[rule.target_attribute].astype(str).values

    u_match = u_ante & (target_arr == rule.target_undesired)
    d_match = d_ante & (target_arr == rule.target_desired)

    return u_ante, u_match, d_ante, d_match


class BootstrapEngine(InferenceEngine):
    """Non-parametric percentile bootstrap confidence interval engine.

    For each action rule the engine draws *n_bootstrap* resamples (with
    replacement) from *data*, computes the uplift — and optionally the
    realistic rule gain — on every resample, and reports the empirical
    percentile interval as the confidence interval.

    Parameters
    ----------
    n_bootstrap : int, optional
        Number of bootstrap resamples.  Default ``1000``.
    random_state : int, optional
        Seed for the random-number generator, enabling reproducible results.
        ``None`` (default) uses the global NumPy state.

    Notes
    -----
    Resamples that produce zero antecedent support for either the undesired
    or the desired rule are recorded as ``NaN`` and excluded from all
    summary statistics.  This is conservative: sparse rules will exhibit
    wider intervals because many resamples are discarded.

    The point estimate (``uplift_point``) is the mean of the non-NaN
    bootstrap distribution, not the value computed on the full dataset.
    This is consistent with the percentile-bootstrap framework where the
    bootstrap distribution is used as a surrogate for the sampling
    distribution.

    Performance-wise, boolean masks for each rule's conditions are
    pre-computed as numpy arrays before the bootstrap loop.  Integer-index
    fancy indexing then replaces ``DataFrame.iloc`` slicing and per-row
    string comparisons inside the inner loop, giving roughly a 50x speedup
    on large datasets.
    """

    def __init__(self, n_bootstrap: int = 1000, random_state: Optional[int] = None):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        data: pd.DataFrame,
        rules: list,
        confidence_level: float = 0.95,
        intrinsic_utility_table: Optional[dict] = None,
        transition_utility_table: Optional[dict] = None,
        column_values: Optional[dict] = None,
    ) -> List[ConfidenceIntervalResult]:
        """Compute bootstrap percentile confidence intervals for each action rule.

        Parameters
        ----------
        data : pd.DataFrame
            The full original dataset.  Each row is one transaction; columns
            correspond to attribute names as strings.
        rules : list
            List of :class:`~action_rules.inference.base.RuleMasks` objects
            produced by :func:`~action_rules.inference.base.extract_rule_masks`.
        confidence_level : float, optional
            Nominal coverage probability for the interval.  Default ``0.95``.
        intrinsic_utility_table : dict, optional
            Mapping ``(attribute, value) -> float`` for intrinsic utilities.
            When ``None``, gain metrics are omitted from the results.
        transition_utility_table : dict, optional
            Mapping ``(attribute, from_value, to_value) -> float`` for
            transition utilities.  When ``None``, gain metrics are omitted.
        column_values : dict, optional
            Mapping ``int -> (attribute, value)`` from ``Output.column_values``.
            Required when *intrinsic_utility_table* or *transition_utility_table*
            are provided so that integer itemset indices can be resolved.

        Returns
        -------
        List[ConfidenceIntervalResult]
            One result per rule in *rules*, in the same order.

        Notes
        -----
        The alpha level is split symmetrically::

            ci_lower = percentile(samples, 100 * alpha / 2)
            ci_upper = percentile(samples, 100 * (1 - alpha / 2))

        where ``alpha = 1 - confidence_level``.
        """
        # Determine whether gain computation is requested.
        compute_gain = (intrinsic_utility_table is not None) or (transition_utility_table is not None)

        # default_rng is the recommended modern NumPy RNG (PCG64). Unlike
        # RandomState it accepts None (unseeded) or an integer seed identically,
        # and its integers() method avoids the deprecated randint alias.
        rng = np.random.default_rng(self.random_state)

        n = len(data)
        alpha = 1.0 - confidence_level
        # Pre-compute tail percentile positions (avoids repeated arithmetic).
        pct_lower = 100.0 * alpha / 2.0
        pct_upper = 100.0 * (1.0 - alpha / 2.0)

        results: List[ConfidenceIntervalResult] = []

        for rule in rules:
            # Pre-compute boolean masks once per rule so the inner loop only
            # does cheap numpy fancy-indexing and integer summation.
            u_ante, u_match, d_ante, d_match = _precompute_rule_masks(data, rule)

            uplift_samples = np.empty(self.n_bootstrap)
            uplift_samples[:] = np.nan

            gain_samples: Optional[np.ndarray] = np.empty(self.n_bootstrap) if compute_gain else None
            if gain_samples is not None:
                gain_samples[:] = np.nan

            for b in range(self.n_bootstrap):
                # Draw a bootstrap resample (with replacement) using integer
                # index sampling, which is compatible with default_rng.
                indices = rng.integers(0, n, size=n)

                # Fancy-index the pre-computed boolean masks instead of
                # slicing the DataFrame — avoids Python-level iteration and
                # repeated string comparisons.
                n_u_ante = int(u_ante[indices].sum())
                n_u_match = int(u_match[indices].sum())
                n_d_ante = int(d_ante[indices].sum())
                n_d_match = int(d_match[indices].sum())
                n_total = n

                # Skip resamples where antecedent support is zero — uplift is
                # undefined and including a forced 0.0 would bias the interval.
                if n_u_ante == 0 or n_d_ante == 0:
                    continue

                uplift_samples[b] = compute_uplift_from_counts(n_u_ante, n_u_match, n_d_ante, n_d_match, n_total)

                if gain_samples is not None and column_values is not None:
                    conf_u = n_u_match / n_u_ante
                    conf_d = n_d_match / n_d_ante
                    gain_samples[b] = compute_realistic_gain(
                        rule,
                        conf_u,
                        conf_d,
                        intrinsic_utility_table,
                        transition_utility_table,
                        column_values,
                    )

            # Drop NaN entries introduced by zero-support resamples.
            valid_uplift = uplift_samples[~np.isnan(uplift_samples)]

            if valid_uplift.size == 0:
                # No valid resamples — return degenerate interval.
                uplift_point = float('nan')
                uplift_lower = float('nan')
                uplift_upper = float('nan')
                uplift_se = float('nan')
            else:
                uplift_point = float(np.mean(valid_uplift))
                uplift_lower = float(np.percentile(valid_uplift, pct_lower))
                uplift_upper = float(np.percentile(valid_uplift, pct_upper))
                # ddof=1 gives the sample standard deviation, consistent with
                # the standard error interpretation for the bootstrap.
                uplift_se = float(np.std(valid_uplift, ddof=1)) if valid_uplift.size > 1 else float('nan')

            # Compute point estimate summary statistics for support/confidence
            # from the full (non-resampled) dataset using the pre-computed masks.
            n_u_ante_full = int(u_ante.sum())
            n_u_match_full = int(u_match.sum())
            n_d_ante_full = int(d_ante.sum())
            n_d_match_full = int(d_match.sum())
            n_total_full = n

            # Support: number of transactions matching the undesired antecedent
            # (consistent with how the mining algorithm defines rule support).
            support = n_u_ante_full
            confidence = (n_u_match_full / n_u_ante_full) if n_u_ante_full > 0 else 0.0

            # Categorise the rule based on the uplift CI vs. zero threshold.
            # Guard against NaN bounds produced when no valid resamples exist.
            if np.isnan(uplift_lower) or np.isnan(uplift_upper):
                category = None
            else:
                category = categorize_rule(uplift_lower, uplift_upper, threshold=0.0)

            # Build optional gain statistics.
            gain_point: Optional[float] = None
            gain_lower: Optional[float] = None
            gain_upper: Optional[float] = None
            gain_se: Optional[float] = None
            clean_gain_samples: Optional[np.ndarray] = None

            if gain_samples is not None:
                valid_gain = gain_samples[~np.isnan(gain_samples)]
                clean_gain_samples = valid_gain if valid_gain.size > 0 else None

                if valid_gain.size == 0:
                    gain_point = float('nan')
                    gain_lower = float('nan')
                    gain_upper = float('nan')
                    gain_se = float('nan')
                else:
                    gain_point = float(np.mean(valid_gain))
                    gain_lower = float(np.percentile(valid_gain, pct_lower))
                    gain_upper = float(np.percentile(valid_gain, pct_upper))
                    gain_se = float(np.std(valid_gain, ddof=1)) if valid_gain.size > 1 else float('nan')

            results.append(
                ConfidenceIntervalResult(
                    rule_index=rule.rule_index,
                    method='bootstrap',
                    confidence_level=confidence_level,
                    uplift_point=uplift_point,
                    uplift_ci_lower=uplift_lower,
                    uplift_ci_upper=uplift_upper,
                    uplift_se=uplift_se,
                    realistic_rule_gain_point=gain_point,
                    realistic_rule_gain_ci_lower=gain_lower,
                    realistic_rule_gain_ci_upper=gain_upper,
                    realistic_rule_gain_se=gain_se,
                    support=support,
                    confidence=confidence,
                    category=category,
                    samples_uplift=valid_uplift if valid_uplift.size > 0 else None,
                    samples_gain=clean_gain_samples,
                )
            )

        return results
