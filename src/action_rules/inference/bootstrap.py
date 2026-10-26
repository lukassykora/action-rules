"""Bootstrap confidence interval engine for action rules."""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import (
    ConfidenceIntervalResult,
    InferenceEngine,
    RuleCategory,
    RuleMasks,
    bonferroni_confidence_level,
    categorize_rule,
    compute_realistic_gain,
    compute_uplift_from_counts,
    validate_simultaneous_m,
)

# Emit one warning per rule when the fraction of resamples that hit a
# degenerate ``n_u == 0`` or ``n_d == 0`` configuration exceeds this
# tolerance — the percentile interval is then conditioning on
# non-degeneracy, which is a different estimand than the unconditional one.
DEGENERATE_RESAMPLE_TOLERANCE = 0.01


def _precompute_rule_masks(
    data: pd.DataFrame,
    rule: RuleMasks,
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
    """Non-parametric bootstrap confidence interval engine.

    Supports two bootstrap variants:

    * **percentile** (default) — the empirical percentile interval of the
      bootstrap distribution.
    * **bca** — Bias-Corrected and Accelerated (BCa) bootstrap (Efron, 1987).
      Corrects for median bias and skewness using jackknife-estimated
      acceleration.  Recommended when the statistic is asymmetrically
      distributed or when sample sizes are small.

    For each action rule the engine draws *n_bootstrap* resamples (with
    replacement) from *data*, computes the uplift — and optionally the
    realistic rule gain — on every resample, and reports the confidence
    interval according to the chosen *bootstrap_type*.

    Parameters
    ----------
    n_bootstrap : int, optional
        Number of bootstrap resamples.  Default ``1000``.
    random_state : int, optional
        Seed for the random-number generator, enabling reproducible results.
        ``None`` (default) uses the global NumPy state.
    bootstrap_type : str, optional
        Which bootstrap CI variant to use.  Must be one of
        ``{'percentile', 'bca'}``.  Default ``'percentile'``.
    resample_mode : str, optional
        ``'rows'`` (default) draws ``n`` row indices with replacement.
        ``'multinomial'`` draws the five cell counts (undesired match,
        undesired non-match, desired match, desired non-match, other) in one
        vectorised multinomial call; it targets the same distribution as
        ``'rows'`` because the two antecedents of an action rule are disjoint.
    simultaneous_m : int, optional
        Bonferroni family size.  When ``> 1`` each interval is computed at
        ``alpha_eff = (1 - confidence_level) / simultaneous_m``.

    Notes
    -----
    Resamples that produce zero antecedent support for either the undesired
    or the desired rule are recorded as ``NaN`` and excluded from all
    summary statistics. Dropping draws does not imply conservativeness.
    More than 1% undefined draws makes the interval unsupported. BCa also
    requires every leave-one-out estimate to be defined.

    For the percentile bootstrap, the point estimate (``uplift_point``) is
    the mean of the non-NaN bootstrap distribution, not the value computed on
    the full dataset.  This is consistent with the percentile-bootstrap
    framework where the bootstrap distribution is used as a surrogate for the
    sampling distribution.

    For BCa, ``uplift_point`` is also the bootstrap mean (for consistency),
    while the CI bounds are derived from adjusted percentiles.

    Performance-wise, boolean masks for each rule's conditions are
    pre-computed as numpy arrays before the bootstrap loop.  Integer-index
    fancy indexing then replaces ``DataFrame.iloc`` slicing and per-row
    string comparisons inside the inner loop, giving roughly a 50x speedup
    on large datasets.

    References
    ----------
    Efron, B. (1987). Better bootstrap confidence intervals. *Journal of the
    American Statistical Association*, 82(397), 171–185.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        random_state: Optional[int] = None,
        *,
        bootstrap_type: str = "percentile",
        resample_mode: str = "rows",
        simultaneous_m: Optional[int] = None,
    ):
        valid = {"percentile", "bca"}
        if bootstrap_type not in valid:
            raise ValueError(f"Unknown bootstrap_type '{bootstrap_type}'. Choose from {valid}.")
        valid_modes = {"rows", "multinomial"}
        if resample_mode not in valid_modes:
            raise ValueError(f"Unknown resample_mode '{resample_mode}'. Choose from {valid_modes}.")
        validate_simultaneous_m(simultaneous_m)
        if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 2:
            raise ValueError("n_bootstrap must be an integer >= 2.")
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.bootstrap_type = bootstrap_type
        self.resample_mode = resample_mode
        self.simultaneous_m = simultaneous_m

    def _resample_multinomial(
        self,
        rng: np.random.Generator,
        masks: tuple,
        n: int,
        rule: RuleMasks,
        uplift_samples: np.ndarray,
        gain_samples: Optional[np.ndarray],
        gain_args: Optional[tuple],
    ) -> int:
        """Fill the sample arrays from vectorised 5-cell multinomial draws.

        Returns the number of degenerate resamples (``n_u = 0`` or ``n_d = 0``).
        """
        u_ante, u_match, d_ante, d_match = masks
        c_u_match = int(u_match.sum())
        c_d_match = int(d_match.sum())
        counts = [c_u_match, int(u_ante.sum()) - c_u_match, c_d_match, int(d_ante.sum()) - c_d_match]
        counts.append(n - sum(counts))
        probs = np.array(counts, dtype=float) / float(n)
        draws = rng.multinomial(n, probs, size=self.n_bootstrap)
        nu_arr = draws[:, 0] + draws[:, 1]
        nd_arr = draws[:, 2] + draws[:, 3]
        valid_mask = (nu_arr > 0) & (nd_arr > 0)
        if not np.any(valid_mask):
            return self.n_bootstrap
        conf_u_arr = draws[valid_mask, 0] / nu_arr[valid_mask]
        conf_d_arr = draws[valid_mask, 2] / nd_arr[valid_mask]
        uplift_samples[valid_mask] = (conf_d_arr + conf_u_arr - 1.0) * (nu_arr[valid_mask] / float(n))
        if gain_samples is not None and gain_args is not None:
            a = compute_realistic_gain(rule, 0.0, 1.0, *gain_args)
            b = compute_realistic_gain(rule, 1.0, 1.0, *gain_args) - a
            gain_samples[valid_mask] = a + b * (conf_u_arr + conf_d_arr - 1.0)
        return int((~valid_mask).sum())

    def _resample_rows(
        self,
        rng: np.random.Generator,
        masks: tuple,
        n: int,
        rule: RuleMasks,
        uplift_samples: np.ndarray,
        gain_samples: Optional[np.ndarray],
        gain_args: Optional[tuple],
    ) -> int:
        """Fill the sample arrays from row-level resamples drawn with replacement.

        Returns the number of degenerate resamples (``n_u = 0`` or ``n_d = 0``).
        """
        u_ante, u_match, d_ante, d_match = masks
        n_undefined = 0
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

            # Skip resamples where antecedent support is zero — uplift is
            # undefined for these draws.  Track the fraction so the caller
            # can diagnose unstable percentile intervals on low-support rules.
            if n_u_ante == 0 or n_d_ante == 0:
                n_undefined += 1
                continue

            uplift_samples[b] = compute_uplift_from_counts(n_u_ante, n_u_match, n_d_ante, n_d_match, n)

            if gain_samples is not None and gain_args is not None:
                gain_samples[b] = compute_realistic_gain(rule, n_u_match / n_u_ante, n_d_match / n_d_ante, *gain_args)
        return n_undefined

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bca_ci(
        samples: np.ndarray,
        original_estimate: float,
        jackknife_estimates: np.ndarray,
        confidence_level: float,
    ) -> tuple:
        """Compute BCa confidence interval.

        Implements the Bias-Corrected and Accelerated (BCa) bootstrap
        (Efron, 1987).  Adjusts the percentile cut-points using a bias
        correction term ``z0`` (derived from the fraction of bootstrap
        replicates below the original estimate) and an acceleration factor
        ``a`` (derived from the jackknife distribution of the statistic).

        Parameters
        ----------
        samples : np.ndarray
            Bootstrap replicates (may contain NaN).
        original_estimate : float
            Statistic computed on the full dataset.
        jackknife_estimates : np.ndarray
            Leave-one-out estimates, shape ``(n,)``.
        confidence_level : float
            Nominal coverage.

        Returns
        -------
        tuple
            ``(point, lower, upper, se)`` where *point* is the bootstrap mean
            of valid replicates, *lower* and *upper* are the BCa CI bounds,
            and *se* is the bootstrap standard deviation.
        """
        from scipy.stats import norm

        valid = samples[~np.isnan(samples)]
        if valid.size == 0:
            nan = float('nan')
            return nan, nan, nan, nan

        point = float(np.mean(valid))
        se = float(np.std(valid, ddof=1)) if valid.size > 1 else float('nan')

        # Bias correction z0: proportion of bootstrap replicates strictly
        # below the full-data estimate, mapped through the normal quantile.
        if not np.all(np.isfinite(jackknife_estimates)) or not np.isfinite(original_estimate):
            return point, float('nan'), float('nan'), se
        proportion_below = np.mean(valid < original_estimate) + 0.5 * np.mean(valid == original_estimate)
        proportion_below = np.clip(proportion_below, 1e-10, 1.0 - 1e-10)
        z0 = norm.ppf(proportion_below)

        # Acceleration factor via jackknife (Efron, 1987, eq. 6.7).
        jack_mean = np.mean(jackknife_estimates)
        diff = jack_mean - jackknife_estimates
        sum_diff_sq = np.sum(diff**2)
        a = np.sum(diff**3) / (6.0 * sum_diff_sq**1.5) if sum_diff_sq > 0 else 0.0

        # Adjusted percentile positions.
        alpha = 1.0 - confidence_level
        z_lower = norm.ppf(alpha / 2.0)
        z_upper = norm.ppf(1.0 - alpha / 2.0)

        def _adj(z_alpha: float) -> float:
            # BCa adjustment formula (Efron, 1987, eq. 6.5).
            numer = z0 + z_alpha
            denom = 1.0 - a * numer
            # Guard against near-zero denominator to avoid numerical blow-up.
            if abs(denom) < 1e-10:
                denom = 1e-10 if denom >= 0 else -1e-10
            return norm.cdf(z0 + numer / denom)

        pct_lo = np.clip(_adj(z_lower) * 100.0, 0.0, 100.0)
        pct_hi = np.clip(_adj(z_upper) * 100.0, 0.0, 100.0)

        if pct_lo > pct_hi:
            return point, float('nan'), float('nan'), se
        lower = float(np.percentile(valid, pct_lo))
        upper = float(np.percentile(valid, pct_hi))

        return point, lower, upper, se

    @staticmethod
    def _jackknife_uplift(
        u_ante: np.ndarray,
        u_match: np.ndarray,
        d_ante: np.ndarray,
        d_match: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Compute leave-one-out uplift estimates using pre-computed boolean masks.

        Each estimate leaves out one row and recomputes the uplift from the
        adjusted counts.  This is O(N) because it only subtracts the
        contribution of each row from the pre-computed totals, rather than
        re-scanning all rows N times.

        Parameters
        ----------
        u_ante : np.ndarray
            Boolean mask — rows satisfying the undesired antecedent.
        u_match : np.ndarray
            Boolean mask — rows satisfying undesired antecedent with correct target.
        d_ante : np.ndarray
            Boolean mask — rows satisfying the desired antecedent.
        d_match : np.ndarray
            Boolean mask — rows satisfying desired antecedent with correct target.
        n : int
            Total number of rows in the dataset.

        Returns
        -------
        np.ndarray
            Leave-one-out uplift estimates, shape ``(n,)``.
        """
        full_u_ante = int(u_ante.sum())
        full_u_match = int(u_match.sum())
        full_d_ante = int(d_ante.sum())
        full_d_match = int(d_match.sum())

        nu = full_u_ante - u_ante.astype(int)
        nd = full_d_ante - d_ante.astype(int)
        valid = (nu > 0) & (nd > 0) & (n > 1)
        jack = np.full(n, np.nan)
        delta = (full_u_match - u_match[valid]) / nu[valid] + (full_d_match - d_match[valid]) / nd[valid] - 1.0
        jack[valid] = delta * nu[valid] / (n - 1)
        return jack

    @staticmethod
    def _jackknife_gain(
        u_ante: np.ndarray,
        u_match: np.ndarray,
        d_ante: np.ndarray,
        d_match: np.ndarray,
        n: int,
        rule: RuleMasks,
        intrinsic_utility_table: Optional[dict],
        transition_utility_table: Optional[dict],
        column_values: dict,
    ) -> np.ndarray:
        """Leave-one-out realistic rule gain estimates.

        Uses the same O(N) subtraction trick as :meth:`_jackknife_uplift` —
        full totals are computed once and each leave-one-out count is derived
        by subtracting the contribution of the omitted row.

        Parameters
        ----------
        u_ante : np.ndarray
            Boolean mask — rows satisfying the undesired antecedent.
        u_match : np.ndarray
            Boolean mask — rows satisfying undesired antecedent with correct target.
        d_ante : np.ndarray
            Boolean mask — rows satisfying the desired antecedent.
        d_match : np.ndarray
            Boolean mask — rows satisfying desired antecedent with correct target.
        n : int
            Total number of rows in the dataset.
        rule : RuleMasks
            Rule object forwarded to :func:`compute_realistic_gain`.
        intrinsic_utility_table : dict, optional
            Intrinsic utility mapping forwarded to :func:`compute_realistic_gain`.
        transition_utility_table : dict, optional
            Transition utility mapping forwarded to :func:`compute_realistic_gain`.
        column_values : dict
            Integer-index to ``(attribute, value)`` mapping.

        Returns
        -------
        np.ndarray
            Leave-one-out gain estimates, shape ``(n,)``.
        """
        full_u_ante = int(u_ante.sum())
        full_u_match = int(u_match.sum())
        full_d_ante = int(d_ante.sum())
        full_d_match = int(d_match.sum())

        nu = full_u_ante - u_ante.astype(int)
        nd = full_d_ante - d_ante.astype(int)
        valid = (nu > 0) & (nd > 0) & (n > 1)
        jack = np.full(n, np.nan)
        delta = (full_u_match - u_match[valid]) / nu[valid] + (full_d_match - d_match[valid]) / nd[valid] - 1.0
        a = compute_realistic_gain(rule, 0.0, 1.0, intrinsic_utility_table, transition_utility_table, column_values)
        b = compute_realistic_gain(rule, 1.0, 1.0, intrinsic_utility_table, transition_utility_table, column_values) - a
        jack[valid] = a + b * delta
        return jack

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
        """Compute bootstrap confidence intervals for each action rule.

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
        For ``bootstrap_type='percentile'``::

            ci_lower = percentile(samples, 100 * alpha / 2)
            ci_upper = percentile(samples, 100 * (1 - alpha / 2))

        For ``bootstrap_type='bca'``, the percentile cut-points are adjusted
        using bias correction and a jackknife-estimated acceleration factor
        (Efron, 1987).

        where ``alpha = 1 - confidence_level``.
        """
        # Determine whether gain computation is requested.
        compute_gain = (intrinsic_utility_table is not None) or (transition_utility_table is not None)

        # default_rng is the recommended modern NumPy RNG (PCG64). Unlike
        # RandomState it accepts None (unseeded) or an integer seed identically,
        # and its integers() method avoids the deprecated randint alias.
        rng = np.random.default_rng(self.random_state)

        n = len(data)
        # Results report the nominal (family-wise) level; the per-rule level
        # carries the Bonferroni adjustment.
        level = bonferroni_confidence_level(confidence_level, self.simultaneous_m)
        alpha = 1.0 - level
        gain_args = (
            (intrinsic_utility_table, transition_utility_table, column_values)
            if compute_gain and column_values is not None
            else None
        )
        # Pre-compute tail percentile positions (avoids repeated arithmetic).
        pct_lower = 100.0 * alpha / 2.0
        pct_upper = 100.0 * (1.0 - alpha / 2.0)

        results: List[ConfidenceIntervalResult] = []

        rule: RuleMasks
        for rule in rules:
            # Pre-compute boolean masks once per rule so the inner loop only
            # does cheap numpy fancy-indexing and integer summation.
            u_ante, u_match, d_ante, d_match = _precompute_rule_masks(data, rule)

            uplift_samples = np.empty(self.n_bootstrap)
            uplift_samples[:] = np.nan

            gain_samples: Optional[np.ndarray] = np.empty(self.n_bootstrap) if compute_gain else None
            if gain_samples is not None:
                gain_samples[:] = np.nan

            masks = (u_ante, u_match, d_ante, d_match)
            # The multinomial shortcut assumes disjoint antecedents; fall back
            # to row resampling for any rule where they overlap.
            use_multinomial = self.resample_mode == "multinomial" and n > 0 and not np.any(u_ante & d_ante)
            resample = self._resample_multinomial if use_multinomial else self._resample_rows
            n_undefined = resample(rng, masks, n, rule, uplift_samples, gain_samples, gain_args)

            # Compute full-dataset counts from pre-computed masks.
            # These are needed by both percentile and BCa branches.
            n_u_ante_full = int(u_ante.sum())
            n_u_match_full = int(u_match.sum())
            n_d_ante_full = int(d_ante.sum())
            n_d_match_full = int(d_match.sum())
            n_total_full = n

            if self.bootstrap_type == "bca":
                # BCa bootstrap CI (Efron, 1987).
                # The full-data estimate is used for bias correction (z0).
                original_uplift = compute_uplift_from_counts(
                    n_u_ante_full, n_u_match_full, n_d_ante_full, n_d_match_full, n_total_full
                )
                jack_uplift = self._jackknife_uplift(u_ante, u_match, d_ante, d_match, n)
                uplift_point, uplift_lower, uplift_upper, uplift_se = self._bca_ci(
                    uplift_samples, original_uplift, jack_uplift, level
                )

                gain_point: Optional[float] = None
                gain_lower: Optional[float] = None
                gain_upper: Optional[float] = None
                gain_se: Optional[float] = None
                clean_gain_samples: Optional[np.ndarray] = None

                if gain_samples is not None and column_values is not None:
                    # Guard against zero-support to avoid division by zero in
                    # gain computation for the original estimate.
                    if n_u_ante_full > 0 and n_d_ante_full > 0:
                        original_gain = compute_realistic_gain(
                            rule,
                            n_u_match_full / n_u_ante_full,
                            n_d_match_full / n_d_ante_full,
                            intrinsic_utility_table,
                            transition_utility_table,
                            column_values,
                        )
                    else:
                        original_gain = 0.0
                    jack_gain = self._jackknife_gain(
                        u_ante,
                        u_match,
                        d_ante,
                        d_match,
                        n,
                        rule,
                        intrinsic_utility_table,
                        transition_utility_table,
                        column_values,
                    )
                    gain_point, gain_lower, gain_upper, gain_se = self._bca_ci(
                        gain_samples, original_gain, jack_gain, level
                    )
                    valid_gain = gain_samples[~np.isnan(gain_samples)]
                    clean_gain_samples = valid_gain if valid_gain.size > 0 else None
                elif gain_samples is not None:
                    # gain_samples requested but column_values missing — skip gain.
                    valid_gain = gain_samples[~np.isnan(gain_samples)]
                    clean_gain_samples = valid_gain if valid_gain.size > 0 else None

                # valid_uplift needed for samples_uplift field.
                valid_uplift = uplift_samples[~np.isnan(uplift_samples)]

            else:
                # Percentile bootstrap (default).
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

                # Build optional gain statistics.
                gain_point = None
                gain_lower = None
                gain_upper = None
                gain_se = None
                clean_gain_samples = None

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

            # Support: number of transactions matching the undesired antecedent
            # (consistent with how the mining algorithm defines rule support).
            support = n_u_ante_full
            confidence = (n_u_match_full / n_u_ante_full) if n_u_ante_full > 0 else 0.0

            # Categorise the rule based on the uplift CI vs. zero threshold.
            # Guard against NaN bounds produced when no valid resamples exist.
            if np.isnan(uplift_lower) or np.isnan(uplift_upper):
                category = RuleCategory.UNSUPPORTED
            else:
                category = categorize_rule(uplift_lower, uplift_upper, threshold=0.0)

            undefined_fraction = n_undefined / self.n_bootstrap if self.n_bootstrap > 0 else 0.0
            if undefined_fraction > DEGENERATE_RESAMPLE_TOLERANCE:
                warnings.warn(
                    f"Rule {rule.rule_index}: {undefined_fraction:.1%} of bootstrap "
                    f"resamples were degenerate (n_u=0 or n_d=0) and were dropped. "
                    f"Interval is UNSUPPORTED because the undefined fraction exceeds 1%. "
                    f"Consider raising rule support or using analytic_type='auto'.",
                    UserWarning,
                    stacklevel=3,
                )

            if undefined_fraction > DEGENERATE_RESAMPLE_TOLERANCE:
                uplift_lower = uplift_upper = uplift_se = float('nan')
                if compute_gain:
                    gain_lower = gain_upper = gain_se = float('nan')
                category = RuleCategory.UNSUPPORTED

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
                    undefined_bootstrap_fraction=undefined_fraction,
                    samples_uplift=valid_uplift if valid_uplift.size > 0 else None,
                    samples_gain=clean_gain_samples,
                )
            )

        return results
