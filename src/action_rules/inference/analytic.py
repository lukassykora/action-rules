"""Analytic (closed-form) confidence interval engine for action rules.

Supports three interval types for binomial proportions:

- **Wald** — standard normal approximation (default).
- **Wilson** — Wilson score interval, more accurate for small samples
  or extreme proportions.
- **Auto** — selects Wilson or Wald per-rule based on sample size and
  proportion magnitude (Agresti & Coull, 1998).

All types use the delta method to propagate uncertainty from the two
independent Bernoulli confidences into the uplift measure.
"""

from math import sqrt
from typing import List, Optional

import pandas as pd

from .base import (
    ConfidenceIntervalResult,
    InferenceEngine,
    categorize_rule,
    compute_group_counts,
    compute_realistic_gain,
)


class AnalyticEngine(InferenceEngine):
    """Analytic CI engine supporting Wald and Wilson score intervals.

    For each action rule the engine computes closed-form confidence intervals
    for the uplift measure (and optionally for the realistic rule gain) by
    applying the delta method to propagate variance from the two independent
    Bernoulli confidence estimates.

    Notes
    -----
    **Wald** variance of the uplift is derived as follows (delta method):

    Let ``p_u = conf_u`` and ``p_d = conf_d`` be two independent Bernoulli
    proportions.  The uplift is:

    .. math::

        \\text{uplift} = (p_d + p_u - 1) \\cdot \\frac{n_u}{N}

    where :math:`n_u` is the undesired antecedent count and :math:`N` the
    total dataset size.  Variance (first-order delta method):

    .. math::

        \\text{Var}(p_u) &= \\frac{p_u(1-p_u)}{n_u} \\\\
        \\text{Var}(p_d) &= \\frac{p_d(1-p_d)}{n_d} \\\\
        \\text{Var}(\\text{uplift}) &= \\left(\\frac{n_u}{N}\\right)^2
            \\left(\\text{Var}(p_u) + \\text{Var}(p_d)\\right)

    **Wilson** score interval replaces the raw proportions with continuity-
    corrected Wilson centers and their corresponding SEs before applying the
    same delta-method formula.  Wilson is more accurate for small samples or
    extreme proportions (Brown et al., 2001).

    The final interval is in both cases:

    .. math::

        \\text{uplift} \\pm z_{1-\\alpha/2} \\cdot \\text{SE}(\\text{uplift})

    For gain uncertainty only the target-gain component is stochastic
    (``rule_gain`` is deterministic given the utility tables), so:

    .. math::

        \\text{SE}(\\text{gain}) = |\\text{target\\_gain}| \\cdot
            \\sqrt{\\text{Var}(p_u) + \\text{Var}(p_d)}
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, analytic_type: str = "wald") -> None:
        """Initialise the engine with the chosen interval type.

        Parameters
        ----------
        analytic_type : str, optional
            Which closed-form interval to use.  One of:

            ``'wald'``
                Standard Wald (normal approximation) interval.  Default and
                backward-compatible choice.
            ``'wilson'``
                Wilson score interval.  More accurate for small samples or
                when proportions are near 0 or 1.
            ``'auto'``
                Selects Wilson when ``n < 40`` or ``p < 0.05`` / ``p > 0.95``
                for either side; falls back to Wald otherwise.

        Raises
        ------
        ValueError
            If *analytic_type* is not one of the recognised values.
        """
        valid = {"wald", "wilson", "auto"}
        if analytic_type not in valid:
            raise ValueError(f"Unknown analytic_type '{analytic_type}'. Choose from {valid}.")
        self.analytic_type = analytic_type

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wilson_ci(x: float, n: float, z: float):
        """Wilson score interval for a binomial proportion.

        Computes the Wilson-adjusted centre and the corresponding
        half-width expressed as a standard-error equivalent so that the
        result can be used in the same delta-method formula as the Wald SE.

        Parameters
        ----------
        x : float
            Number of successes.
        n : float
            Number of trials.
        z : float
            Critical value (e.g. 1.96 for 95 % two-sided).

        Returns
        -------
        p_tilde : float
            Wilson score centre (shifted proportion).
        se_wilson : float
            Half-width divided by *z*, analogous to a standard error.

        Notes
        -----
        Wilson score interval (Wilson, 1927):

        .. math::

            \\tilde{p} = \\frac{x + z^2/2}{n + z^2}

        .. math::

            w = \\frac{z \\sqrt{n}}{n + z^2}
                \\sqrt{\\hat{p}(1-\\hat{p}) + \\frac{z^2}{4n}}
        """
        p_hat = x / n
        denom = n + z**2
        p_tilde = (x + z**2 / 2.0) / denom
        w = z * sqrt(n) / denom * sqrt(p_hat * (1.0 - p_hat) + z**2 / (4.0 * n))
        se_wilson = w / z if z > 0 else 0.0
        return p_tilde, se_wilson

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
        """Compute analytic confidence intervals for each action rule.

        The interval type (Wald / Wilson / auto) is determined by
        ``self.analytic_type`` set at construction time.

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

        Raises
        ------
        ImportError
            If ``scipy`` is not installed.

        Notes
        -----
        Rules with zero antecedent support (undesired or desired side) return
        ``NaN`` for all interval bounds and the SE, matching the behaviour of
        :class:`~action_rules.inference.bootstrap.BootstrapEngine`.
        """
        # Lazy import so the rest of the package does not hard-depend on scipy.
        try:
            from scipy.stats import norm
        except ImportError:
            raise ImportError("scipy is required for analytic CI computation. Install it with: pip install scipy")

        compute_gain = (intrinsic_utility_table is not None) or (transition_utility_table is not None)

        alpha = 1.0 - confidence_level
        # z-score for two-sided interval (e.g. 1.96 for 95 %).
        z = norm.ppf(1.0 - alpha / 2.0)

        results: List[ConfidenceIntervalResult] = []

        for rule in rules:
            n_u_ante, n_u_match, n_d_ante, n_d_match, n_total = compute_group_counts(data, rule)

            # Guard: degenerate case — no antecedent support on either side.
            if n_u_ante == 0 or n_d_ante == 0 or n_total == 0:
                nan = float('nan')
                support = n_u_ante
                confidence = 0.0

                gain_point: Optional[float] = nan if compute_gain else None
                gain_lower: Optional[float] = nan if compute_gain else None
                gain_upper: Optional[float] = nan if compute_gain else None
                gain_se: Optional[float] = nan if compute_gain else None

                results.append(
                    ConfidenceIntervalResult(
                        rule_index=rule.rule_index,
                        method='analytic',
                        confidence_level=confidence_level,
                        uplift_point=nan,
                        uplift_ci_lower=nan,
                        uplift_ci_upper=nan,
                        uplift_se=nan,
                        realistic_rule_gain_point=gain_point,
                        realistic_rule_gain_ci_lower=gain_lower,
                        realistic_rule_gain_ci_upper=gain_upper,
                        realistic_rule_gain_se=gain_se,
                        support=support,
                        confidence=confidence,
                        category=None,
                        samples_uplift=None,
                        samples_gain=None,
                    )
                )
                continue

            # Bernoulli confidence estimates.
            p_u = n_u_match / n_u_ante
            p_d = n_d_match / n_d_ante

            scale = n_u_ante / n_total

            # Decide which variant to use for this rule.
            use_wilson = self.analytic_type == "wilson" or (
                self.analytic_type == "auto"
                and (n_u_ante < 40 or n_d_ante < 40 or p_u < 0.05 or p_u > 0.95 or p_d < 0.05 or p_d > 0.95)
            )

            if use_wilson:
                # Wilson score interval (Wilson, 1927); more accurate for small
                # samples or extreme proportions (Brown et al., 2001).
                p_tilde_u, se_wu = self._wilson_ci(n_u_match, n_u_ante, z)
                p_tilde_d, se_wd = self._wilson_ci(n_d_match, n_d_ante, z)
                d = p_tilde_d + p_tilde_u - 1.0
                uplift_point = d * scale
                var_d = se_wu**2 + se_wd**2
                se_uplift = scale * sqrt(var_d)
            else:
                # Uplift point estimate (Ras et al., 2009):
                # d = p_d - (1 - p_u) = p_d + p_u - 1
                # uplift = d * n_u_ante / n_total
                d = p_d + p_u - 1.0
                uplift_point = d * scale

                # Delta-method variance for uplift.
                # Var(p_u) = p_u(1-p_u)/n_u_ante,  Var(p_d) = p_d(1-p_d)/n_d_ante
                # Var(d)   = Var(p_u) + Var(p_d)  (independent)
                # Var(uplift) = (n_u_ante/n_total)^2 * Var(d)
                var_p_u = p_u * (1.0 - p_u) / n_u_ante
                var_p_d = p_d * (1.0 - p_d) / n_d_ante
                var_d = var_p_u + var_p_d
                var_uplift = scale**2 * var_d
                se_uplift = sqrt(var_uplift)

            # Symmetric interval around the point estimate (same for both types).
            uplift_lower = uplift_point - z * se_uplift
            uplift_upper = uplift_point + z * se_uplift

            # Support and confidence from the full dataset.
            support = n_u_ante
            confidence = p_u

            category = categorize_rule(uplift_lower, uplift_upper, threshold=0.0)

            # Optional gain statistics.
            if compute_gain and column_values is not None:
                # Use Wilson-adjusted proportions when Wilson is active so the
                # gain point estimate is consistent with the uplift CI centre.
                gain_p_u = p_tilde_u if use_wilson else p_u
                gain_p_d = p_tilde_d if use_wilson else p_d
                gain_point = compute_realistic_gain(
                    rule,
                    gain_p_u,
                    gain_p_d,
                    intrinsic_utility_table,
                    transition_utility_table,
                    column_values,
                )

                # Only the target-gain component is stochastic; rule_gain is
                # a deterministic function of the utility tables.
                intrinsic = intrinsic_utility_table or {}
                transition = transition_utility_table or {}

                target_attr = rule.target_attribute
                target_u_val = rule.target_undesired
                target_d_val = rule.target_desired

                u_target_undesired = intrinsic.get((target_attr, target_u_val), 0.0)
                u_target_desired = intrinsic.get((target_attr, target_d_val), 0.0)
                trans_target = transition.get((target_attr, target_u_val, target_d_val), 0.0)
                target_gain = u_target_desired - u_target_undesired + trans_target

                # SE of gain = |target_gain| * sqrt(Var(d))  (delta method on gain).
                # var_d is set by whichever branch (Wald/Wilson) ran above.
                se_gain = abs(target_gain) * sqrt(var_d)

                gain_lower = gain_point - z * se_gain
                gain_upper = gain_point + z * se_gain
                gain_se: Optional[float] = se_gain
            else:
                gain_point = None
                gain_lower = None
                gain_upper = None
                gain_se = None

            results.append(
                ConfidenceIntervalResult(
                    rule_index=rule.rule_index,
                    method='analytic',
                    confidence_level=confidence_level,
                    uplift_point=uplift_point,
                    uplift_ci_lower=uplift_lower,
                    uplift_ci_upper=uplift_upper,
                    uplift_se=se_uplift,
                    realistic_rule_gain_point=gain_point,
                    realistic_rule_gain_ci_lower=gain_lower,
                    realistic_rule_gain_ci_upper=gain_upper,
                    realistic_rule_gain_se=gain_se,
                    support=support,
                    confidence=confidence,
                    category=category,
                    samples_uplift=None,
                    samples_gain=None,
                )
            )

        return results
