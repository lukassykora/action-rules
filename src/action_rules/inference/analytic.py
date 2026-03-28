"""Analytic (closed-form) confidence interval engine for action rules.

Uses the Wald normal approximation with the delta method to propagate
uncertainty from the two Bernoulli confidences into the uplift measure.
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
    """Wald normal-approximation CI engine using the delta method.

    For each action rule the engine computes closed-form confidence intervals
    for the uplift measure (and optionally for the realistic rule gain) by
    applying the delta method to propagate variance from the two independent
    Bernoulli confidence estimates.

    Notes
    -----
    The variance of the uplift is derived as follows (delta method):

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

    The interval is then:

    .. math::

        \\text{uplift} \\pm z_{1-\\alpha/2} \\cdot \\text{SE}(\\text{uplift})

    This engine is stateless and can be reused across multiple datasets
    with the same instance.

    For gain uncertainty only the target-gain component is stochastic
    (``rule_gain`` is deterministic given the utility tables), so:

    .. math::

        \\text{SE}(\\text{gain}) = |\\text{target\\_gain}| \\cdot
            \\sqrt{\\text{Var}(p_u) + \\text{Var}(p_d)}
    """

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
        """Compute analytic (Wald) confidence intervals for each action rule.

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
            raise ImportError(
                "scipy is required for analytic CI computation. Install it with: pip install scipy"
            )

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

            # Uplift point estimate (Ras et al., 2009):
            # d = p_d - (1 - p_u) = p_d + p_u - 1
            # uplift = d * n_u_ante / n_total
            d = p_d + p_u - 1.0
            uplift_point = d * n_u_ante / n_total

            # Delta-method variance for uplift.
            # Var(p_u) = p_u(1-p_u)/n_u_ante,  Var(p_d) = p_d(1-p_d)/n_d_ante
            # Var(d)   = Var(p_u) + Var(p_d)  (independent)
            # Var(uplift) = (n_u_ante/n_total)^2 * Var(d)
            var_p_u = p_u * (1.0 - p_u) / n_u_ante
            var_p_d = p_d * (1.0 - p_d) / n_d_ante
            var_d = var_p_u + var_p_d
            scale = n_u_ante / n_total
            var_uplift = scale**2 * var_d
            se_uplift = sqrt(var_uplift)

            # Wald interval (symmetric around the point estimate).
            uplift_lower = uplift_point - z * se_uplift
            uplift_upper = uplift_point + z * se_uplift

            # Support and confidence from the full dataset.
            support = n_u_ante
            confidence = p_u

            category = categorize_rule(uplift_lower, uplift_upper, threshold=0.0)

            # Optional gain statistics.
            if compute_gain and column_values is not None:
                gain_point = compute_realistic_gain(
                    rule,
                    p_u,
                    p_d,
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

                # SE of gain = |target_gain| * sqrt(Var(d))  (delta method on gain)
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
