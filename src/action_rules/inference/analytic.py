"""Analytic (closed-form) confidence interval engine for action rules.

Supports three interval types for binomial proportions:

- **Wald** — standard normal approximation (default).
- **Wilson / Newcombe-Wilson** — Wilson score interval per side combined via
  the Newcombe construction (Newcombe, 1998) for the unscaled rule contrast
  :math:`\\delta = p_d + p_u - 1`.  More accurate for small samples or
  extreme proportions; produces an *asymmetric* interval for the rule metric.
- **Auto** — selects Newcombe-Wilson when either side is small
  (``n < 40``) or extreme (``p < 0.05`` or ``p > 0.95``); falls back to Wald.

Both branches transform an interval for :math:`\\delta` to the requested
metric: multiply by ``n_u/N`` for population-scaled uplift, or apply
``C_flex + Δu_target · δ`` for the per-matched-customer realistic rule gain.

References
----------
- Wilson, E. B. (1927). Probable inference, the law of succession, and
  statistical inference. *J. Amer. Statist. Assoc.*, 22(158), 209-212.
- Newcombe, R. G. (1998). Interval estimation for the difference between
  independent proportions: comparison of eleven methods. *Statist. Med.*,
  17(8), 873-890.
- Agresti, A. & Coull, B. A. (1998). Approximate is better than exact for
  interval estimation of binomial proportions. *Amer. Statist.*, 52(2),
  119-126.
"""

from math import sqrt
from typing import List, Optional, Tuple

import pandas as pd

from .base import (
    ConfidenceIntervalResult,
    InferenceEngine,
    categorize_rule,
    compute_group_counts,
    compute_realistic_gain,
)


class AnalyticEngine(InferenceEngine):
    """Analytic CI engine supporting Wald and Newcombe-Wilson intervals.

    For each action rule the engine computes closed-form confidence intervals
    for the uplift measure (and optionally for the realistic rule gain).

    Notes
    -----
    Let :math:`p_u = \\hat{p}_u` and :math:`p_d = \\hat{p}_d` be the two
    independent Bernoulli MLE proportions.  Define the unscaled rule contrast

    .. math::

        \\delta = p_d + p_u - 1.

    **Wald** branch propagates the variance of :math:`\\delta` (sum of two
    independent binomials) via the delta method:

    .. math::

        \\text{Var}(\\delta) = \\frac{p_u(1-p_u)}{n_u} + \\frac{p_d(1-p_d)}{n_d}

    and produces a symmetric interval :math:`\\delta \\pm z\\sqrt{\\text{Var}(\\delta)}`.

    **Newcombe-Wilson** branch (Newcombe, 1998, method 10) computes single-
    proportion Wilson intervals :math:`[L_u, U_u]` and :math:`[L_d, U_d]`
    and combines them as

    .. math::

        L_{\\delta} = \\hat\\delta - \\sqrt{(p_d - L_d)^2 + (p_u - L_u)^2}, \\\\
        U_{\\delta} = \\hat\\delta + \\sqrt{(U_d - p_d)^2 + (U_u - p_u)^2}.

    This interval is generally asymmetric and is correct for the sum of two
    independent proportions even when sample sizes are small.

    For both branches the :math:`\\delta` interval is linearly transformed:

    - **Population-scaled uplift**: multiply by ``scale = n_u / N``.
    - **Realistic rule gain**: ``C_flex + Δu_target · δ`` (where ``C_flex``
      is the deterministic flexible/transition utility component).  If
      ``Δu_target < 0`` the transformed lower / upper bounds are swapped.
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
            ``'newcombe_wilson'`` (preferred) or ``'wilson'`` (alias)
                Newcombe-Wilson interval for the unscaled rule contrast
                :math:`\\delta = p_d + p_u - 1` (Newcombe, 1998), built by
                combining two single-proportion Wilson score intervals.
                More accurate for small samples or proportions near 0 or 1.
                ``'wilson'`` is retained as a backward-compatible alias and
                resolves to the same Newcombe-Wilson construction.
            ``'auto'``
                Selects the Newcombe-Wilson branch when either side has
                ``n < 40`` or ``p < 0.05`` / ``p > 0.95``; falls back to
                Wald otherwise (Agresti & Coull, 1998).

        Raises
        ------
        ValueError
            If *analytic_type* is not one of the recognised values.
        """
        valid = {"wald", "wilson", "newcombe_wilson", "auto"}
        if analytic_type not in valid:
            raise ValueError(f"Unknown analytic_type '{analytic_type}'. Choose from {valid}.")
        # Canonicalise: 'wilson' is a backward-compatible alias for
        # 'newcombe_wilson'.  Both invoke the same Newcombe-Wilson code path.
        self.analytic_type = "newcombe_wilson" if analytic_type == "wilson" else analytic_type

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wilson_bounds(x: float, n: float, z: float) -> Tuple[float, float, float]:
        """Single-proportion Wilson score interval (Wilson, 1927).

        Returns the MLE point estimate together with the Wilson lower and
        upper bounds, clipped to ``[0, 1]``.

        Parameters
        ----------
        x : float
            Number of successes.
        n : float
            Number of trials.
        z : float
            Critical value (e.g. 1.96 for a two-sided 95 % interval).

        Returns
        -------
        p_hat : float
            Maximum-likelihood proportion ``x / n``.
        lower : float
            Wilson lower bound.
        upper : float
            Wilson upper bound.

        Notes
        -----
        .. math::

            L,U = \\frac{\\hat p + z^2/(2n) \\mp z \\sqrt{\\hat p(1-\\hat p)/n + z^2/(4n^2)}}{1 + z^2/n}
        """
        if n <= 0:
            nan = float('nan')
            return nan, nan, nan
        p_hat = x / n
        denom = 1.0 + (z * z) / n
        centre = p_hat + (z * z) / (2.0 * n)
        radius = z * sqrt(p_hat * (1.0 - p_hat) / n + (z * z) / (4.0 * n * n))
        lower = max(0.0, (centre - radius) / denom)
        upper = min(1.0, (centre + radius) / denom)
        return p_hat, lower, upper

    @staticmethod
    def _newcombe_delta_interval(
        p_u: float,
        l_u: float,
        u_u: float,
        p_d: float,
        l_d: float,
        u_d: float,
    ) -> Tuple[float, float, float]:
        """Newcombe-Wilson interval for delta = p_d + p_u - 1.

        Combines two single-proportion Wilson intervals using Newcombe's
        method-10 construction for the difference of independent proportions,
        adapted to the *sum* :math:`p_d + p_u - 1` via the identity
        :math:`p_d + p_u - 1 = p_d - (1 - p_u)`.

        Parameters
        ----------
        p_u, l_u, u_u : float
            Undesired-side point estimate and Wilson lower / upper bounds.
        p_d, l_d, u_d : float
            Desired-side point estimate and Wilson lower / upper bounds.

        Returns
        -------
        delta_hat : float
            Point estimate :math:`p_d + p_u - 1`.
        lower : float
            Newcombe-Wilson lower bound.
        upper : float
            Newcombe-Wilson upper bound.
        """
        delta_hat = p_d + p_u - 1.0
        lower = delta_hat - sqrt((p_d - l_d) ** 2 + (p_u - l_u) ** 2)
        upper = delta_hat + sqrt((u_d - p_d) ** 2 + (u_u - p_u) ** 2)
        return delta_hat, lower, upper

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

        The interval type (Wald / Newcombe-Wilson / auto) is determined by
        ``self.analytic_type`` set at construction time.

        Parameters
        ----------
        data : pd.DataFrame
            The full original dataset.
        rules : list
            List of :class:`~action_rules.inference.base.RuleMasks` objects.
        confidence_level : float, optional
            Nominal coverage probability.  Default ``0.95``.
        intrinsic_utility_table : dict, optional
            ``(attribute, value) -> float`` mapping.  When ``None``, gain
            statistics are omitted.
        transition_utility_table : dict, optional
            ``(attribute, from_value, to_value) -> float`` mapping.
        column_values : dict, optional
            ``int -> (attribute, value)`` mapping from ``Output.column_values``.

        Returns
        -------
        List[ConfidenceIntervalResult]
            One result per rule, in the same order.

        Raises
        ------
        ImportError
            If ``scipy`` is not installed.

        Notes
        -----
        For the Newcombe-Wilson branch the reported ``uplift_se`` field is
        the average half-width divided by ``z``: an SE-equivalent for an
        asymmetric interval.  Endpoint asymmetry is preserved in
        ``uplift_ci_lower`` and ``uplift_ci_upper``.
        """
        try:
            from scipy.stats import norm
        except ImportError:
            raise ImportError("scipy is required for analytic CI computation. Install it with: pip install scipy")

        compute_gain = (intrinsic_utility_table is not None) or (transition_utility_table is not None)

        alpha = 1.0 - confidence_level
        z = norm.ppf(1.0 - alpha / 2.0)

        results: List[ConfidenceIntervalResult] = []

        for rule in rules:
            n_u_ante, n_u_match, n_d_ante, n_d_match, n_total = compute_group_counts(data, rule)

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

            p_u = n_u_match / n_u_ante
            p_d = n_d_match / n_d_ante
            scale = n_u_ante / n_total
            delta_hat = p_d + p_u - 1.0

            # Select branch.
            use_wilson = self.analytic_type == "newcombe_wilson" or (
                self.analytic_type == "auto"
                and (n_u_ante < 40 or n_d_ante < 40 or p_u < 0.05 or p_u > 0.95 or p_d < 0.05 or p_d > 0.95)
            )

            if use_wilson:
                # Newcombe (1998) combination of single-proportion Wilson intervals.
                _, l_u, u_u = self._wilson_bounds(n_u_match, n_u_ante, z)
                _, l_d, u_d = self._wilson_bounds(n_d_match, n_d_ante, z)
                _, delta_lower, delta_upper = self._newcombe_delta_interval(
                    p_u, l_u, u_u, p_d, l_d, u_d
                )
            else:
                # Wald delta-method interval for delta = p_d + p_u - 1.
                var_p_u = p_u * (1.0 - p_u) / n_u_ante
                var_p_d = p_d * (1.0 - p_d) / n_d_ante
                var_delta = var_p_u + var_p_d
                delta_se = sqrt(var_delta)
                delta_lower = delta_hat - z * delta_se
                delta_upper = delta_hat + z * delta_se

            # Transform delta -> population-scaled uplift.
            uplift_point = delta_hat * scale
            uplift_lower = delta_lower * scale
            uplift_upper = delta_upper * scale
            # Average half-width divided by z gives an SE-equivalent for an
            # asymmetric interval; reduces to the exact Wald SE in the Wald
            # branch.
            se_uplift = (uplift_upper - uplift_lower) / (2.0 * z) if z > 0 else 0.0

            support = n_u_ante
            confidence = p_u
            category = categorize_rule(uplift_lower, uplift_upper, threshold=0.0)

            # Optional realistic rule gain via the same delta interval.
            if compute_gain and column_values is not None:
                # Point estimate uses MLE proportions (consistent with Wald).
                gain_point = compute_realistic_gain(
                    rule,
                    p_u,
                    p_d,
                    intrinsic_utility_table,
                    transition_utility_table,
                    column_values,
                )

                intrinsic = intrinsic_utility_table or {}
                transition = transition_utility_table or {}
                target_attr = rule.target_attribute
                target_u_val = rule.target_undesired
                target_d_val = rule.target_desired
                u_target_undesired = intrinsic.get((target_attr, target_u_val), 0.0)
                u_target_desired = intrinsic.get((target_attr, target_d_val), 0.0)
                trans_target = transition.get((target_attr, target_u_val, target_d_val), 0.0)
                target_gain = u_target_desired - u_target_undesired + trans_target

                # gain = C_flex + target_gain * delta. Apply target_gain to
                # the delta endpoints; swap if target_gain is negative so
                # that lower <= upper is preserved.
                c_flex = gain_point - target_gain * delta_hat
                end_lo = c_flex + target_gain * delta_lower
                end_hi = c_flex + target_gain * delta_upper
                gain_lower = min(end_lo, end_hi)
                gain_upper = max(end_lo, end_hi)
                gain_se: Optional[float] = (gain_upper - gain_lower) / (2.0 * z) if z > 0 else 0.0
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
