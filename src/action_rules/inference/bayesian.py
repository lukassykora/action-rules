"""Bayesian credible interval engine for action rules.

Uses a Beta-Binomial conjugate model: the two Bernoulli confidences
(undesired and desired) each receive independent Beta priors.  Monte Carlo
samples are drawn from the resulting Beta posteriors and composed into
posterior distributions for the uplift and (optionally) the realistic rule
gain.  Equal-tailed credible intervals are reported as percentiles of the
posterior samples.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import (
    ConfidenceIntervalResult,
    InferenceEngine,
    categorize_rule,
    compute_group_counts,
)


class BayesianEngine(InferenceEngine):
    """Beta-Binomial conjugate model with Monte Carlo posterior sampling.

    For each action rule the engine updates a Beta prior with the observed
    success counts for the undesired and desired classification rules and
    draws *n_mc* samples from the resulting posteriors.  The posterior
    samples are then composed into a distribution over the uplift (and
    optionally the realistic rule gain), from which equal-tailed credible
    intervals are extracted as percentiles.

    Parameters
    ----------
    n_mc : int, optional
        Number of Monte Carlo draws from the posterior.  Default ``10000``.
    prior_alpha : float, optional
        Alpha parameter of the symmetric Beta prior.  Default ``1.0``
        (flat/uniform prior — equivalent to Bayes-Laplace smoothing).
    prior_beta : float, optional
        Beta parameter of the symmetric Beta prior.  Default ``1.0``.
    random_state : int, optional
        Seed for :func:`numpy.random.default_rng`, enabling reproducible
        results.  ``None`` (default) uses a randomly initialised generator.

    Notes
    -----
    The Beta posterior parameters follow the standard conjugate update
    (Bernoulli likelihood with Beta prior):

    .. math::

        \\alpha_u^* = \\alpha_0 + n_{u,\\text{match}}, \\quad
        \\beta_u^* = \\beta_0 + (n_{u,\\text{ante}} - n_{u,\\text{match}})

    and analogously for the desired side.

    When *n_u_ante* or *n_d_ante* is zero, no likelihood information is
    available and the posterior equals the prior.  Samples are still drawn
    (a ``Beta(1, 1)`` with the default flat prior), resulting in very wide
    intervals that correctly reflect the lack of evidence.

    The point estimate reported is the posterior mean of the uplift samples
    (not the analytic posterior mean of each Beta), which is consistent with
    the bootstrap engine's convention of reporting the mean of the sampling
    distribution surrogate.

    The SE reported is the standard deviation of the posterior samples
    (``ddof=1``).

    Scipy is **not** required; all sampling uses :mod:`numpy`'s built-in
    ``rng.beta``.
    """

    def __init__(
        self,
        n_mc: int = 10000,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_mc = n_mc
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
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
        """Compute Bayesian credible intervals for each action rule.

        Parameters
        ----------
        data : pd.DataFrame
            The full original dataset.  Each row is one transaction; columns
            correspond to attribute names as strings.
        rules : list
            List of :class:`~action_rules.inference.base.RuleMasks` objects
            produced by :func:`~action_rules.inference.base.extract_rule_masks`.
        confidence_level : float, optional
            Desired nominal coverage probability.  Default ``0.95``.
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
        The credible interval is computed as the equal-tailed percentile
        interval of the Monte Carlo posterior samples::

            ci_lower = percentile(samples, 100 * alpha / 2)
            ci_upper = percentile(samples, 100 * (1 - alpha / 2))

        where ``alpha = 1 - confidence_level``.
        """
        compute_gain = (intrinsic_utility_table is not None) or (transition_utility_table is not None)

        # New-style RNG for reproducibility (numpy >= 1.17).
        rng = np.random.default_rng(self.random_state)

        alpha = 1.0 - confidence_level
        pct_lower = 100.0 * alpha / 2.0
        pct_upper = 100.0 * (1.0 - alpha / 2.0)

        results: List[ConfidenceIntervalResult] = []

        for rule in rules:
            n_u_ante, n_u_match, n_d_ante, n_d_match, n_total = compute_group_counts(data, rule)

            # ------------------------------------------------------------------
            # Beta posterior parameters (conjugate update).
            # When antecedent count is zero the posterior equals the prior —
            # no observation to update from, so we draw from the prior directly.
            # ------------------------------------------------------------------
            # Beta posterior for p_u (undesired confidence).
            alpha_u = self.prior_alpha + n_u_match
            beta_u = self.prior_beta + max(n_u_ante - n_u_match, 0)

            # Beta posterior for p_d (desired confidence).
            alpha_d = self.prior_alpha + n_d_match
            beta_d = self.prior_beta + max(n_d_ante - n_d_match, 0)

            # Draw MC samples from the two independent posteriors.
            p_u_draws = rng.beta(alpha_u, beta_u, size=self.n_mc)
            p_d_draws = rng.beta(alpha_d, beta_d, size=self.n_mc)

            # Compose uplift posterior samples.
            # d = p_d - (1 - p_u) = p_d + p_u - 1   (Ras et al., 2009)
            # uplift = d * n_u_ante / n_total
            d_draws = p_u_draws + p_d_draws - 1.0
            # When n_u_ante is 0 the scale factor is 0; uplift collapses to 0.
            uplift_draws = d_draws * (n_u_ante / n_total) if n_total > 0 else np.zeros(self.n_mc)

            # Posterior summary statistics for uplift.
            uplift_point = float(np.mean(uplift_draws))
            uplift_lower = float(np.percentile(uplift_draws, pct_lower))
            uplift_upper = float(np.percentile(uplift_draws, pct_upper))
            uplift_se = float(np.std(uplift_draws, ddof=1))

            # Support and confidence from the full dataset (for reporting).
            support = n_u_ante
            confidence = (n_u_match / n_u_ante) if n_u_ante > 0 else 0.0

            category = categorize_rule(uplift_lower, uplift_upper, threshold=0.0)

            # ------------------------------------------------------------------
            # Optional gain statistics.
            # rule_gain is a deterministic constant given the utility tables.
            # The only stochastic component is target_gain * d, where d is drawn
            # from the posterior.
            # ------------------------------------------------------------------
            gain_point: Optional[float] = None
            gain_lower: Optional[float] = None
            gain_upper: Optional[float] = None
            gain_se: Optional[float] = None
            gain_draws: Optional[np.ndarray] = None

            if compute_gain and column_values is not None:
                intrinsic = intrinsic_utility_table or {}
                transition = transition_utility_table or {}

                # Accumulate deterministic part: intrinsic utility difference
                # and transition costs for all items in the itemsets.
                u_undesired = 0.0
                for idx in rule.undesired_itemset:
                    attr, val = column_values[idx]
                    u_undesired += intrinsic.get((attr, val), 0.0)

                u_desired = 0.0
                for idx in rule.desired_itemset:
                    attr, val = column_values[idx]
                    u_desired += intrinsic.get((attr, val), 0.0)

                transition_gain = 0.0
                for u_idx, d_idx in zip(rule.undesired_itemset, rule.desired_itemset):
                    if u_idx != d_idx:
                        u_attr, u_val = column_values[u_idx]
                        _d_attr, d_val = column_values[d_idx]
                        transition_gain += transition.get((u_attr, u_val, d_val), 0.0)

                # rule_gain is the same constant for every MC draw.
                rule_gain = u_desired - u_undesired + transition_gain

                # Target utilities — deterministic constants.
                target_attr = rule.target_attribute
                target_u_val = rule.target_undesired
                target_d_val = rule.target_desired

                u_tgt_u = intrinsic.get((target_attr, target_u_val), 0.0)
                u_tgt_d = intrinsic.get((target_attr, target_d_val), 0.0)
                trans_tgt = transition.get((target_attr, target_u_val, target_d_val), 0.0)
                target_gain = u_tgt_d - u_tgt_u + trans_tgt

                # Vectorised over MC draws:
                # realistic_rule_gain = rule_gain + d * target_gain
                gain_draws = rule_gain + d_draws * target_gain

                gain_point = float(np.mean(gain_draws))
                gain_lower = float(np.percentile(gain_draws, pct_lower))
                gain_upper = float(np.percentile(gain_draws, pct_upper))
                gain_se = float(np.std(gain_draws, ddof=1))

            results.append(
                ConfidenceIntervalResult(
                    rule_index=rule.rule_index,
                    method='bayesian',
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
                    samples_uplift=uplift_draws,
                    samples_gain=gain_draws,
                )
            )

        return results
