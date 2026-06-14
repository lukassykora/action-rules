"""Synthetic-data coverage simulation for action-rule confidence intervals.

The simulation uses a known categorical data-generating process (DGP) so the
*true* uplift of every mined rule can be computed analytically.  Each CI
method's empirical 95 % coverage and mean width can then be measured by
repeatedly drawing fresh datasets, mining rules, computing CIs, and checking
whether each interval contains the true value.

Used by both the article notebook (``notebooks/article/03_coverage_simulation.ipynb``)
and a smoke test in ``tests/test_coverage_simulation.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from action_rules import ActionRules

# ---------------------------------------------------------------------------
# Data-generating process (DGP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DGPParams:
    """Parameters of the synthetic data-generating process.

    The target probability follows a logistic model:

        P(Y=1 | S1, S2, F1, F2) = sigmoid(
            intercept
            + s1_effect * 1{S1='1'}
            + s2_effect * 1{S2='B'}
            + f1_effect * 1{F1='y'}
            + f2_effect * 1{F2='v'}
            + interaction * 1{S1='1', F1='y'}
        )

    Attribute prevalences are independent Bernoulli draws with ``p_*`` rates.
    """

    intercept: float = -1.0
    s1_effect: float = 0.7
    s2_effect: float = 0.3
    f1_effect: float = 1.2
    f2_effect: float = 0.4
    interaction: float = 0.4
    p_s1: float = 0.5
    p_s2: float = 0.5
    p_f1: float = 0.5
    p_f2: float = 0.5

    def prob_y(self, s1: str, s2: str, f1: str, f2: str) -> float:
        """Return P(Y=1) for the given attribute combination under the logistic DGP."""
        z = self.intercept
        if s1 == '1':
            z += self.s1_effect
        if s2 == 'B':
            z += self.s2_effect
        if f1 == 'y':
            z += self.f1_effect
        if f2 == 'v':
            z += self.f2_effect
        if s1 == '1' and f1 == 'y':
            z += self.interaction
        return float(1.0 / (1.0 + np.exp(-z)))

    def joint_prob(self, s1: str, s2: str, f1: str, f2: str) -> float:
        """Return the joint prior probability of the given attribute combination."""
        p_s1 = self.p_s1 if s1 == '1' else 1 - self.p_s1
        p_s2 = self.p_s2 if s2 == 'B' else 1 - self.p_s2
        p_f1 = self.p_f1 if f1 == 'y' else 1 - self.p_f1
        p_f2 = self.p_f2 if f2 == 'v' else 1 - self.p_f2
        return p_s1 * p_s2 * p_f1 * p_f2


_S1_VALUES = ('0', '1')
_S2_VALUES = ('A', 'B')
_F1_VALUES = ('x', 'y')
_F2_VALUES = ('u', 'v')
_FLEX_PAIRS = [(f1, f2) for f1 in _F1_VALUES for f2 in _F2_VALUES]


def generate_dataset(n: int, params: DGPParams, seed: int = 0) -> pd.DataFrame:
    """Draw an i.i.d. dataset of ``n`` rows from the DGP."""
    rng = np.random.default_rng(seed)
    s1 = rng.choice(_S1_VALUES, size=n, p=[1 - params.p_s1, params.p_s1])
    s2 = rng.choice(_S2_VALUES, size=n, p=[1 - params.p_s2, params.p_s2])
    f1 = rng.choice(_F1_VALUES, size=n, p=[1 - params.p_f1, params.p_f1])
    f2 = rng.choice(_F2_VALUES, size=n, p=[1 - params.p_f2, params.p_f2])
    probs = np.array([params.prob_y(*row) for row in zip(s1, s2, f1, f2)])
    y = (rng.random(n) < probs).astype(int).astype(str)
    return pd.DataFrame({'S1': s1, 'S2': s2, 'F1': f1, 'F2': f2, 'Y': y})


# ---------------------------------------------------------------------------
# True uplift of a mined rule
# ---------------------------------------------------------------------------


def _decode_itemset(itemset: Sequence[int], column_values: Dict[int, Tuple[str, str]]) -> Dict[str, str]:
    return {col: val for col, val in (column_values[i] for i in itemset)}


def true_uplift(action_rule: dict, column_values: dict, params: DGPParams) -> float:
    """Analytic uplift of a rule under the known DGP.

    The classical uplift formula (Ras et al., 2009) is::

        d = P(Y=desired | desired_antecedent) + P(Y=undesired | undesired_antecedent) - 1
        uplift = d * P(undesired_antecedent)

    Both probabilities are evaluated from the DGP itself (which is the
    asymptotic limit of the sample-estimated quantities).  ``desired_antecedent``
    differs from ``undesired_antecedent`` only in the flexible attributes; the
    stable attributes are shared.  The function therefore looks up the
    flexible values from each side of the action rule.
    """
    undesired_part = action_rule['undesired']
    desired_part = action_rule['desired']

    undesired_decoded = _decode_itemset(undesired_part['itemset'], column_values)
    desired_decoded = _decode_itemset(desired_part['itemset'], column_values)
    target_attr, target_undesired = column_values[undesired_part['target']]
    _, target_desired = column_values[desired_part['target']]
    assert target_attr == 'Y'

    # Stable conditions are attributes that are in both decoded itemsets with
    # the same value.  Flexible attributes are those whose value differs.
    stable: Dict[str, str] = {}
    flex_undesired: Dict[str, str] = {}
    flex_desired: Dict[str, str] = {}
    for attr, val in undesired_decoded.items():
        if attr in desired_decoded and desired_decoded[attr] == val:
            stable[attr] = val
        else:
            flex_undesired[attr] = val
    for attr, val in desired_decoded.items():
        if attr not in stable:
            flex_desired[attr] = val

    # Build the marginalisation set: we need to integrate over any DGP
    # attribute (S1, S2, F1, F2) not pinned by either ``stable`` or the
    # corresponding flexible side.  We always know the rule's S* values
    # (they're stable) so the only attributes that might be unspecified are
    # flexible attributes that the rule didn't mention.
    all_attrs = {'S1', 'S2', 'F1', 'F2'}
    free_attrs = sorted(all_attrs - set(stable.keys()) - set(flex_undesired.keys()))

    def _enumerate(side: Dict[str, str]) -> List[Dict[str, str]]:
        states = [dict(stable, **side)]
        for attr in free_attrs:
            new_states = []
            values = (
                _F1_VALUES
                if attr == 'F1'
                else _F2_VALUES if attr == 'F2' else (_S1_VALUES if attr == 'S1' else _S2_VALUES)
            )
            for st in states:
                for v in values:
                    new_states.append(dict(st, **{attr: v}))
            states = new_states
        return states

    undesired_states = _enumerate(flex_undesired)
    desired_states = _enumerate(flex_desired)

    # P(undesired_antecedent) = sum over states of joint_prob(state)
    # P(Y=undesired | undesired_antecedent) = sum P(state) * P(Y=undesired | state) / P(undesired_antecedent)
    def _marginal_p_y(states: List[Dict[str, str]], y_target: str) -> Tuple[float, float]:
        total = 0.0
        weighted = 0.0
        for st in states:
            jp = params.joint_prob(st['S1'], st['S2'], st['F1'], st['F2'])
            py = params.prob_y(st['S1'], st['S2'], st['F1'], st['F2'])
            p_y_target = py if y_target == '1' else 1.0 - py
            total += jp
            weighted += jp * p_y_target
        return weighted / total if total > 0 else 0.0, total

    p_y_undesired_given, p_undesired_ante = _marginal_p_y(undesired_states, target_undesired)
    p_y_desired_given, _ = _marginal_p_y(desired_states, target_desired)
    d = p_y_desired_given + p_y_undesired_given - 1.0
    return float(d * p_undesired_ante)


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------


@dataclass
class CoverageRecord:
    """One row of the coverage simulation output."""

    n: int
    replicate: int
    method: str
    rule_index: int
    true_uplift: float
    point: float
    ci_lower: float
    ci_upper: float
    width: float
    covered: bool
    se: float
    elapsed_seconds: float


def run_replicate(
    n: int,
    replicate_seed: int,
    params: DGPParams,
    *,
    min_undesired_support: int = 5,
    min_desired_support: int = 5,
    n_bootstrap: int = 200,
    n_mc: int = 2000,
    methods: Sequence[str] = ('bootstrap_percentile', 'bootstrap_bca', 'wald', 'wilson', 'bayesian'),
    confidence_level: float = 0.95,
) -> List[CoverageRecord]:
    """Run a single replicate and return one record per (rule, method).

    Returns an empty list when no rules are mined on the synthetic sample
    (rare but possible for very small ``n``).
    """
    df = generate_dataset(n, params, seed=replicate_seed)
    ar = ActionRules(
        min_stable_attributes=1,
        min_flexible_attributes=1,
        min_undesired_support=min_undesired_support,
        min_desired_support=min_desired_support,
        min_undesired_confidence=0.5,
        min_desired_confidence=0.5,
    )
    try:
        ar.fit(
            df,
            stable_attributes=['S1', 'S2'],
            flexible_attributes=['F1', 'F2'],
            target='Y',
            target_undesired_state='0',
            target_desired_state='1',
        )
    except Exception:
        return []
    if ar.output is None or not ar.output.action_rules:
        return []

    column_values = ar.output.column_values
    truths = [true_uplift(rule, column_values, params) for rule in ar.output.action_rules]

    records: List[CoverageRecord] = []
    method_kwargs: dict = {
        'bootstrap_percentile': dict(method='bootstrap', bootstrap_type='percentile', n_bootstrap=n_bootstrap),
        'bootstrap_bca': dict(method='bootstrap', bootstrap_type='bca', n_bootstrap=n_bootstrap),
        'wald': dict(method='analytic', analytic_type='wald'),
        'wilson': dict(method='analytic', analytic_type='wilson'),
        'bayesian': dict(method='bayesian', n_mc=n_mc),
    }
    for label in methods:
        kwargs = dict(method_kwargs[label])
        kwargs.update(confidence_level=confidence_level, random_state=replicate_seed)
        start = time.perf_counter()
        try:
            results = ar.confidence_intervals(df, **kwargs)
        except Exception:
            continue
        elapsed = time.perf_counter() - start
        for r, truth in zip(results, truths):
            width = r.uplift_ci_upper - r.uplift_ci_lower
            covered = bool(r.uplift_ci_lower <= truth <= r.uplift_ci_upper)
            records.append(
                CoverageRecord(
                    n=n,
                    replicate=replicate_seed,
                    method=label,
                    rule_index=r.rule_index,
                    true_uplift=float(truth),
                    point=float(r.uplift_point),
                    ci_lower=float(r.uplift_ci_lower),
                    ci_upper=float(r.uplift_ci_upper),
                    width=float(width),
                    covered=covered,
                    se=float(r.uplift_se),
                    elapsed_seconds=elapsed,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_records(records: List[CoverageRecord]) -> pd.DataFrame:
    """Aggregate per (n, method): coverage rate, mean width, mean runtime."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([rec.__dict__ for rec in records])
    grouped = (
        df.groupby(['n', 'method'])
        .agg(
            empirical_coverage=('covered', 'mean'),
            mean_width=('width', 'mean'),
            median_width=('width', 'median'),
            mean_se=('se', 'mean'),
            mean_runtime_s=('elapsed_seconds', 'mean'),
            n_rules_evaluated=('covered', 'count'),
        )
        .reset_index()
    )
    return grouped


def run_grid(
    sample_sizes: Sequence[int],
    n_replicates: int,
    *,
    params: DGPParams = DGPParams(),
    base_seed: int = 0,
    progress: bool = False,
    **replicate_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full simulation grid and return (records_df, summary_df).

    Parameters
    ----------
    sample_sizes : sequence of int
    n_replicates : int
    params : DGPParams
    base_seed : int
        Each replicate uses ``base_seed + (size_index * 1000) + r`` as its
        per-replicate seed so the same call produces a reproducible run.
    progress : bool
        Print a single ``[n=…, r=…]`` line per started replicate.
    **replicate_kwargs
        Forwarded to :func:`run_replicate`.
    """
    all_records: List[CoverageRecord] = []
    for size_i, n in enumerate(sample_sizes):
        for r in range(n_replicates):
            seed = base_seed + size_i * 1000 + r
            if progress:
                print(f"[n={n}, r={r}, seed={seed}] running…", flush=True)
            recs = run_replicate(n, seed, params, **replicate_kwargs)
            all_records.extend(recs)
    records_df = pd.DataFrame([rec.__dict__ for rec in all_records])
    summary_df = aggregate_records(all_records)
    return records_df, summary_df
