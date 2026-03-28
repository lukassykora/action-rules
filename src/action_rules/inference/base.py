"""Base module for confidence interval computation on action rules.

Provides the abstract base class, data classes, enums, and shared helper
functions consumed by all CI engines (analytic, bootstrap, Bayesian).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RuleCategory(Enum):
    """Classification of an action rule based on its confidence interval.

    Attributes
    ----------
    ACCEPT : str
        The entire CI is at or above the threshold — rule is reliable.
    REJECT : str
        The entire CI is below the threshold — rule should be discarded.
    UNCERTAIN : str
        The CI straddles the threshold — insufficient evidence to decide.
    """

    ACCEPT = 'accept'
    REJECT = 'reject'
    UNCERTAIN = 'uncertain'


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RuleMasks:
    """Structured representation of a single action rule for data-driven inference.

    Parameters
    ----------
    mask_undesired : dict
        Mapping of {attribute: value} for conditions in the undesired classification
        rule (stable attributes at their fixed value plus flexible attributes at their
        undesired value).
    mask_desired : dict
        Mapping of {attribute: value} for conditions in the desired classification
        rule (stable attributes at their fixed value plus flexible attributes at their
        desired value).
    target_attribute : str
        Name of the target column.
    target_undesired : str
        Target value representing the undesired outcome.
    target_desired : str
        Target value representing the desired outcome.
    rule_index : int
        Zero-based position of this rule in the original ``Output.action_rules`` list.
    undesired_itemset : tuple
        Original integer column indices that form the undesired itemset.
        Preserved so that utility tables (keyed by integer index) can be looked up
        without re-encoding.
    desired_itemset : tuple
        Original integer column indices that form the desired itemset.
    """

    mask_undesired: dict
    mask_desired: dict
    target_attribute: str
    target_undesired: str
    target_desired: str
    rule_index: int
    undesired_itemset: tuple
    desired_itemset: tuple


@dataclass
class ConfidenceIntervalResult:
    """Confidence interval result for a single action rule.

    Parameters
    ----------
    rule_index : int
        Zero-based index of the action rule in the source ``Output.action_rules``.
    method : str
        Name of the CI method used: ``'bootstrap'``, ``'analytic'``, or ``'bayesian'``.
    confidence_level : float
        Nominal coverage probability, e.g. ``0.95``.
    uplift_point : float
        Point estimate of the uplift measure.
    uplift_ci_lower : float
        Lower bound of the uplift confidence interval.
    uplift_ci_upper : float
        Upper bound of the uplift confidence interval.
    uplift_se : float
        Standard error of the uplift estimate.
    realistic_rule_gain_point : float, optional
        Point estimate of the realistic rule gain (only when utility tables are given).
    realistic_rule_gain_ci_lower : float, optional
        Lower bound of the realistic rule gain CI.
    realistic_rule_gain_ci_upper : float, optional
        Upper bound of the realistic rule gain CI.
    realistic_rule_gain_se : float, optional
        Standard error of the realistic rule gain.
    support : int
        Transaction support of the action rule.
    confidence : float
        Confidence of the action rule.
    category : RuleCategory, optional
        Qualitative verdict derived from the uplift CI vs. a threshold.
    samples_uplift : np.ndarray, optional
        Raw bootstrap/posterior samples for uplift (excluded from repr to keep it brief).
    samples_gain : np.ndarray, optional
        Raw bootstrap/posterior samples for gain (excluded from repr to keep it brief).
    """

    rule_index: int
    method: str
    confidence_level: float
    uplift_point: float
    uplift_ci_lower: float
    uplift_ci_upper: float
    uplift_se: float
    realistic_rule_gain_point: Optional[float] = None
    realistic_rule_gain_ci_lower: Optional[float] = None
    realistic_rule_gain_ci_upper: Optional[float] = None
    realistic_rule_gain_se: Optional[float] = None
    support: int = 0
    confidence: float = 0.0
    category: Optional[RuleCategory] = None
    samples_uplift: Optional[np.ndarray] = field(default=None, repr=False)
    samples_gain: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class InferenceEngine(ABC):
    """Abstract base class for CI engines operating on action rules.

    All concrete engines must implement ``compute()``, which processes a
    dataset together with the parsed action rules and returns one
    :class:`ConfidenceIntervalResult` per rule.

    Notes
    -----
    Engines should be stateless with respect to data — the ``compute()``
    method receives all required inputs at call time so that the same engine
    instance can be reused across multiple datasets.
    """

    @abstractmethod
    def compute(
        self,
        data: pd.DataFrame,
        rules: list,
        confidence_level: float = 0.95,
        intrinsic_utility_table: Optional[dict] = None,
        transition_utility_table: Optional[dict] = None,
        column_values: Optional[dict] = None,
    ) -> List[ConfidenceIntervalResult]:
        """Compute confidence intervals for a list of action rules.

        Parameters
        ----------
        data : pd.DataFrame
            The full original dataset used for inference.  Each row is one
            transaction; columns correspond to attribute names.
        rules : list
            List of :class:`RuleMasks` objects produced by
            :func:`extract_rule_masks`.
        confidence_level : float, optional
            Desired nominal coverage probability.  Default is ``0.95``.
        intrinsic_utility_table : dict, optional
            Mapping ``(attribute, value) -> float`` for intrinsic utilities.
            Pass ``None`` (default) when utility-based gain is not required.
        transition_utility_table : dict, optional
            Mapping ``(attribute, from_value, to_value) -> float`` for
            transition utilities.  Pass ``None`` (default) when not required.
        column_values : dict, optional
            Mapping ``int -> (attribute, value)`` from ``Output.column_values``.
            Required when utility tables are provided so that integer itemset
            indices can be resolved to ``(attribute, value)`` keys.

        Returns
        -------
        List[ConfidenceIntervalResult]
            One result object per rule in *rules*, in the same order.
        """


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_rule_masks(output) -> List[RuleMasks]:
    """Parse an ``Output`` object and return one :class:`RuleMasks` per action rule.

    The function iterates over ``output.action_rules``, resolves integer
    column indices through ``output.column_values``, and assembles the
    attribute-value dicts required by the inference helpers.

    Parameters
    ----------
    output : Output
        A populated :class:`~action_rules.output.output.Output` instance.
        Required attributes: ``action_rules``, ``column_values``, ``target``.

    Returns
    -------
    List[RuleMasks]
        One :class:`RuleMasks` per entry in ``output.action_rules``, in
        the same order.

    Notes
    -----
    The ``undesired_itemset`` and ``desired_itemset`` fields carry the raw
    integer indices so that downstream code can look up utility tables that
    are also keyed by integer index (as used inside :class:`~action_rules.rules.rules.Rules`).

    Each action rule dict is expected to have the shape::

        {
            'undesired': {
                'itemset': tuple[int, ...],
                'support': int,
                'confidence': float,
                'target': int,
            },
            'desired': {
                'itemset': tuple[int, ...],
                'support': int,
                'confidence': float,
                'target': int,
            },
            'uplift': float,
            'support': int,
            'confidence': float,
            # optional utility fields:
            # 'max_rule_gain', 'realistic_rule_gain', 'realistic_dataset_gain'
        }
    """
    masks: List[RuleMasks] = []

    for rule_index, action_rule in enumerate(output.action_rules):
        undesired_part = action_rule['undesired']
        desired_part = action_rule['desired']

        undesired_itemset: Tuple[int, ...] = tuple(undesired_part['itemset'])
        desired_itemset: Tuple[int, ...] = tuple(desired_part['itemset'])

        mask_undesired: Dict[str, str] = {}
        mask_desired: Dict[str, str] = {}

        # Resolve each column index to (attribute, value) and build masks.
        for u_idx, d_idx in zip(undesired_itemset, desired_itemset):
            u_attr, u_val = output.column_values[u_idx]
            d_attr, d_val = output.column_values[d_idx]
            mask_undesired[str(u_attr)] = str(u_val)
            mask_desired[str(d_attr)] = str(d_val)

        # Resolve target indices.
        target_attr = str(output.target)
        target_undesired = str(output.column_values[undesired_part['target']][1])
        target_desired = str(output.column_values[desired_part['target']][1])

        masks.append(
            RuleMasks(
                mask_undesired=mask_undesired,
                mask_desired=mask_desired,
                target_attribute=target_attr,
                target_undesired=target_undesired,
                target_desired=target_desired,
                rule_index=rule_index,
                undesired_itemset=undesired_itemset,
                desired_itemset=desired_itemset,
            )
        )

    return masks


def apply_mask(data: pd.DataFrame, mask: dict) -> pd.Series:
    """Return a boolean Series indicating rows that satisfy all conditions in *mask*.

    Both the column values in *data* and the mask values are cast to ``str``
    before comparison, so that numeric columns encoded as integers compare
    correctly with string-typed mask values.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to filter.
    mask : dict
        Mapping of ``{column_name: expected_value}`` conditions.  All
        conditions are combined with a logical AND.

    Returns
    -------
    pd.Series
        Boolean Series of length ``len(data)``.  ``True`` where every
        condition is satisfied.

    Notes
    -----
    If *mask* is empty the returned Series is all ``True``.
    """
    result = pd.Series([True] * len(data), index=data.index)
    for col, val in mask.items():
        result = result & (data[col].astype(str) == str(val))
    return result


def compute_group_counts(
    data: pd.DataFrame,
    rule_masks: RuleMasks,
) -> Tuple[int, int, int, int, int]:
    """Count antecedent and matching rows for both sides of an action rule.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to evaluate against.
    rule_masks : RuleMasks
        Parsed masks for one action rule.

    Returns
    -------
    tuple
        A 5-tuple ``(n_undesired_antecedent, n_undesired_match,
        n_desired_antecedent, n_desired_match, n_total)`` where:

        - ``n_undesired_antecedent`` — rows satisfying ``mask_undesired``
        - ``n_undesired_match`` — rows satisfying ``mask_undesired`` **and**
          whose target equals ``target_undesired``
        - ``n_desired_antecedent`` — rows satisfying ``mask_desired``
        - ``n_desired_match`` — rows satisfying ``mask_desired`` **and**
          whose target equals ``target_desired``
        - ``n_total`` — total number of rows in *data*
    """
    target_col = rule_masks.target_attribute

    undesired_ante = apply_mask(data, rule_masks.mask_undesired)
    desired_ante = apply_mask(data, rule_masks.mask_desired)

    target_series = data[target_col].astype(str)

    n_undesired_antecedent = int(undesired_ante.sum())
    n_undesired_match = int((undesired_ante & (target_series == rule_masks.target_undesired)).sum())

    n_desired_antecedent = int(desired_ante.sum())
    n_desired_match = int((desired_ante & (target_series == rule_masks.target_desired)).sum())

    n_total = len(data)

    return n_undesired_antecedent, n_undesired_match, n_desired_antecedent, n_desired_match, n_total


def compute_uplift_from_counts(
    n_u_ante: int,
    n_u_match: int,
    n_d_ante: int,
    n_d_match: int,
    n_total: int,
) -> float:
    """Compute the uplift measure from pre-computed group counts.

    Implements the uplift formula (Ras et al., 2009)::

        conf_u = n_u_match / n_u_ante
        conf_d = n_d_match / n_d_ante
        d      = conf_d - (1 - conf_u)   # = conf_d + conf_u - 1
        uplift = d * n_u_ante / n_total

    Parameters
    ----------
    n_u_ante : int
        Number of rows satisfying the undesired antecedent.
    n_u_match : int
        Number of rows satisfying the undesired antecedent whose target is the
        undesired value.
    n_d_ante : int
        Number of rows satisfying the desired antecedent.
    n_d_match : int
        Number of rows satisfying the desired antecedent whose target is the
        desired value.
    n_total : int
        Total number of rows in the dataset.

    Returns
    -------
    float
        Uplift value.  Returns ``0.0`` when any denominator is zero.

    Notes
    -----
    Division-by-zero guards return ``0.0`` rather than ``NaN`` to keep
    downstream aggregations (mean, CI bounds) numerically stable when a
    bootstrap resample happens to contain zero antecedent rows.
    """
    if n_u_ante == 0 or n_d_ante == 0 or n_total == 0:
        return 0.0

    conf_u = n_u_match / n_u_ante
    conf_d = n_d_match / n_d_ante

    # Uplift measure (Ras et al., 2009)
    # d = conf_d - (1 - conf_u)
    d = conf_d - (1.0 - conf_u)
    uplift = d * n_u_ante / n_total
    return float(uplift)


def compute_realistic_gain(
    rule_masks: RuleMasks,
    conf_u: float,
    conf_d: float,
    intrinsic_utility_table: Optional[dict],
    transition_utility_table: Optional[dict],
    column_values: dict,
) -> float:
    """Compute the realistic rule gain for a single action rule.

    Mirrors the formula inside :meth:`~action_rules.rules.rules.Rules.compute_rule_utility`
    but accepts string-keyed utility tables (keyed by ``(attribute, value)``
    tuples) and resolves integer itemset indices through *column_values*.

    Formula::

        rule_gain           = sum(intrinsic[desired]) - sum(intrinsic[undesired])
                              + transition_gains
        d                   = conf_d - (1 - conf_u)
        target_gain         = intrinsic[target_desired] - intrinsic[target_undesired]
                              + transition[target]
        realistic_rule_gain = rule_gain + d * target_gain

    Parameters
    ----------
    rule_masks : RuleMasks
        Parsed masks including ``undesired_itemset`` and ``desired_itemset``
        (tuples of integer column indices).
    conf_u : float
        Confidence of the undesired classification rule on the current data.
    conf_d : float
        Confidence of the desired classification rule on the current data.
    intrinsic_utility_table : dict, optional
        ``(attribute, value) -> float`` mapping.  ``None`` is treated as
        an empty dict (all utilities default to ``0.0``).
    transition_utility_table : dict, optional
        ``(attribute, from_value, to_value) -> float`` mapping.  ``None``
        is treated as an empty dict.
    column_values : dict
        ``int -> (attribute, value)`` mapping from
        ``Output.column_values``.

    Returns
    -------
    float
        The realistic rule gain.  Returns ``0.0`` when both utility tables
        are empty or ``None``.
    """
    intrinsic = intrinsic_utility_table or {}
    transition = transition_utility_table or {}

    # Accumulate intrinsic utilities for undesired itemset.
    u_undesired = 0.0
    for idx in rule_masks.undesired_itemset:
        attr, val = column_values[idx]
        u_undesired += intrinsic.get((attr, val), 0.0)

    # Accumulate intrinsic utilities for desired itemset.
    u_desired = 0.0
    for idx in rule_masks.desired_itemset:
        attr, val = column_values[idx]
        u_desired += intrinsic.get((attr, val), 0.0)

    # Transition gain for flexible items (indices differ between the two itemsets).
    transition_gain = 0.0
    for u_idx, d_idx in zip(rule_masks.undesired_itemset, rule_masks.desired_itemset):
        if u_idx != d_idx:
            u_attr, u_val = column_values[u_idx]
            _d_attr, d_val = column_values[d_idx]
            # Attribute is the same for both sides of a flexible attribute change.
            transition_gain += transition.get((u_attr, u_val, d_val), 0.0)

    rule_gain = u_desired - u_undesired + transition_gain

    # Target utilities keyed by the string target values directly.
    target_attr = rule_masks.target_attribute
    target_u_val = rule_masks.target_undesired
    target_d_val = rule_masks.target_desired

    u_target_undesired = intrinsic.get((target_attr, target_u_val), 0.0)
    u_target_desired = intrinsic.get((target_attr, target_d_val), 0.0)
    transition_gain_target = transition.get((target_attr, target_u_val, target_d_val), 0.0)

    target_gain = u_target_desired - u_target_undesired + transition_gain_target

    # d = conf_d - (1 - conf_u) = conf_d + conf_u - 1
    d = conf_d - (1.0 - conf_u)
    realistic_rule_gain = rule_gain + d * target_gain
    return float(realistic_rule_gain)


def categorize_rule(ci_lower: float, ci_upper: float, threshold: float) -> RuleCategory:
    """Assign a qualitative verdict to a rule based on its CI and a threshold.

    Parameters
    ----------
    ci_lower : float
        Lower bound of the confidence interval (e.g. for uplift).
    ci_upper : float
        Upper bound of the confidence interval.
    threshold : float
        Decision boundary.  Typically ``0.0`` for uplift (any positive gain
        is desired).

    Returns
    -------
    RuleCategory
        - :attr:`RuleCategory.ACCEPT` when ``ci_lower >= threshold``
        - :attr:`RuleCategory.REJECT` when ``ci_upper < threshold``
        - :attr:`RuleCategory.UNCERTAIN` otherwise
    """
    if ci_lower >= threshold:
        return RuleCategory.ACCEPT
    if ci_upper < threshold:
        return RuleCategory.REJECT
    return RuleCategory.UNCERTAIN
