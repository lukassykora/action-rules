"""Rule-level targeting metrics for action-rule cross-validation.

The metrics treat each *action rule* as a candidate targeting policy and
quantify how the policies would perform if deployed.  Each function takes a
parallel set of arrays where one entry corresponds to one rule:

- ``scores`` — the ranking signal (e.g. point-estimate uplift on the training
  fold, or a lower confidence bound, or a risk-adjusted score).
- ``outcomes`` — the realized effect on the held-out fold (e.g. test-set
  uplift, test-set realistic gain).
- ``supports`` — optional rule support on the held-out fold; used as weights
  in the Qini / AUUC curves so that high-coverage rules contribute more to
  the cumulative effect.

When ``supports`` is omitted the curves treat every rule as equally weighted,
which corresponds to a "rule-counted" interpretation.  When ``supports`` is
provided the curves use *coverage-weighted* cumulative sums, which matches
the standard "instance-counted" Qini definition (Radcliffe, 2007).
"""

from typing import List, Optional, Tuple, Union

import numpy as np

ArrayLike = Union[List[float], List[int], np.ndarray]


def _asarray(x: ArrayLike, dtype=float) -> np.ndarray:
    """Cast ``x`` to a 1-D NumPy array of the requested dtype."""
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}.")
    return arr


def _validate_lengths(*arrays: np.ndarray) -> None:
    if len({a.shape[0] for a in arrays}) > 1:
        shapes = ", ".join(str(a.shape) for a in arrays)
        raise ValueError(f"All input arrays must have the same length; got {shapes}.")


def _top_k_indices(scores: np.ndarray, k_fraction: float) -> np.ndarray:
    """Return indices of the top ``k_fraction`` entries of ``scores`` (descending).

    At least one entry is always selected.  Ties are broken by NumPy's
    stable sort, so the result is deterministic for a given input.
    """
    if not 0.0 < k_fraction <= 1.0:
        raise ValueError(f"k_fraction must be in (0, 1]; got {k_fraction}.")
    n = scores.shape[0]
    if n == 0:
        return np.array([], dtype=np.intp)
    k = max(1, int(np.ceil(n * k_fraction)))
    # Sort descending — stable on argsort by negating then using kind='stable'.
    order = np.argsort(-scores, kind='stable')
    return order[:k]


# ---------------------------------------------------------------------------
# Top-k metrics
# ---------------------------------------------------------------------------


def uplift_at_k(
    scores: ArrayLike,
    outcomes: ArrayLike,
    k_fraction: float = 0.2,
) -> float:
    """Mean held-out uplift of the top ``k_fraction`` of rules ranked by ``scores``.

    Parameters
    ----------
    scores : array_like
        Ranking score per rule (higher is better).  Typically the
        train-fold point estimate of uplift, the lower CI bound, or a
        risk-adjusted score.
    outcomes : array_like
        Realized test-fold uplift per rule.
    k_fraction : float
        Fraction of rules to select.  Must satisfy ``0 < k_fraction <= 1``.

    Returns
    -------
    float
        Mean of the top-k outcomes.  Returns ``0.0`` when *outcomes* is empty.
    """
    s = _asarray(scores)
    o = _asarray(outcomes)
    _validate_lengths(s, o)
    if s.shape[0] == 0:
        return 0.0
    idx = _top_k_indices(s, k_fraction)
    return float(np.mean(o[idx]))


def incremental_profit_at_k(
    scores: ArrayLike,
    gains: ArrayLike,
    supports: Optional[ArrayLike] = None,
    k_fraction: float = 0.2,
) -> float:
    """Total profit captured by the top ``k_fraction`` of rules.

    Sums each selected rule's realized gain weighted by its test support
    (``supports * gains``).  Without supports, the metric collapses to the
    sum of gains across the selected rules.

    Parameters
    ----------
    scores : array_like
        Ranking score per rule.
    gains : array_like
        Per-rule realized gain (e.g. realistic_rule_gain on the test fold).
    supports : array_like, optional
        Per-rule test-fold support.  When provided, the metric weights each
        gain by its support — i.e. expected total profit when targeting the
        selected rules.
    k_fraction : float
        Fraction of rules to select.

    Returns
    -------
    float
    """
    s = _asarray(scores)
    g = _asarray(gains)
    if supports is None:
        _validate_lengths(s, g)
        w = np.ones_like(g)
    else:
        w = _asarray(supports)
        _validate_lengths(s, g, w)
    if s.shape[0] == 0:
        return 0.0
    idx = _top_k_indices(s, k_fraction)
    return float(np.sum(g[idx] * w[idx]))


def realistic_gain_at_k(
    scores: ArrayLike,
    gains: ArrayLike,
    k_fraction: float = 0.2,
) -> float:
    """Mean realized realistic gain of the top ``k_fraction`` of rules.

    Identical in shape to :func:`uplift_at_k` but applied to ``realistic_rule_gain``
    outcomes.  Provided as a separate alias to keep call sites self-documenting.
    """
    return uplift_at_k(scores, gains, k_fraction=k_fraction)


# ---------------------------------------------------------------------------
# Qini / AUUC curves
# ---------------------------------------------------------------------------


def qini_curve(
    scores: ArrayLike,
    outcomes: ArrayLike,
    supports: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the rule-level Qini curve.

    The curve is obtained by ranking rules from best to worst by ``scores``
    and accumulating the (optionally support-weighted) outcomes.  Both axes
    are normalized to ``[0, 1]`` so different fold sizes are comparable.

    Parameters
    ----------
    scores : array_like
        Per-rule ranking score (higher = recommend earlier).
    outcomes : array_like
        Per-rule realized effect on the held-out fold.
    supports : array_like, optional
        Per-rule test-fold support.  When provided, the x-axis is
        cumulative coverage (fraction of total support) and the y-axis is
        cumulative weighted effect.  When omitted, both axes are simple
        fractional counts.

    Returns
    -------
    tuple
        ``(x, y)`` where ``x`` is cumulative coverage in ``[0, 1]`` and
        ``y`` is cumulative effect in ``[0, 1]``.  The arrays both have a
        leading zero element so the curve always starts at ``(0, 0)``.

    Notes
    -----
    When the sum of effects is non-positive (degenerate case), the y-axis
    is left unnormalized to preserve sign information.  This mirrors the
    behavior of ``scikit-uplift``'s :func:`uplift_auc_score`.

    References
    ----------
    Radcliffe, N. J. (2007). Using Control Groups to Target on Predicted Lift:
    Building and Assessing Uplift Models.  Direct Marketing Analytics Journal.
    """
    s = _asarray(scores)
    o = _asarray(outcomes)
    if supports is None:
        w = np.ones_like(o)
    else:
        w = _asarray(supports)
    _validate_lengths(s, o, w)

    n = s.shape[0]
    if n == 0:
        return np.array([0.0]), np.array([0.0])

    order = np.argsort(-s, kind='stable')
    cum_w = np.cumsum(w[order])
    cum_effect = np.cumsum(o[order] * w[order])

    total_w = cum_w[-1] if cum_w[-1] > 0 else 1.0
    total_effect = cum_effect[-1]
    norm_effect = total_effect if total_effect != 0 else 1.0
    # Preserve sign on degenerate (non-positive total) cases by skipping
    # normalization — see the Notes section above.
    y_denom = norm_effect if total_effect > 0 else 1.0

    x = np.concatenate(([0.0], cum_w / total_w))
    y = np.concatenate(([0.0], cum_effect / y_denom))
    return x, y


def auuc(
    scores: ArrayLike,
    outcomes: ArrayLike,
    supports: Optional[ArrayLike] = None,
) -> float:
    """Area Under the Uplift Curve (AUUC).

    Trapezoidal integral of the :func:`qini_curve` output. Higher values
    indicate a better targeting ranking. Equal to ``0.5`` for a perfectly
    uninformative ranking.

    Parameters
    ----------
    scores, outcomes, supports : array_like
        See :func:`qini_curve`.

    Returns
    -------
    float
        AUUC value.  Returns ``0.0`` when the input is empty.
    """
    x, y = qini_curve(scores, outcomes, supports=supports)
    if x.shape[0] < 2:
        return 0.0
    return float(np.trapezoid(y, x))


def qini_coefficient(
    scores: ArrayLike,
    outcomes: ArrayLike,
    supports: Optional[ArrayLike] = None,
) -> float:
    """Qini coefficient: AUUC minus the area under the random baseline.

    The random baseline is the diagonal of the unit square (area ``0.5``),
    so the Qini coefficient is in ``[-0.5, 0.5]`` for well-behaved inputs.
    A perfect ranking achieves close to ``0.5``; a random ranking yields
    approximately ``0``.

    Parameters
    ----------
    scores, outcomes, supports : array_like
        See :func:`qini_curve`.

    Returns
    -------
    float
    """
    return auuc(scores, outcomes, supports=supports) - 0.5
