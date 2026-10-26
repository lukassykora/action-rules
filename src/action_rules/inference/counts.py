"""Count-level inference for conditional and population action-rule estimands.

This research interface keeps sampling design and estimand explicit. It does
not mine rules or turn observational contrasts into intervention effects.
"""

from typing import Dict

import numpy as np
from scipy.stats import beta, norm


def _cp(x, n, failure):
    """Return Clopper-Pearson component bounds."""
    lo = np.where(x == 0, 0.0, beta.ppf(failure / 2, x, n - x + 1))
    hi = np.where(x == n, 1.0, beta.ppf(1 - failure / 2, x + 1, n - x))
    return lo, hi


def _quantile(draws, probability):
    """Evaluate possibly different empirical quantiles in each row."""
    values = np.sort(draws, axis=1)
    count = np.isfinite(values).sum(axis=1)
    p = np.broadcast_to(np.asarray(probability), count.shape)
    index = np.maximum(count - 1, 0) * np.clip(p, 0, 1)
    lower = np.floor(index).astype(int)
    upper = np.ceil(index).astype(int)
    first = np.take_along_axis(values, lower[:, None], axis=1)[:, 0]
    last = np.take_along_axis(values, upper[:, None], axis=1)[:, 0]
    return first + (last - first) * (index - lower)


def _summary(point, lower, upper, supported):
    """Make unsupported interval bounds explicit."""
    valid = supported & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    return dict(
        point=point, lower=np.where(valid, lower, np.nan), upper=np.where(valid, upper, np.nan), supported=valid
    )


def _corners(qlo, qhi, lo, hi):
    """Bound a product without assuming that its factors are positive."""
    corners = np.array([qlo * lo, qlo * hi, qhi * lo, qhi * hi])
    return corners.min(axis=0), corners.max(axis=0)


def infer_counts(
    counts,
    a=0.0,
    b=1.0,
    *,
    method="wilson",
    design="random",
    level=0.95,
    family_size=1,
    n_resamples=999,
    n_mc=4000,
    random_state=None,
) -> Dict[str, dict]:
    """Infer G, U and H from disjoint-rule five-cell counts.

    Parameters
    ----------
    counts : array_like
        Rows are undesired match/nonmatch, desired match/nonmatch, other.
    a, b : float or array_like
        Fixed coefficients of G = a + b * delta, one pair per row.
    method : str
        wald, wald_rd, wilson, cp, percentile, bca, bayes, jeffreys,
        bayes_rd or jeffreys_rd. CP family protection is per estimand.
    design : str
        fixed conditions on group counts; random resamples multinomial counts.
    level : float
        Nominal marginal or family confidence level.
    family_size : int
        Discovery-frozen family size for CP component Bonferroni bounds.
    n_resamples, n_mc : int
        Bootstrap and posterior sample counts, respectively.
    random_state : int, optional
        Seed for NumPy's generator.

    Returns
    -------
    dict
        G, U and H each map to point, lower, upper and supported arrays.
        Points are plug-in estimates for every method. Unsupported bounds
        are NaN, including nonconstant zero-width Wald/bootstrap intervals.
    """
    c = np.atleast_2d(np.asarray(counts, dtype=float))
    if c.shape[1] != 5 or np.any(~np.isfinite(c)) or np.any(c < 0) or np.any(c != np.floor(c)):
        raise ValueError("counts must contain five finite nonnegative integer cells")
    if not 0 < level < 1 or not isinstance(family_size, int) or family_size < 1:
        raise ValueError("level must be in (0,1) and family_size an integer >= 1")
    if design not in {"fixed", "random"}:
        raise ValueError("design must be fixed or random")
    valid_methods = {
        "wald",
        "wald_rd",
        "wilson",
        "cp",
        "percentile",
        "bca",
        "bayes",
        "jeffreys",
        "bayes_rd",
        "jeffreys_rd",
    }
    if method not in valid_methods:
        raise ValueError("Unknown count-level method")
    if n_resamples < 2 or n_mc < 2:
        raise ValueError("At least two draws are required")
    n = c.sum(axis=1)
    nu, nd = c[:, :2].sum(axis=1), c[:, 2:4].sum(axis=1)
    observed = (nu > 0) & (nd > 0) & (n > 0)
    aa, bb = np.broadcast_to(a, n.shape), np.broadcast_to(b, n.shape)
    if np.any(~np.isfinite(aa)) or np.any(~np.isfinite(bb)):
        raise ValueError("Utility coefficients must be finite")
    with np.errstate(divide='ignore', invalid='ignore'):
        pu, pd, q = c[:, 0] / nu, c[:, 2] / nd, nu / n
        delta = pu + pd - 1
        gain = aa + bb * delta
        points = dict(G=gain, U=q * delta, H=q * gain)
        vd = pu * (1 - pu) / nu + pd * (1 - pd) / nd
    z, failure = norm.ppf(1 - (1 - level) / 2), 1 - level
    context = locals()
    if method in {"wald", "wald_rd", "wilson", "cp"}:
        result = _analytic_intervals(context)
    else:
        rng = np.random.default_rng(random_state)
        if method in {'percentile', 'bca'}:
            result = _bootstrap_intervals(context, rng)
        else:
            result = _posterior_intervals(context, rng)
    # With both antecedents observed, G=a is known even if a jackknife
    # deletion would empty a singleton group. It needs no outcome inference.
    constant = observed & (bb == 0)
    for endpoint in ['point', 'lower', 'upper']:
        result['G'][endpoint] = np.where(constant, aa, result['G'][endpoint])
    result['G']['supported'] |= constant
    return result


def _analytic_intervals(context):
    """Construct analytic intervals from validated counts."""
    c, n, nu, nd, aa, bb, pu, pd, q, delta, gain, points, vd, observed, z, failure, method, design, family_size = (
        context['c'],
        context['n'],
        context['nu'],
        context['nd'],
        context['aa'],
        context['bb'],
        context['pu'],
        context['pd'],
        context['q'],
        context['delta'],
        context['gain'],
        context['points'],
        context['vd'],
        context['observed'],
        context['z'],
        context['failure'],
        context['method'],
        context['design'],
        context['family_size'],
    )
    bounds = {}
    if method in {"wald", "wald_rd"}:
        dl, dh = delta - z * np.sqrt(vd), delta + z * np.sqrt(vd)
    elif method == 'wilson':

        def wilson(p, size):
            den = 1 + z * z / size
            middle = p + z * z / (2 * size)
            radius = z * np.sqrt(p * (1 - p) / size + z * z / (4 * size**2))
            return (middle - radius) / den, (middle + radius) / den

        with np.errstate(divide='ignore', invalid='ignore'):
            lu, uu = wilson(pu, nu)
            ld, ud = wilson(pd, nd)
        dl = delta - np.sqrt((pu - lu) ** 2 + (pd - ld) ** 2)
        dh = delta + np.sqrt((uu - pu) ** 2 + (ud - pd) ** 2)
    else:
        lu, uu = _cp(c[:, 0], nu, failure / (2 * family_size))
        ld, ud = _cp(c[:, 2], nd, failure / (2 * family_size))
        dl, dh = lu + ld - 1, uu + ud - 1
    gl, gh = np.minimum(aa + bb * dl, aa + bb * dh), np.maximum(aa + bb * dl, aa + bb * dh)
    bounds = dict(G=(gl, gh), U=(q * dl, q * dh), H=(q * gl, q * gh))
    if method == 'wald_rd':
        vq = q * (1 - q) / n
        for key, variance in [('U', q * q * vd + delta * delta * vq), ('H', q * q * bb * bb * vd + gain * gain * vq)]:
            se = np.sqrt(variance)
            bounds[key] = (points[key] - z * se, points[key] + z * se)
    if method == 'cp' and design == 'random':
        lu, uu = _cp(c[:, 0], nu, failure / (3 * family_size))
        ld, ud = _cp(c[:, 2], nd, failure / (3 * family_size))
        ql, qh = _cp(nu, n, failure / (3 * family_size))
        ddlo, ddhi = lu + ld - 1, uu + ud - 1
        gglo, gghi = np.minimum(aa + bb * ddlo, aa + bb * ddhi), np.maximum(aa + bb * ddlo, aa + bb * ddhi)
        bounds['U'], bounds['H'] = _corners(ql, qh, ddlo, ddhi), _corners(ql, qh, gglo, gghi)
    return {
        key: _summary(points[key], lo, hi, observed & ((hi > lo) | (key == 'G') & (bb == 0)))
        for key, (lo, hi) in bounds.items()
    }


def _bootstrap_intervals(context, rng):
    """Construct bootstrap intervals from validated counts."""
    c, n, nu, nd, aa, bb, pu, pd, q, points, failure, observed, method, design, n_resamples = (
        context['c'],
        context['n'],
        context['nu'],
        context['nd'],
        context['aa'],
        context['bb'],
        context['pu'],
        context['pd'],
        context['q'],
        context['points'],
        context['failure'],
        context['observed'],
        context['method'],
        context['design'],
        context['n_resamples'],
    )
    bounds = {}
    draws = np.zeros((len(c), n_resamples, 5))
    for i, row in enumerate(c):
        if not observed[i]:
            continue
        if design == 'random':
            draws[i] = rng.multinomial(int(n[i]), row / n[i], size=n_resamples)
        else:
            xu, xd = rng.binomial(int(nu[i]), pu[i], n_resamples), rng.binomial(int(nd[i]), pd[i], n_resamples)
            draws[i] = np.column_stack([xu, nu[i] - xu, xd, nd[i] - xd, np.full(n_resamples, row[4])])
    du, dd = draws[:, :, :2].sum(axis=2), draws[:, :, 2:4].sum(axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ds = draws[:, :, 0] / du + draws[:, :, 2] / dd - 1
        qs = du / n[:, None] if design == 'random' else np.broadcast_to(q[:, None], ds.shape)
    gs = aa[:, None] + bb[:, None] * ds
    samples = dict(G=gs, U=qs * ds, H=qs * gs)
    fraction = (~np.isfinite(ds)).mean(axis=1)
    for key, sample in samples.items():
        supported = observed & (fraction <= 0.01)
        if method == 'percentile':
            lo, hi = _quantile(sample, failure / 2), _quantile(sample, 1 - failure / 2)
        else:
            lo, hi, jack_supported = _bca_bounds(context, key, sample)
            supported &= jack_supported
        supported &= (hi > lo) | ((key == 'G') & (bb == 0))
        bounds[key] = _summary(points[key], lo, hi, supported)
    return bounds


def _bca_bounds(context, key, sample):
    """Construct bca intervals from validated counts."""
    c, n, nu, nd, aa, bb, pu, pd, q, points, failure, design = (
        context['c'],
        context['n'],
        context['nu'],
        context['nd'],
        context['aa'],
        context['bb'],
        context['pu'],
        context['pd'],
        context['q'],
        context['points'],
        context['failure'],
        context['design'],
    )
    # Exact cell-weighted row-deletion jackknife under random design;
    # conditional linear-binomial influence contributions under fixed design.
    deleted = c[:, None, :] - np.eye(5)[None, :, :]
    dnu, dnd = deleted[:, :, :2].sum(axis=2), deleted[:, :, 2:4].sum(axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        dj = deleted[:, :, 0] / dnu + deleted[:, :, 2] / dnd - 1
        qj = dnu / (n[:, None] - 1) if design == 'random' else q[:, None]
    gj = aa[:, None] + bb[:, None] * dj
    jack = {'G': gj, 'U': qj * dj, 'H': qj * gj}[key]
    jack_supported = np.all(np.isfinite(jack) | (c == 0), axis=1)
    if design == 'random':
        mean = np.nansum(c * jack, axis=1) / n
        diff = np.where(c > 0, mean[:, None] - jack, 0)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            diff = np.column_stack([(1 - pu) / nu, -pu / nu, (1 - pd) / nd, -pd / nd, np.zeros(len(c))])
        if key == 'G':
            diff *= bb[:, None]
        if key == 'U':
            diff *= q[:, None]
        if key == 'H':
            diff *= (q * bb)[:, None]
    num, den = (c * diff**3).sum(axis=1), 6 * ((c * diff**2).sum(axis=1) ** 1.5)
    acceleration = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    finite = np.isfinite(sample)
    numerator = ((sample < points[key][:, None]) + 0.5 * (sample == points[key][:, None])).sum(axis=1)
    pbelow = np.divide(numerator, finite.sum(axis=1), out=np.full(len(c), 0.5), where=finite.sum(axis=1) > 0)
    z0 = norm.ppf(np.clip(pbelow, 1e-10, 1 - 1e-10))

    def adjusted(t):
        zz = z0 + norm.ppf(t)
        with np.errstate(divide='ignore', invalid='ignore'):
            return norm.cdf(z0 + zz / (1 - acceleration * zz))

    plo, phi = adjusted(failure / 2), adjusted(1 - failure / 2)
    jack_supported &= np.isfinite(plo) & np.isfinite(phi) & (plo <= phi)
    lo, hi = _quantile(sample, np.nan_to_num(plo)), _quantile(sample, np.nan_to_num(phi, nan=1))

    return lo, hi, jack_supported


def _posterior_intervals(context, rng):
    """Construct posterior intervals from validated counts."""
    c, nu, nd, n, q, aa, bb, observed, points, failure, method, n_mc = (
        context['c'],
        context['nu'],
        context['nd'],
        context['n'],
        context['q'],
        context['aa'],
        context['bb'],
        context['observed'],
        context['points'],
        context['failure'],
        context['method'],
        context['n_mc'],
    )
    prior = 0.5 if method.startswith('jeffreys') else 1.0
    psu = rng.beta(c[:, 0, None] + prior, (nu - c[:, 0])[:, None] + prior, size=(len(c), n_mc))
    psd = rng.beta(c[:, 2, None] + prior, (nd - c[:, 2])[:, None] + prior, size=(len(c), n_mc))
    ds = psu + psd - 1
    qs = (
        rng.beta(nu[:, None] + prior, (n - nu)[:, None] + prior, size=(len(c), n_mc))
        if method.endswith('_rd')
        else q[:, None]
    )
    gs = aa[:, None] + bb[:, None] * ds
    return {
        key: _summary(points[key], _quantile(draw, failure / 2), _quantile(draw, 1 - failure / 2), observed)
        for key, draw in dict(G=gs, U=qs * ds, H=qs * gs).items()
    }


def decide_intervals(interval, threshold=0.0, strict=True):
    """Classify a count-interface result using an explicit boundary convention.

    Parameters
    ----------
    interval : dict
        One estimand result returned by infer_counts.
    threshold : float
        Fixed finite utility or contrast threshold.
    strict : bool
        Require a strictly positive lower excess when True. False retains
        the legacy helper's weak lower-bound acceptance convention.

    Returns
    -------
    numpy.ndarray
        accept, reject, uncertain or unsupported labels for each rule.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    lo, hi = np.asarray(interval['lower']), np.asarray(interval['upper'])
    supported = np.asarray(interval['supported']) & np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
    labels = np.full(lo.shape, 'unsupported', dtype='<U11')
    labels[supported] = 'uncertain'
    labels[supported & (hi < threshold)] = 'reject'
    above = lo > threshold if strict else lo >= threshold
    labels[supported & above] = 'accept'
    return labels
