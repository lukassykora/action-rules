"""Visualization functions for action rule confidence interval results.

All ``*_plot`` functions lazy-import matplotlib so that the module can be
imported without matplotlib installed. scipy is also lazy-imported inside
``posterior_plot``.

Each plotting function has a sibling ``*_plot_data`` helper that returns the
raw arrays and labels used to render the plot. These data helpers do not
require matplotlib and are intended for downstream consumers (article-grade
figure scripts, custom dashboards, exports) that need direct access to the
underlying numbers.
"""

from typing import Any, Dict, List, Optional, Tuple


def _import_matplotlib():
    """Lazy-import matplotlib.pyplot and return it.

    Returns
    -------
    module
        ``matplotlib.pyplot``.

    Raises
    ------
    ImportError
        When matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")


def _import_scipy_stats():
    """Lazy-import scipy.stats and return it.

    Returns
    -------
    module
        ``scipy.stats``.

    Raises
    ------
    ImportError
        When scipy is not installed.
    """
    try:
        from scipy import stats

        return stats
    except ImportError:
        raise ImportError("scipy is required for posterior_plot. Install it with: pip install scipy")


def _get_metric_values(result, metric: str) -> Tuple[float, float, float, Optional[object]]:
    """Extract point estimate, CI lower, CI upper, and samples for a given metric.

    Parameters
    ----------
    result : ConfidenceIntervalResult
        A populated result object.
    metric : str
        Either ``'uplift'`` or ``'realistic_rule_gain'``.

    Returns
    -------
    tuple
        ``(point, ci_lower, ci_upper, samples)`` where samples may be None.

    Raises
    ------
    ValueError
        When metric is not ``'uplift'`` or ``'realistic_rule_gain'``, or when
        the required fields are None for ``'realistic_rule_gain'``.
    """
    if metric == 'uplift':
        return result.uplift_point, result.uplift_ci_lower, result.uplift_ci_upper, result.samples_uplift
    elif metric == 'realistic_rule_gain':
        if result.realistic_rule_gain_point is None:
            raise ValueError(
                f"Rule {result.rule_index} has no realistic_rule_gain values. "
                "Run inference with utility tables to populate this field."
            )
        return (
            result.realistic_rule_gain_point,
            result.realistic_rule_gain_ci_lower,
            result.realistic_rule_gain_ci_upper,
            result.samples_gain,
        )
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose 'uplift' or 'realistic_rule_gain'.")


def _categorize(lo: float, hi: float, threshold: Optional[float]) -> Optional[str]:
    """Return ``'accept'`` / ``'reject'`` / ``'uncertain'`` for a CI against a threshold."""
    if threshold is None:
        return None
    if lo >= threshold:
        return 'accept'
    if hi < threshold:
        return 'reject'
    return 'uncertain'


# ---------------------------------------------------------------------------
# Data extraction helpers (no matplotlib required)
# ---------------------------------------------------------------------------


def bootstrap_histogram_data(
    result,
    metric: str = "uplift",
    threshold: Optional[float] = None,
    bins: int = 50,
) -> Dict[str, Any]:
    """Return the raw arrays used by :func:`bootstrap_histogram`.

    Parameters
    ----------
    result : ConfidenceIntervalResult
        Must have ``samples_uplift`` or ``samples_gain`` populated.
    metric : str
        ``'uplift'`` or ``'realistic_rule_gain'``.
    threshold : float, optional
        Decision threshold echoed back in the output.
    bins : int
        Number of histogram bins.

    Returns
    -------
    dict
        Keys:

        - ``samples`` (np.ndarray) — the raw bootstrap / posterior samples
        - ``hist`` (np.ndarray) — density-normalised counts per bin
        - ``bin_edges`` (np.ndarray) — bin edges of length ``bins + 1``
        - ``point`` (float) — point estimate
        - ``ci_lower`` / ``ci_upper`` (float)
        - ``confidence_level`` (float)
        - ``threshold`` (float or None)
        - ``rule_index`` (int)
        - ``method`` (str)
        - ``metric`` (str)

    Raises
    ------
    ValueError
        When the required samples array is None.
    """
    import numpy as np

    point, ci_lower, ci_upper, samples = _get_metric_values(result, metric)
    if samples is None:
        raise ValueError(
            f"Rule {result.rule_index}: samples are None for metric '{metric}'. "
            "Use bootstrap or Bayesian engine to populate samples."
        )

    samples_arr = np.asarray(samples)
    hist, bin_edges = np.histogram(samples_arr, bins=bins, density=True)

    return {
        'samples': samples_arr,
        'hist': hist,
        'bin_edges': bin_edges,
        'point': float(point),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence_level': float(result.confidence_level),
        'threshold': None if threshold is None else float(threshold),
        'rule_index': int(result.rule_index),
        'method': result.method,
        'metric': metric,
    }


def posterior_plot_data(
    result,
    metric: str = "uplift",
    n_grid: int = 500,
) -> Dict[str, Any]:
    """Return the raw arrays used by :func:`posterior_plot`.

    Computes a Gaussian KDE over *n_grid* points spanning the sample support
    (extended by 10 % on each side, mirroring the rendering function).

    Parameters
    ----------
    result : ConfidenceIntervalResult
        Must have ``samples_uplift`` or ``samples_gain`` populated.
    metric : str
        ``'uplift'`` or ``'realistic_rule_gain'``.
    n_grid : int
        Number of points in the KDE evaluation grid.

    Returns
    -------
    dict
        Keys: ``samples``, ``x_grid``, ``kde_density``, ``point``, ``ci_lower``,
        ``ci_upper``, ``posterior_mean``, ``confidence_level``, ``rule_index``,
        ``method``, ``metric``.

    Raises
    ------
    ValueError
        When the required samples array is None.
    ImportError
        When scipy is not installed.
    """
    import numpy as np

    stats = _import_scipy_stats()

    point, ci_lower, ci_upper, samples = _get_metric_values(result, metric)
    if samples is None:
        raise ValueError(
            f"Rule {result.rule_index}: samples are None for metric '{metric}'. "
            "Use bootstrap or Bayesian engine to populate samples."
        )

    samples_arr = np.asarray(samples)
    span = samples_arr.max() - samples_arr.min() + 1e-9
    x_min = samples_arr.min() - 0.1 * span
    x_max = samples_arr.max() + 0.1 * span
    x_grid = np.linspace(x_min, x_max, n_grid)
    kde = stats.gaussian_kde(samples_arr)
    y_grid = kde(x_grid)
    posterior_mean = float(np.mean(samples_arr))

    return {
        'samples': samples_arr,
        'x_grid': x_grid,
        'kde_density': y_grid,
        'point': float(point),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'posterior_mean': posterior_mean,
        'confidence_level': float(result.confidence_level),
        'rule_index': int(result.rule_index),
        'method': result.method,
        'metric': metric,
    }


def forest_plot_data(
    results: List,
    metric: str = "uplift",
    threshold: Optional[float] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return the raw arrays used by :func:`forest_plot`.

    Results are sorted by ascending point estimate (highest at the top of the
    figure, which corresponds to the last position in the returned arrays).

    Parameters
    ----------
    results : list of ConfidenceIntervalResult
    metric : str
        ``'uplift'`` or ``'realistic_rule_gain'``.
    threshold : float, optional
        Decision threshold. When provided, each rule is categorized as
        ``'accept'`` (CI entirely ≥ threshold), ``'reject'`` (CI entirely <
        threshold), or ``'uncertain'`` (CI straddles threshold).
    labels : list of str, optional
        Per-rule labels. Default: ``"Rule {index}"`` derived from
        ``result.rule_index``.

    Returns
    -------
    dict
        Keys: ``labels`` (list[str]), ``rule_indices`` (list[int]),
        ``points`` (list[float]), ``ci_lower`` (list[float]),
        ``ci_upper`` (list[float]), ``xerr_lower`` (list[float]),
        ``xerr_upper`` (list[float]) — non-negative half-widths used directly
        by ``ax.errorbar``, ``categories`` (list[str|None]),
        ``threshold`` (float or None), ``metric`` (str), ``n`` (int).

    Notes
    -----
    Returns ``n=0`` and empty lists when *results* is empty. Callers should
    handle this case as appropriate for their context.
    """
    n = len(results)
    if n == 0:
        return {
            'labels': [],
            'rule_indices': [],
            'points': [],
            'ci_lower': [],
            'ci_upper': [],
            'xerr_lower': [],
            'xerr_upper': [],
            'categories': [],
            'threshold': None if threshold is None else float(threshold),
            'metric': metric,
            'n': 0,
        }

    raw_points: List[float] = []
    raw_lower: List[float] = []
    raw_upper: List[float] = []
    rule_indices: List[int] = []
    for r in results:
        pt, lo, hi, _ = _get_metric_values(r, metric)
        raw_points.append(float(pt))
        raw_lower.append(float(lo))
        raw_upper.append(float(hi))
        rule_indices.append(int(r.rule_index))

    if labels is None:
        labels = [f"Rule {idx}" for idx in rule_indices]
    elif len(labels) != n:
        raise ValueError(f"labels has length {len(labels)} but results has length {n}.")

    order = sorted(range(n), key=lambda i: raw_points[i])
    sorted_labels = [labels[i] for i in order]
    sorted_rule_indices = [rule_indices[i] for i in order]
    sorted_points = [raw_points[i] for i in order]
    sorted_lowers = [raw_lower[i] for i in order]
    sorted_uppers = [raw_upper[i] for i in order]

    xerr_lower = [max(sorted_points[i] - sorted_lowers[i], 0.0) for i in range(n)]
    xerr_upper = [max(sorted_uppers[i] - sorted_points[i], 0.0) for i in range(n)]
    categories = [_categorize(sorted_lowers[i], sorted_uppers[i], threshold) for i in range(n)]

    return {
        'labels': sorted_labels,
        'rule_indices': sorted_rule_indices,
        'points': sorted_points,
        'ci_lower': sorted_lowers,
        'ci_upper': sorted_uppers,
        'xerr_lower': xerr_lower,
        'xerr_upper': xerr_upper,
        'categories': categories,
        'threshold': None if threshold is None else float(threshold),
        'metric': metric,
        'n': n,
    }


def grouped_forest_plot_data(
    results_dict: Dict[str, List],
    metric: str = "uplift",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the raw arrays used by :func:`grouped_forest_plot`.

    Parameters
    ----------
    results_dict : dict
        Method-name → list of ``ConfidenceIntervalResult`` (same shape accepted
        by :func:`grouped_forest_plot`).
    metric : str
        ``'uplift'`` or ``'realistic_rule_gain'``.
    threshold : float, optional
        Threshold passed through unchanged.

    Returns
    -------
    dict
        Keys:

        - ``rule_indices`` (list[int]) — union of rule indices, sorted ascending
        - ``methods`` (list[str]) — preserves insertion order of *results_dict*
        - ``offsets`` (list[float]) — vertical jitter per method (symmetric)
        - ``per_method`` (dict[str, dict]) — for each method name, a dict with
          keys ``points``, ``ci_lower``, ``ci_upper``, ``xerr_lower``,
          ``xerr_upper``, all parallel to ``rule_indices``. Missing rules in a
          given method are filled with ``None``.
        - ``threshold`` (float or None)
        - ``metric`` (str)
        - ``n_rules`` (int)
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    all_rule_indices = sorted({r.rule_index for results in results_dict.values() for r in results})
    n_rules = len(all_rule_indices)

    if n_methods == 1:
        offsets = [0.0]
    elif n_methods == 0:
        offsets = []
    else:
        half = 0.2
        offsets = [round(-half + (2 * half / (n_methods - 1)) * i, 6) for i in range(n_methods)]

    per_method: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for method_name, results in results_dict.items():
        lookup = {r.rule_index: r for r in results}
        points: List[Optional[float]] = []
        lowers: List[Optional[float]] = []
        uppers: List[Optional[float]] = []
        xerr_lo: List[Optional[float]] = []
        xerr_hi: List[Optional[float]] = []
        for rule_idx in all_rule_indices:
            if rule_idx not in lookup:
                points.append(None)
                lowers.append(None)
                uppers.append(None)
                xerr_lo.append(None)
                xerr_hi.append(None)
                continue
            r = lookup[rule_idx]
            pt, lo, hi, _ = _get_metric_values(r, metric)
            points.append(float(pt))
            lowers.append(float(lo))
            uppers.append(float(hi))
            xerr_lo.append(max(float(pt) - float(lo), 0.0))
            xerr_hi.append(max(float(hi) - float(pt), 0.0))
        per_method[method_name] = {
            'points': points,
            'ci_lower': lowers,
            'ci_upper': uppers,
            'xerr_lower': xerr_lo,
            'xerr_upper': xerr_hi,
        }

    return {
        'rule_indices': all_rule_indices,
        'methods': methods,
        'offsets': offsets,
        'per_method': per_method,
        'threshold': None if threshold is None else float(threshold),
        'metric': metric,
        'n_rules': n_rules,
    }


# ---------------------------------------------------------------------------
# Rendering functions
# ---------------------------------------------------------------------------


def bootstrap_histogram(
    result,
    metric: str = "uplift",
    threshold=None,
    bins: int = 50,
    ax=None,
    figsize: Tuple[int, int] = (8, 5),
):
    """Plot bootstrap/MC empirical distribution with CI shading.

    Works with both bootstrap and Bayesian results (any result that has
    ``samples_uplift`` or ``samples_gain`` populated).

    Parameters
    ----------
    result : ConfidenceIntervalResult
        Must have ``samples_uplift`` or ``samples_gain`` populated.
    metric : str
        ``'uplift'`` or ``'realistic_rule_gain'``.
    threshold : float, optional
        If provided, draw a vertical dashed line at this value.
    bins : int
        Number of histogram bins.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    figsize : tuple
        Figure size ``(width, height)`` in inches, used only when *ax* is None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Raises
    ------
    ValueError
        When the required samples array is None.
    ImportError
        When matplotlib is not installed.

    See Also
    --------
    bootstrap_histogram_data : Return the raw arrays without plotting.
    """
    plt = _import_matplotlib()

    data = bootstrap_histogram_data(result, metric=metric, threshold=threshold, bins=bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    samples = data['samples']
    point = data['point']
    ci_lower = data['ci_lower']
    ci_upper = data['ci_upper']

    # Density-normalised histogram of bootstrap/posterior samples.
    ax.hist(samples, bins=bins, density=True, color='steelblue', alpha=0.7, label='Samples')

    # Shade the CI region.
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='red', label=f'{data["confidence_level"]:.0%} CI')

    # Vertical lines for CI bounds.
    ax.axvline(ci_lower, color='red', linestyle='--', linewidth=1.2)
    ax.axvline(ci_upper, color='red', linestyle='--', linewidth=1.2)

    # Vertical line for point estimate.
    ax.axvline(point, color='black', linestyle='-', linewidth=1.5, label=f'Point estimate ({point:.4f})')

    # Compute data-driven x-axis limits before adding reference lines.
    ax.autoscale_view()
    x_lo, x_hi = ax.get_xlim()

    if data['threshold'] is not None:
        ax.axvline(
            data['threshold'],
            color='gray',
            linestyle='--',
            linewidth=1.2,
            label=f'Threshold ({data["threshold"]})',
        )
        ax.set_xlim(x_lo, x_hi)

    metric_label = metric.replace('_', ' ').title()
    method_label = result.method.title()
    ax.set_title(f"Rule {result.rule_index}: {metric_label} Distribution ({method_label})")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def posterior_plot(
    result,
    metric: str = "uplift",
    ax=None,
    figsize: Tuple[int, int] = (8, 5),
):
    """Plot the posterior/bootstrap distribution as a KDE curve.

    Uses a Gaussian KDE for a smooth density representation of the raw
    bootstrap or posterior samples.

    Parameters
    ----------
    result : ConfidenceIntervalResult
        Must have ``samples_uplift`` or ``samples_gain`` populated.
    metric : str
        ``'uplift'`` or ``'realistic_rule_gain'``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    figsize : tuple
        Figure size ``(width, height)`` in inches, used only when *ax* is None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Raises
    ------
    ValueError
        When the required samples array is None.
    ImportError
        When matplotlib or scipy is not installed.

    See Also
    --------
    posterior_plot_data : Return the raw arrays without plotting.
    """
    plt = _import_matplotlib()
    data = posterior_plot_data(result, metric=metric)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x_grid = data['x_grid']
    y_grid = data['kde_density']
    ci_lower = data['ci_lower']
    ci_upper = data['ci_upper']
    posterior_mean = data['posterior_mean']

    ax.plot(x_grid, y_grid, color='steelblue', linewidth=2, label='KDE')

    mask = (x_grid >= ci_lower) & (x_grid <= ci_upper)
    ax.fill_between(
        x_grid[mask],
        y_grid[mask],
        alpha=0.3,
        color='red',
        label=f'{data["confidence_level"]:.0%} CI',
    )

    ax.axvline(posterior_mean, color='black', linestyle='-', linewidth=1.5, label=f'Mean ({posterior_mean:.4f})')

    ax.annotate(
        f'CI: [{ci_lower:.4f}, {ci_upper:.4f}]',
        xy=(0.98, 0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
    )

    metric_label = metric.replace('_', ' ').title()
    ax.set_title(f"Rule {result.rule_index}: Posterior Distribution of {metric_label}")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def forest_plot(
    results,
    metric: str = "uplift",
    threshold=None,
    ax=None,
    figsize: Tuple[Optional[int], Optional[int]] = (10, None),
    labels=None,
    show_categories: bool = False,
):
    """Forest plot: all rules with point estimate and CI error bars.

    Rules are sorted by their point estimate, highest at the top. When a
    threshold is provided, rules are colour-coded green (accept), red (reject),
    or orange (uncertain) based on whether their CI is entirely above, entirely
    below, or straddles the threshold.

    Parameters
    ----------
    results : list of ConfidenceIntervalResult
        Results to plot, one entry per rule.
    metric : str
        ``'uplift'`` or ``'realistic_rule_gain'``.
    threshold : float, optional
        Vertical reference line. Also controls colour-coding when provided.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    figsize : tuple
        Figure size ``(width, height)`` in inches. Height is auto-scaled when
        the second element is None: ``max(4, n_rules * 0.4)``.
    labels : list of str, optional
        Y-axis labels for each rule. Default: ``"Rule {rule_index}"``.
    show_categories : bool
        If True and a *threshold* is provided, annotate each rule with its
        category label (Accept / Reject / Uncertain) on the right side of
        the plot. Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Raises
    ------
    ImportError
        When matplotlib is not installed.

    See Also
    --------
    forest_plot_data : Return the raw arrays without plotting.
    """
    plt = _import_matplotlib()

    data = forest_plot_data(results, metric=metric, threshold=threshold, labels=labels)

    if data['n'] == 0:
        fig, ax_new = plt.subplots(figsize=(figsize[0] or 10, 4))
        ax_new.text(0.5, 0.5, 'No results to display', ha='center', va='center', transform=ax_new.transAxes)
        return fig

    n = data['n']
    category_to_color = {'accept': 'green', 'reject': 'red', 'uncertain': 'orange', None: 'C0'}
    colors = [category_to_color[c] for c in data['categories']]

    fig_h = figsize[1] if figsize[1] is not None else max(4, n * 0.4)
    fig_w = figsize[0] if figsize[0] is not None else 10

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.get_figure()

    y_positions = list(range(n))

    for i in range(n):
        ax.errorbar(
            data['points'][i],
            y_positions[i],
            xerr=[[data['xerr_lower'][i]], [data['xerr_upper'][i]]],
            fmt='o',
            color=colors[i],
            ecolor=colors[i],
            capsize=4,
            markersize=5,
            linewidth=1.2,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(data['labels'], fontsize=8)

    ax.autoscale_view()
    x_lo, x_hi = ax.get_xlim()

    if threshold is not None:
        ax.axvline(threshold, color='gray', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold})')
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Accept'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Uncertain'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Reject'),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

        if show_categories:
            category_labels = {'accept': 'Accept', 'uncertain': 'Uncertain', 'reject': 'Reject'}
            for i in range(n):
                cat = data['categories'][i]
                cat_text = category_labels.get(cat, '')
                ax.annotate(
                    cat_text,
                    xy=(data['ci_upper'][i], y_positions[i]),
                    xytext=(8, 0),
                    textcoords='offset points',
                    va='center',
                    ha='left',
                    fontsize=8,
                    fontweight='bold',
                    color=colors[i],
                )

    ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)
    ax.set_xlim(x_lo, x_hi)

    metric_label = metric.replace('_', ' ').title()
    ax.set_xlabel(metric_label)
    ax.set_title(f"Forest Plot: {metric_label}")

    fig.tight_layout()
    return fig


def grouped_forest_plot(
    results_dict: Dict[str, List],
    metric: str = "uplift",
    threshold=None,
    ax=None,
    figsize: Tuple[Optional[int], Optional[int]] = (10, None),
):
    """Grouped forest plot: all rules, multiple methods overlaid.

    For each rule index present across all methods, each method is drawn at a
    small vertical offset so that markers for different methods are visually
    separated. A legend identifies each method.

    Parameters
    ----------
    results_dict : dict
        Keys are method names (e.g. ``'bootstrap'``, ``'analytic'``,
        ``'bayesian'``), values are lists of :class:`ConfidenceIntervalResult`.
    metric : str
        ``'uplift'`` or ``'realistic_rule_gain'``.
    threshold : float, optional
        Vertical reference line at this value.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    figsize : tuple
        Figure size ``(width, height)`` in inches. Height is auto-scaled when
        the second element is None: ``max(4, n_rules * 0.5)``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Raises
    ------
    ImportError
        When matplotlib is not installed.

    See Also
    --------
    grouped_forest_plot_data : Return the raw arrays without plotting.
    """
    plt = _import_matplotlib()

    data = grouped_forest_plot_data(results_dict, metric=metric, threshold=threshold)
    n_rules = data['n_rules']

    if n_rules == 0:
        fig, ax_new = plt.subplots(figsize=(figsize[0] or 10, 4))
        ax_new.text(0.5, 0.5, 'No results to display', ha='center', va='center', transform=ax_new.transAxes)
        return fig

    methods = data['methods']
    offsets = data['offsets']

    default_colors = {'bootstrap': 'C0', 'analytic': 'C1', 'bayesian': 'C2'}
    colors = [default_colors.get(m, f'C{i}') for i, m in enumerate(methods)]

    fig_h = figsize[1] if figsize[1] is not None else max(4, n_rules * 0.5)
    fig_w = figsize[0] if figsize[0] is not None else 10

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.get_figure()

    y_labels = [f"Rule {idx}" for idx in data['rule_indices']]
    y_base = {rule_idx: i for i, rule_idx in enumerate(data['rule_indices'])}

    for m_i, method_name in enumerate(methods):
        method_data = data['per_method'][method_name]
        offset = offsets[m_i]
        color = colors[m_i]
        first_drawn_for_method = True
        for j, rule_idx in enumerate(data['rule_indices']):
            pt = method_data['points'][j]
            if pt is None:
                continue
            y_pos = y_base[rule_idx] + offset
            ax.errorbar(
                pt,
                y_pos,
                xerr=[[method_data['xerr_lower'][j]], [method_data['xerr_upper'][j]]],
                fmt='o',
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5,
                linewidth=1.2,
                label=method_name if first_drawn_for_method else '_nolegend_',
            )
            first_drawn_for_method = False

    ax.set_yticks(list(range(n_rules)))
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.autoscale_view()
    x_lo, x_hi = ax.get_xlim()

    if threshold is not None:
        ax.axvline(threshold, color='gray', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold})')

    ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)
    ax.set_xlim(x_lo, x_hi)

    metric_label = metric.replace('_', ' ').title()
    ax.set_xlabel(metric_label)
    ax.set_title(f"Grouped Forest Plot: {metric_label}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
