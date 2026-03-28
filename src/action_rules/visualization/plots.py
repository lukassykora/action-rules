"""Visualization functions for action rule confidence interval results.

All functions lazy-import matplotlib so that the module can be imported
without matplotlib installed. scipy is also lazy-imported inside
``posterior_plot``.
"""

from typing import Dict, List, Optional, Tuple


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
    """
    plt = _import_matplotlib()

    point, ci_lower, ci_upper, samples = _get_metric_values(result, metric)

    if samples is None:
        raise ValueError(
            f"Rule {result.rule_index}: samples are None for metric '{metric}'. "
            "Use bootstrap or Bayesian engine to populate samples."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Density-normalised histogram of bootstrap/posterior samples.
    ax.hist(samples, bins=bins, density=True, color='steelblue', alpha=0.7, label='Samples')

    # Shade the CI region.
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='red', label=f'{result.confidence_level:.0%} CI')

    # Vertical lines for CI bounds.
    ax.axvline(ci_lower, color='red', linestyle='--', linewidth=1.2)
    ax.axvline(ci_upper, color='red', linestyle='--', linewidth=1.2)

    # Vertical line for point estimate.
    ax.axvline(point, color='black', linestyle='-', linewidth=1.5, label=f'Point estimate ({point:.4f})')

    # Compute data-driven x-axis limits before adding reference lines.
    # This prevents the threshold line from stretching the axis when the data
    # is far from the threshold (e.g. gain ~ 250 but threshold = 0).
    ax.autoscale_view()
    x_lo, x_hi = ax.get_xlim()

    # Optional threshold line.
    if threshold is not None:
        ax.axvline(threshold, color='gray', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold})')
        ax.set_xlim(x_lo, x_hi)  # Restore data-driven limits.

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
    """
    plt = _import_matplotlib()
    stats = _import_scipy_stats()

    import numpy as np

    point, ci_lower, ci_upper, samples = _get_metric_values(result, metric)

    if samples is None:
        raise ValueError(
            f"Rule {result.rule_index}: samples are None for metric '{metric}'. "
            "Use bootstrap or Bayesian engine to populate samples."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Build a Gaussian KDE for smooth density estimation.
    kde = stats.gaussian_kde(samples)
    x_min = samples.min() - 0.1 * (samples.max() - samples.min() + 1e-9)
    x_max = samples.max() + 0.1 * (samples.max() - samples.min() + 1e-9)
    x_grid = np.linspace(x_min, x_max, 500)
    y_grid = kde(x_grid)

    ax.plot(x_grid, y_grid, color='steelblue', linewidth=2, label='KDE')

    # Shade credible interval region under the KDE curve.
    mask = (x_grid >= ci_lower) & (x_grid <= ci_upper)
    ax.fill_between(
        x_grid[mask],
        y_grid[mask],
        alpha=0.3,
        color='red',
        label=f'{result.confidence_level:.0%} CI',
    )

    # Posterior mean line.
    posterior_mean = float(np.mean(samples))
    ax.axvline(posterior_mean, color='black', linestyle='-', linewidth=1.5, label=f'Mean ({posterior_mean:.4f})')

    # Annotate CI bounds.
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
        Y-axis labels for each rule. Default: ``"Rule 0"``, ``"Rule 1"``, ...
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
    """
    plt = _import_matplotlib()

    n = len(results)
    if n == 0:
        fig, ax_new = plt.subplots(figsize=(figsize[0], 4))
        ax_new.text(0.5, 0.5, 'No results to display', ha='center', va='center', transform=ax_new.transAxes)
        return fig

    # Build a sorted index (highest point estimate at the top = last y position).
    points = []
    lowers = []
    uppers = []
    for r in results:
        pt, lo, hi, _ = _get_metric_values(r, metric)
        points.append(pt)
        lowers.append(lo)
        uppers.append(hi)

    # Sort ascending so that highest value ends up at the top of the y-axis.
    order = sorted(range(n), key=lambda i: points[i])

    if labels is None:
        labels = [f"Rule {results[i].rule_index}" for i in range(n)]

    sorted_labels = [labels[i] for i in order]
    sorted_points = [points[i] for i in order]
    sorted_lowers = [lowers[i] for i in order]
    sorted_uppers = [uppers[i] for i in order]

    # Determine colour for each rule.
    def _rule_color(lo: float, hi: float) -> str:
        if threshold is None:
            return 'C0'
        if lo >= threshold:
            return 'green'
        if hi < threshold:
            return 'red'
        return 'orange'

    colors = [_rule_color(sorted_lowers[i], sorted_uppers[i]) for i in range(n)]

    # Auto-scale height.
    fig_h = figsize[1] if figsize[1] is not None else max(4, n * 0.4)
    fig_w = figsize[0] if figsize[0] is not None else 10

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.get_figure()

    y_positions = list(range(n))

    for i in range(n):
        xerr_lo = sorted_points[i] - sorted_lowers[i]
        xerr_hi = sorted_uppers[i] - sorted_points[i]
        ax.errorbar(
            sorted_points[i],
            y_positions[i],
            xerr=[[max(xerr_lo, 0.0)], [max(xerr_hi, 0.0)]],
            fmt='o',
            color=colors[i],
            ecolor=colors[i],
            capsize=4,
            markersize=5,
            linewidth=1.2,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_labels, fontsize=8)

    # Save data-driven x-axis limits before adding reference lines.
    # This prevents threshold/zero lines from stretching the axis when the data
    # is far away (e.g. realistic_rule_gain ~ 250 but threshold = 0).
    ax.autoscale_view()
    x_lo, x_hi = ax.get_xlim()

    if threshold is not None:
        ax.axvline(threshold, color='gray', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold})')
        # Legend entries for categories.
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Accept'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Uncertain'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Reject'),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

        # Annotate each rule with its category label on the right.
        if show_categories:
            category_labels = {'green': 'Accept', 'orange': 'Uncertain', 'red': 'Reject'}
            for i in range(n):
                cat_text = category_labels.get(colors[i], '')
                ax.annotate(
                    cat_text,
                    xy=(sorted_uppers[i], y_positions[i]),
                    xytext=(8, 0),
                    textcoords='offset points',
                    va='center',
                    ha='left',
                    fontsize=8,
                    fontweight='bold',
                    color=colors[i],
                )

    ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)

    # Restore data-driven x-axis limits so reference lines do not stretch the axis.
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
    """
    plt = _import_matplotlib()

    # Collect all rule indices across all method lists.
    all_rule_indices = sorted({r.rule_index for results in results_dict.values() for r in results})
    n_rules = len(all_rule_indices)

    if n_rules == 0:
        fig, ax_new = plt.subplots(figsize=(figsize[0] or 10, 4))
        ax_new.text(0.5, 0.5, 'No results to display', ha='center', va='center', transform=ax_new.transAxes)
        return fig

    # Build a lookup: method -> {rule_index: result}
    method_lookup: Dict[str, Dict[int, object]] = {}
    for method_name, results in results_dict.items():
        method_lookup[method_name] = {r.rule_index: r for r in results}

    methods = list(results_dict.keys())
    n_methods = len(methods)

    # Colour cycle — prefer semantic colours for known method names.
    default_colors = {'bootstrap': 'C0', 'analytic': 'C1', 'bayesian': 'C2'}
    colors = [default_colors.get(m, f'C{i}') for i, m in enumerate(methods)]

    # Vertical offsets to separate methods per rule row.
    if n_methods == 1:
        offsets = [0.0]
    else:
        half = 0.2
        offsets = [round(-half + (2 * half / (n_methods - 1)) * i, 6) for i in range(n_methods)]

    # Auto-scale height.
    fig_h = figsize[1] if figsize[1] is not None else max(4, n_rules * 0.5)
    fig_w = figsize[0] if figsize[0] is not None else 10

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.get_figure()

    y_labels = [f"Rule {idx}" for idx in all_rule_indices]
    y_base = {rule_idx: i for i, rule_idx in enumerate(all_rule_indices)}

    for m_i, method_name in enumerate(methods):
        lookup = method_lookup[method_name]
        offset = offsets[m_i]
        color = colors[m_i]

        for rule_idx in all_rule_indices:
            if rule_idx not in lookup:
                continue
            r = lookup[rule_idx]
            pt, lo, hi, _ = _get_metric_values(r, metric)
            y_pos = y_base[rule_idx] + offset
            xerr_lo = max(pt - lo, 0.0)
            xerr_hi = max(hi - pt, 0.0)
            ax.errorbar(
                pt,
                y_pos,
                xerr=[[xerr_lo], [xerr_hi]],
                fmt='o',
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5,
                linewidth=1.2,
                label=method_name if rule_idx == all_rule_indices[0] else '_nolegend_',
            )

    ax.set_yticks(list(range(n_rules)))
    ax.set_yticklabels(y_labels, fontsize=8)

    # Save data-driven x-axis limits before adding reference lines.
    ax.autoscale_view()
    x_lo, x_hi = ax.get_xlim()

    if threshold is not None:
        ax.axvline(threshold, color='gray', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold})')

    ax.axvline(0, color='black', linewidth=0.8, alpha=0.4)

    # Restore data-driven x-axis limits so reference lines do not stretch the axis.
    ax.set_xlim(x_lo, x_hi)

    metric_label = metric.replace('_', ' ').title()
    ax.set_xlabel(metric_label)
    ax.set_title(f"Grouped Forest Plot: {metric_label}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
