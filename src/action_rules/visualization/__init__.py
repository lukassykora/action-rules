"""Visualization module for action rules (requires matplotlib)."""

from .plots import bootstrap_histogram, forest_plot, grouped_forest_plot, posterior_plot

__all__ = ['bootstrap_histogram', 'posterior_plot', 'forest_plot', 'grouped_forest_plot']
