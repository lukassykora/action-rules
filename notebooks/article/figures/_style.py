"""B&W publication-quality styling helpers for the article figures.

Every article figure imports from this module to ensure consistent typography,
grayscale-safe colour and marker mappings, and figure sizing.  Default sizes
match Springer Nature single-column (~3.5 in) and double-column (~7.0 in)
widths.
"""

from __future__ import annotations

from typing import Dict


SINGLE_COL_INCHES = 3.5
DOUBLE_COL_INCHES = 7.0

# Method-specific styling (greyscale-safe).  Methods are distinguished by
# marker shape *and* line/edge style, so colour adds no information.
METHOD_STYLE: Dict[str, Dict] = {
    'bootstrap_percentile': dict(marker='o', linestyle='-', face='white', edge='black'),
    'bootstrap_bca':        dict(marker='s', linestyle='--', face='lightgray', edge='black'),
    'wald':                 dict(marker='^', linestyle=':', face='dimgray', edge='black'),
    'wilson':               dict(marker='D', linestyle='-.', face='black', edge='black'),
    'bayesian':             dict(marker='v', linestyle=(0, (3, 1, 1, 1)), face='gray', edge='black'),
}

METHOD_LABEL: Dict[str, str] = {
    'bootstrap_percentile': 'Bootstrap (pct.)',
    'bootstrap_bca': 'Bootstrap (BCa)',
    'wald': 'Wald',
    'wilson': 'Newcombe-Wilson',
    'bayesian': 'Bayesian',
}

CATEGORY_STYLE = {
    'accept':    dict(marker='o', face='white', edge='black'),
    'uncertain': dict(marker='s', face='lightgray', edge='black'),
    'reject':    dict(marker='^', face='black', edge='black'),
}

DATASET_MARKER = {
    'Telco Customer Churn': 'o',
    'UCI Bank Marketing': 's',
    'IBM Employee Attrition': '^',
    'Taiwan Credit Card Default': 'D',
}


def apply_rc():
    """Apply the Springer Nature-friendly rcParams (call once per notebook)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.size': 9,
            'axes.titlesize': 9,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.dpi': 200,
            'savefig.dpi': 1200,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': False,
            'lines.linewidth': 1.0,
            'patch.linewidth': 0.8,
            'pdf.fonttype': 42,  # embed TrueType so submission graders accept the PDF
            'ps.fonttype': 42,
        }
    )
