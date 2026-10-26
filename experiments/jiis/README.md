# JIIS Reproducibility Scripts (Studies E1–E5)

This directory contains the empirical study scripts for the article:

> *High-Utility Action Rule Mining Under Uncertainty: Confidence Intervals, Sample-Split Inference, and Selection-Robust Ranking*

## Usage

From the repository root with `action-rules` installed:

```bash
pip install -e .
pip install -r experiments/jiis/requirements-research.txt
sh experiments/jiis/scripts/reproduce.sh
```

## Scripts

- `scripts/e1_calibration.py` — E1 synthetic count-table calibration ($10^4$ Monte Carlo replicates per cell across 24 cells).
- `scripts/e2_plasmode.py` and `scripts/summarize_e2.py` — E2 plasmode post-selection study (1,000 resampled datasets per benchmark, comparing full-sample, discovery-half, and honest inference-half coverage).
- `scripts/e3_e4_benchmarks.py` — E3 empirical interval comparison and sensitivity grid, plus E4 repeated $10 \times 5$ cross-validation ranking study.
- `scripts/e5_runtime.py` — E5 vectorised count-level and legacy API wall-clock benchmark.
- `scripts/generate_displays.py` — Regenerates manuscript LaTeX tables and vector figures (`Fig2.pdf`, `Fig3.pdf`) from `results/*.csv`.
