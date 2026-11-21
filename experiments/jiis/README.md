# JIIS reproducibility scripts (studies E1–E5)

These scripts accompany *Uncertainty Quantification for High-Utility Action Rules: Estimands, Calibration, and Honest Validation*.

From the action-rules repository root, use Python 3.12 and run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[inference]' -r experiments/jiis/requirements-research.txt
JIIS_PYTHON="$PWD/.venv/bin/python" sh experiments/jiis/scripts/reproduce.sh
```

The requirements file pins the direct dependencies; the editable installation supplies the package from this checkout. The complete manuscript research archive additionally contains the exact input snapshots, hashes and recorded results. Its separate installation instructions apply to its own directory layout.

- E1: 24 regimes with 2,000 generated datasets each, nine interval configurations and three estimands. This is 48,000 datasets and 1,296,000 target-specific interval records, not independent datasets for each method.
- E2: 1,000 Attrition, 400 Telco, 300 Bank and 300 Credit replicates; the miner runs in each replicate. Summarization retains empty families in the family false-acceptance denominator.
- E3: full-data decisions and sensitivity analyses.
- E4: ten repetitions of five-fold cross-validation per dataset, with five ranking strategies.
- E5: one warm-up and seven serial timings; bootstrap B=999 and posterior M=10,000. Timings are hardware-dependent.

Dataset loaders use bundled input files or the primary download locations. To reproduce the archived analysis with its exact inputs, place the archive's data directory under experiments/jiis/data before running. Check source data licenses before redistribution. E2 resumes existing checkpoints; remove only the relevant copied e2_*_replicates.csv files for a fresh E2 run.

The run writes results, logs, caches and generated displays under experiments/jiis. An end-to-end run is computationally intensive. Read E1 CSVs with keep_default_na=False so the effect label `null` is preserved.
