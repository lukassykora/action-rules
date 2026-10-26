#!/bin/sh
# Run from the extracted research artifact; requires the documented Python environment.
set -eu
STUDY_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STUDY_PYTHON=${JIIS_PYTHON:-python3}
cd "$STUDY_DIR"
mkdir -p results logs tmp
export MPLCONFIGDIR="$STUDY_DIR/tmp/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
# E2 resumes from checkpoints. Delete e2_*_replicates.csv only to start afresh.
"$STUDY_PYTHON" scripts/e1_calibration.py
"$STUDY_PYTHON" scripts/e2_plasmode.py
"$STUDY_PYTHON" scripts/summarize_e2.py
"$STUDY_PYTHON" scripts/e3_e4_benchmarks.py
"$STUDY_PYTHON" scripts/e5_runtime.py
"$STUDY_PYTHON" scripts/generate_displays.py
printf '%s\n' 'E1-E5 completed. Timing results depend on this machine; rebuild the manuscript with the documented LaTeX command.'
