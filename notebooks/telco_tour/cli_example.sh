#!/usr/bin/env bash
# CLI variant of the Telco case study (paper Section 4.4).
# Identical to the paper listing; only the source CSV path points at the
# repository copy of the semicolon-delimited Telco file.
set -euo pipefail

# The repository Telco file is semicolon-delimited. Convert it once because
# the CLI accepts comma-separated input.
python - <<'PY'
import pandas as pd
pd.read_csv("../data/telco.csv", sep=";").to_csv("telco_comma.csv", index=False)
PY

stable='gender,SeniorCitizen,Partner'
flexible='PhoneService,InternetService,OnlineSecurity,'
flexible+='DeviceProtection,TechSupport,StreamingTV'

action-rules --csv_path telco_comma.csv \
  --stable_attributes "$stable" \
  --flexible_attributes "$flexible" \
  --target Churn --undesired_state Yes --desired_state No \
  --min_stable_attributes 2 --min_flexible_attributes 1 \
  --min_undesired_support 220 --min_undesired_confidence 0.6 \
  --min_desired_support  110 --min_desired_confidence  0.6 \
  --use_gpu false \
  --output_json_path telco_rules_cli.json
