#!/usr/bin/env bash
# CLI variant of the Telco case study (the "CLI alternative" shown in the
# paper's reproducible-workflow section). Identical to the paper listing with
# two documented differences: the source CSV path points at the repository
# copy of the semicolon-delimited Telco file, and the output is written to
# telco_rules_cli.json so it does not overwrite the telco_rules.json produced
# by case_study.py in the same directory.
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
