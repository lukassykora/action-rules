"""Reproducible Telco case study (paper appendix, action-rules 2.0.1).

Identical to Appendix A of the Rule Challenge 2026 paper; only the data
path points at the repository copy of the semicolon-delimited Telco CSV.
"""
import pandas as pd
from action_rules import ActionRules

df = pd.read_csv("../data/telco.csv", sep=";")

intrinsic = {("Churn","No"): 400.0, ("Churn","Yes"): 0.0,
             ("InternetService","Fiber optic"): 70.0,
             ("InternetService","DSL"): 60.0,
             ("InternetService","No"): 0.0,
             ("OnlineSecurity","Yes"): -10.0, ("OnlineSecurity","No"): 0.0,
             ("DeviceProtection","Yes"): -8.0, ("DeviceProtection","No"): 0.0,
             ("TechSupport","Yes"): -12.0, ("TechSupport","No"): 0.0,
             ("StreamingTV","Yes"): -5.0, ("StreamingTV","No"): 0.0}
transition = {("OnlineSecurity","No","Yes"): -5.0,
              ("DeviceProtection","No","Yes"): -4.0,
              ("TechSupport","No","Yes"): -6.0,
              ("StreamingTV","No","Yes"): -3.0}

ar = ActionRules(
    min_stable_attributes=2, min_flexible_attributes=1,
    min_undesired_support=220, min_desired_support=110,
    min_undesired_confidence=0.6, min_desired_confidence=0.6,
    intrinsic_utility_table=intrinsic,
    transition_utility_table=transition,
)
ar.fit(
    data=df,
    stable_attributes=["gender", "SeniorCitizen", "Partner"],
    flexible_attributes=["PhoneService", "InternetService",
                          "OnlineSecurity", "DeviceProtection",
                          "TechSupport", "StreamingTV"],
    target="Churn",
    target_undesired_state="Yes", target_desired_state="No",
)

# Confidence intervals: bootstrap is resampling-based; analytic is faster.
# Threshold = 150 = the operator's minimum acceptable per-customer return.
ar.confidence_intervals(
    data=df, method="analytic", analytic_type="auto",
    metric="realistic_rule_gain", threshold=150.0,
)

# Export with categories and confidence intervals embedded.
with open("telco_rules.json", "w") as fh:
    fh.write(ar.get_rules().get_export_notation())
