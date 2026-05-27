"""Out-of-sample evaluation for action rules: cross-validation and targeting metrics.

This module adds K-fold cross-validation around ``ActionRules.fit`` together
with rule-level targeting metrics (uplift@k, Qini, AUUC, profit@k).  The CV
results capture both **stability** (mean ± std across folds) and an optional
**stratified bootstrap CI over out-of-fold rules** (Bates, Hastie & Tibshirani,
"Cross-Validation: What does it estimate and how well does it do it?",
arXiv:2104.00673, 2021 — naive CV intervals can have below-nominal coverage,
so we report fold spread separately from bootstrap-derived CIs).

Targeting metric references
---------------------------
- Radcliffe, N. J. (2007) "Using Control Groups to Target on Predicted Lift:
  Building and Assessing Uplift Models." Direct Marketing Analytics Journal.
- Devriendt, F., Moldovan, D., Verbeke, W. (2018) "A Literature Survey and
  Experimental Evaluation of the State-of-the-Art in Uplift Modeling: A
  Stepping Stone Toward the Development of Prescriptive Analytics for
  Business Analytics." Big Data 6 (1), arXiv:1812.04344.
"""

from .cv import CrossValidationResult, CrossValidator, cross_validate
from .metrics import (
    auuc,
    incremental_profit_at_k,
    qini_coefficient,
    qini_curve,
    realistic_gain_at_k,
    uplift_at_k,
)

__all__ = [
    'CrossValidator',
    'CrossValidationResult',
    'cross_validate',
    'uplift_at_k',
    'qini_curve',
    'qini_coefficient',
    'auuc',
    'incremental_profit_at_k',
    'realistic_gain_at_k',
]
