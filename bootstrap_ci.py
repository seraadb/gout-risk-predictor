"""
Compute a 95% bootstrap confidence interval for the deployed model's
AUC-ROC on the NHANES temporal external test set (2017-2018), without
retraining anything.

WHY: Table I in the paper reports point-estimate AUC-ROC values only.
Reviewers commonly expect a 95% CI alongside a point estimate on a
held-out test set, so this script resamples the temporal test set (not
the model) 2000 times and reports the 2.5th/97.5th percentile of the
resulting AUC-ROC distribution -- the standard nonparametric bootstrap
approach for a fixed, already-trained classifier.

SCOPE: this computes a CI for the one model you actually deployed
(the saved gout_model.pkl bundle -- XGBoost per your Results section).
Getting CIs for the other five candidate models in Table I would need
their saved prediction arrays (or retraining them), which is a much
bigger ask, so this script does not attempt that. If you want those
too, let me know whether gout_prediction_pipeline.py already saves
per-model predictions anywhere -- if it does, the same bootstrap loop
below can be pointed at each one.

USAGE (same gout_project folder, venv activated):
    python bootstrap_ci.py

Requires outputs/gout_model.pkl to already exist.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import gout_prediction_pipeline as gp  # noqa: E402

CONFIG = gp.CONFIG
out_dir = CONFIG["output_dir"]
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(CONFIG.get("random_state", 42))

print("Loading saved model bundle...")
bundle = joblib.load(out_dir / "gout_model.pkl")
pipe = bundle["pipeline"]
model_name = bundle["model_name"]
predictor_columns = bundle["predictor_columns"]

print("Loading temporal external test set (NHANES", CONFIG["temporal_test_cycles"][0], ")...")
test_raw = gp.load_multiple_cycles(
    CONFIG["temporal_test_cycles"], CONFIG["data_dir"], cfg=CONFIG
)
test_raw = gp.define_outcome(test_raw)
test = gp.build_features(test_raw).dropna(subset=["gout"])

X_test = test[predictor_columns]
y_test = test["gout"].values

print(f"Scoring {model_name} on {len(y_test)} temporal external rows...")
y_proba = pipe.predict_proba(X_test)[:, 1]
point_estimate = roc_auc_score(y_test, y_proba)

print(f"Point-estimate AUC-ROC: {point_estimate:.4f}")
print(f"Running {N_BOOTSTRAP} bootstrap resamples...")

n = len(y_test)
boot_aucs = []
for i in range(N_BOOTSTRAP):
    idx = RNG.integers(0, n, n)
    y_b, p_b = y_test[idx], y_proba[idx]
    if len(np.unique(y_b)) < 2:
        continue  # skip degenerate resamples with only one class present
    boot_aucs.append(roc_auc_score(y_b, p_b))

boot_aucs = np.array(boot_aucs)
ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])

print("\n--- RESULT ---")
print(f"Model: {model_name}")
print(f"Temporal external AUC-ROC: {point_estimate:.4f}")
print(f"95% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]  (n={len(boot_aucs)} valid resamples)")
print("\nSend me this output and I'll add the CI to Table I and the Results text.")
