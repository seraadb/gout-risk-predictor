"""
Regenerate the SHAP summary plot with REAL feature names, without
re-running model training (which is the slow part).

Fixes a bug in gout_prediction_pipeline.py's run_shap_analysis(), which
passed a plain NumPy array (no column names) into shap.summary_plot(),
so the chart showed "Feature 0", "Feature 1", ... instead of the actual
predictor names (serum_uric_acid, bmi, sex_Female, etc.).

USAGE (from the same gout_project folder, with venv activated):
    python regenerate_shap.py

Requires outputs/gout_model.pkl to already exist (from your earlier
successful run of gout_prediction_pipeline.py) and the NHANES data to
still be in nhanes_raw/.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, str(Path(__file__).parent))
import gout_prediction_pipeline as gp  # noqa: E402

CONFIG = gp.CONFIG
out_dir = CONFIG["output_dir"]

print("Loading saved model bundle...")
bundle = joblib.load(out_dir / "gout_model.pkl")
pipe = bundle["pipeline"]
model_name = bundle["model_name"]
predictor_columns = bundle["predictor_columns"]

print("Reloading development-set data to rebuild the SHAP input sample...")
dev_raw = gp.load_multiple_cycles(
    CONFIG["development_cycles"], CONFIG["data_dir"], cfg=CONFIG
)
dev_raw = gp.define_outcome(dev_raw)
dev = gp.build_features(dev_raw).dropna(subset=["gout"])
X_dev = dev[predictor_columns]

# Sample up to 300 rows for a faster, still-representative plot
X_sample = X_dev.sample(n=min(300, len(X_dev)), random_state=CONFIG["random_state"])

preprocessor = pipe.named_steps["preprocess"]
model = pipe.named_steps["model"]

X_transformed = preprocessor.transform(X_sample)
feature_names = preprocessor.get_feature_names_out()
# Clean up sklearn's "num__"/"cat__" column-transformer prefixes for readability
feature_names = [f.split("__", 1)[-1] for f in feature_names]
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

print(f"Computing SHAP values for {model_name}...")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed_df)
except Exception:
    explainer = shap.KernelExplainer(model.predict_proba, X_transformed_df[:100])
    shap_values = explainer.shap_values(X_transformed_df[:200])

shap.summary_plot(shap_values, X_transformed_df, show=False)
plt.tight_layout()
out_path = out_dir / f"shap_summary_{model_name}.png"
plt.savefig(out_path, dpi=150)
plt.close()

print("\nDone. Saved:", out_path.resolve())
print("Send this new PNG back and I'll add it to the paper with correct labels.")
