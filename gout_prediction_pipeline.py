"""
Temporal External Validation of Machine Learning Models for Gout Risk
Prediction Using NHANES Data
=============================================================================
Author: Seraa Datta Bhowmik (24BHT0029)

This script implements the full pipeline described in the project:
  1. Load & merge NHANES cycles (2007-2023)
  2. Define outcome (physician-diagnosed gout) and candidate predictors
  3. Split into a DEVELOPMENT set (2007-2016) and a TEMPORAL EXTERNAL
     TEST set (2017-2023) -- the test set is NEVER touched during
     model training or tuning
  4. Train & internally validate 6 candidate models on the development set
  5. Lock the models and evaluate them ONCE on the temporal test set
  6. Report discrimination (AUC-ROC, AUC-PR), calibration (Brier score,
     calibration curve), Decision Curve Analysis, and subgroup fairness
  7. SHAP interpretability

-----------------------------------------------------------------------------
DATA YOU NEED TO DOWNLOAD FIRST (not included here -- NHANES is public but
too large to bundle):
  https://wwwn.cdc.gov/nchs/nhanes/Default.aspx

For each cycle you need, at minimum:
  - DEMO_<cycle>.XPT   (Demographics)
  - MCQ_<cycle>.XPT    (Medical Conditions questionnaire -> gout outcome)
  - BMX_<cycle>.XPT    (Body measures -> BMI, waist circumference)
  - BIOPRO_<cycle>.XPT or ALB_CR (labs -> uric acid, creatinine, lipids, glucose)
  - DIQ_<cycle>.XPT    (Diabetes questionnaire)
  - BPQ_<cycle>.XPT    (Blood pressure / hypertension questionnaire)
  - RXQ_RX_<cycle>.XPT (Medications -> diuretics, NSAIDs)
  - ALQ_<cycle>.XPT    (Alcohol use)
  - PAQ_<cycle>.XPT    (Physical activity)
  - DBQ_<cycle>.XPT / DR1TOT_<cycle>.XPT (Dietary fiber intake)

Cycle suffixes: 2007-2008="_E", 2009-2010="_F", 2011-2012="_G",
2013-2014="_H", 2015-2016="_I", 2017-2020="_P" (pre-pandemic),
2021-2023="_L" (August 2021-August 2023 cycle). Check the NHANES site for
the exact file/suffix mapping for each cycle you use.

-----------------------------------------------------------------------------
INSTALL DEPENDENCIES:
  pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib \
      scipy statsmodels xport --break-system-packages
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

import xgboost as xgb
import lightgbm as lgb


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CONFIG = {
    "data_dir": Path("./nhanes_raw"),          # where you put downloaded .XPT files
    "output_dir": Path("./outputs"),
    # NHANES cycle groups -- edit paths/suffixes to match what you download
    "development_cycles": ["2007-2008", "2009-2010", "2011-2012",
                            "2013-2014", "2015-2016"],
    # NOTE: 2021-2023 was dropped. NHANES restructured its Prescription
    # Medications file (RXQ_RX) for this cycle -- it now only contains two
    # screener questions (RXQ033, RXQ050), not actual drug names, so a
    # reliable medication-based gout proxy could not be built for it. The
    # temporal test set is therefore 2017-2018 only (self-reported gout,
    # same definition as the development set) -- a narrower but clean,
    # apples-to-apples temporal gap.
    "temporal_test_cycles": ["2017-2018"],
    # NHANES dropped the self-reported gout question after 2017-2018, and
    # its replacement medication file for 2021-2023 turned out to lack drug
    # names entirely (see note above) -- so no cycle currently uses the
    # medication-based proxy. Left in place / empty for future extension.
    "cycles_needing_medication_proxy": set(),
    "gout_medication_keywords": [
        "ALLOPURINOL", "FEBUXOSTAT", "COLCHICINE", "PROBENECID",
        "PEGLOTICASE", "LESINURAD",
    ],
    "random_state": 42,
    "n_cv_folds": 5,
}
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. DATA LOADING & MERGING
# =============================================================================

def load_xpt(filepath):
    """Load a single NHANES .XPT (SAS transport) file into a DataFrame."""
    return pd.read_sas(filepath, format="xport")


def build_gout_medication_flag(rxq_path: Path, keywords) -> pd.DataFrame:
    """
    RXQ_RX is a LONG file: one row per medication reported per person, so it
    cannot be outer-merged directly like the other (one-row-per-person)
    components. This aggregates it down to a single SEQN-level boolean flag:
    did this person report taking ANY gout-specific medication
    (allopurinol, febuxostat, colchicine, probenecid, pegloticase, etc.)?
    Used as a proxy outcome for cycles where NHANES no longer asks about
    gout directly (2021-2023 onward).
    """
    rxq = load_xpt(rxq_path)
    drug_col = None
    for candidate in ["RXDDRUG", "RXDDRGID"]:
        if candidate in rxq.columns:
            drug_col = candidate
            break
    if drug_col is None:
        print(f"  [warn] Could not find a drug-name column in {rxq_path.name}; "
              f"medication proxy will be all-zero for this cycle.")
        return pd.DataFrame({"SEQN": [], "gout_medication_flag": []})

    drug_names = rxq[drug_col].astype(str).str.upper()
    pattern = "|".join(keywords)
    is_gout_med = drug_names.str.contains(pattern, na=False, regex=True)
    rxq["gout_medication_flag"] = is_gout_med.astype(int)

    flag_by_person = (
        rxq.groupby("SEQN")["gout_medication_flag"]
        .max()
        .reset_index()
    )
    return flag_by_person


def load_nhanes_cycle(cycle_dir: Path, cycle_name: str, cfg: dict = None) -> pd.DataFrame:
    """
    Load and merge all component files for a single NHANES cycle on SEQN
    (the unique respondent ID). Adjust filenames to match what you actually
    download for each cycle.
    """
    files = {
        "demo":    cycle_dir / "DEMO.XPT",
        "mcq":     cycle_dir / "MCQ.XPT",
        "bmx":     cycle_dir / "BMX.XPT",
        "biopro":  cycle_dir / "BIOPRO.XPT",   # standard biochem profile: uric acid, creatinine
        "hdl":     cycle_dir / "HDL.XPT",      # HDL cholesterol
        "glu":     cycle_dir / "GLU.XPT",      # fasting glucose (fasting subsample)
        "trigly":  cycle_dir / "TRIGLY.XPT",   # LDL cholesterol + triglycerides (fasting subsample)
        "diq":     cycle_dir / "DIQ.XPT",
        "bpq":     cycle_dir / "BPQ.XPT",
        "alq":     cycle_dir / "ALQ.XPT",
        "paq":     cycle_dir / "PAQ.XPT",
        "diet":    cycle_dir / "DR1TOT.XPT",
    }

    merged = None
    for key, path in files.items():
        if not path.exists():
            print(f"  [warn] {cycle_name}: missing {path.name}, skipping component")
            continue
        df = load_xpt(path)
        if merged is None:
            merged = df
        else:
            cols_before = set(merged.columns) | set(df.columns)
            merged = merged.merge(df, on="SEQN", how="outer")
            suffixed = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y")]
            if suffixed:
                print(f"  [warn] merging {key} caused column-name collisions "
                      f"(suffixed with _x/_y): {suffixed}")

    # RXQ_RX (prescription medications) is long-format and merged separately
    # as a pre-aggregated per-person flag, for cycles that need the
    # medication-based gout proxy (self-report question no longer exists).
    needs_proxy = cfg is not None and cycle_name in cfg.get("cycles_needing_medication_proxy", set())
    rxq_path = cycle_dir / "RXQ_RX.XPT"
    if needs_proxy:
        if rxq_path.exists():
            flag_df = build_gout_medication_flag(rxq_path, cfg["gout_medication_keywords"])
            merged = merged.merge(flag_df, on="SEQN", how="left")
            merged["gout_medication_flag"] = merged["gout_medication_flag"].fillna(0).astype(int)
            n_flagged = merged["gout_medication_flag"].sum()
            print(f"  [info] {cycle_name}: medication proxy applied, "
                  f"{n_flagged} respondents flagged on gout medications")
        else:
            print(f"  [warn] {cycle_name}: RXQ_RX.XPT missing, cannot build "
                  f"medication proxy -- gout outcome will be undefined here")
            merged["gout_medication_flag"] = 0

    merged["nhanes_cycle"] = cycle_name
    return merged


def load_multiple_cycles(cycle_names, data_dir: Path, cfg: dict = None) -> pd.DataFrame:
    """Load and stack multiple NHANES cycles into one long DataFrame."""
    frames = []
    for cycle in cycle_names:
        cycle_dir = data_dir / cycle
        print(f"Loading cycle {cycle} ...")
        frames.append(load_nhanes_cycle(cycle_dir, cycle, cfg=cfg))
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# 3. OUTCOME DEFINITION AND FEATURE ENGINEERING
# =============================================================================

def define_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Outcome: gout, defined two ways depending on what's available per row.

    1. Self-report (preferred): NHANES MCQ160N/MCQ160n, "doctor ever told
       you had gout" -- present in NHANES 2007-2016 and standalone 2017-2018
       only. NHANES dropped this question from 2019-2020 onward.
    2. Medication proxy (fallback): 'gout_medication_flag', built from the
       RXQ_RX file, used for cycles where self-report no longer exists
       (2021-2023). This is a different construct (treated gout vs.
       diagnosed gout) and should be reported as a methodological
       limitation/sensitivity analysis in the paper.

    IMPORTANT: after concatenating multiple cycles, a single dataframe can
    contain BOTH self-report rows (from 2007-2018) and medication-proxy-only
    rows (from 2021-2023) at once -- self-report is simply NaN/absent for
    the proxy-only rows. This resolves the outcome per ROW, not per whole
    dataframe, so both groups get a correct 'gout' value instead of the
    proxy-only rows silently defaulting to 0.
    """
    self_report_candidates = [c for c in df.columns if c.upper().startswith("MCQ160N")]
    has_self_report = len(self_report_candidates) > 0
    has_proxy = "gout_medication_flag" in df.columns

    if not has_self_report and not has_proxy:
        raise KeyError(
            "Could not find a gout outcome column (no MCQ160N-style self-report "
            "column, and no medication proxy flag was built). Columns actually "
            f"present:\n{sorted(df.columns.tolist())}\n"
            "Check that RXQ_RX.XPT is present for cycles needing the "
            "medication proxy, and that cfg['cycles_needing_medication_proxy'] "
            "includes this cycle."
        )

    gout = pd.Series(np.nan, index=df.index)
    outcome_type = pd.Series("undefined", index=df.index)

    if has_self_report:
        outcome_col = self_report_candidates[0]
        self_report_mask = df[outcome_col].notna()
        gout.loc[self_report_mask] = (df.loc[self_report_mask, outcome_col] == 1).astype(int)
        outcome_type.loc[self_report_mask] = "self_report"
        print(f"  {self_report_mask.sum()} rows use self-reported '{outcome_col}'.")

    if has_proxy:
        # Only fill in rows that self-report didn't already cover.
        still_undefined = gout.isna()
        proxy_mask = still_undefined & df["gout_medication_flag"].notna()
        gout.loc[proxy_mask] = df.loc[proxy_mask, "gout_medication_flag"].astype(int)
        outcome_type.loc[proxy_mask] = "medication_proxy"
        print(f"  {proxy_mask.sum()} additional rows use the medication-based proxy.")

    df["gout"] = gout
    df["gout_outcome_type"] = outcome_type
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the candidate predictor set from raw NHANES variables,
    matching the domains listed on the 'Predictors & Candidate Models' slide:
    demographic, metabolic, laboratory, comorbidity/medication, lifestyle.
    Rename / derive columns as needed to match your actual downloaded files.
    """
    feat = pd.DataFrame({"SEQN": df["SEQN"]})

    # --- Demographic ---
    feat["age"]              = df.get("RIDAGEYR")
    feat["sex"]               = df.get("RIAGENDR")          # 1=Male, 2=Female
    feat["race_ethnicity"]    = df.get("RIDRETH3")
    feat["poverty_income"]    = df.get("INDFMPIR")
    feat["education"]         = df.get("DMDEDUC2")

    # --- Metabolic ---
    feat["bmi"]                = df.get("BMXBMI")
    feat["waist_circumference"] = df.get("BMXWAIST")
    # Triglyceride-glucose (TyG) index = ln(fasting TG [mg/dL] * fasting glucose [mg/dL] / 2)
    if {"LBXTR", "LBXGLU"}.issubset(df.columns):
        feat["tyg_index"] = np.log(df["LBXTR"] * df["LBXGLU"] / 2)

    # --- Laboratory ---
    feat["serum_uric_acid"] = df.get("LBXSUA")
    feat["creatinine"]      = df.get("LBXSCR")
    feat["hdl_cholesterol"] = df.get("LBDHDD")
    feat["ldl_cholesterol"] = df.get("LBDLDL")
    feat["fasting_glucose"] = df.get("LBXGLU")

    # --- Comorbidity / medications (derive binary flags -- adjust to your data) ---
    feat["hypertension"] = (df.get("BPQ020") == 1).astype("Int64")
    feat["diabetes"]     = (df.get("DIQ010") == 1).astype("Int64")
    # Diuretic / NSAID use typically requires parsing RXQ_RX drug codes --
    # placeholder flags here; replace with your actual medication-lookup logic.
    feat["diuretic_use"] = df.get("diuretic_flag", pd.Series(np.nan, index=df.index))
    feat["nsaid_use"]    = df.get("nsaid_flag", pd.Series(np.nan, index=df.index))

    # --- Lifestyle ---
    feat["alcohol_intake"]    = df.get("ALQ130")
    feat["dietary_fiber"]     = df.get("DR1TFIBE")
    feat["physical_activity"] = df.get("PAQ650")
    feat["smoking"]           = df.get("SMQ020")

    feat["nhanes_cycle"] = df["nhanes_cycle"].values
    feat["gout"] = df["gout"].values
    feat["gout_outcome_type"] = df.get("gout_outcome_type", pd.Series("unknown", index=df.index)).values
    return feat


PREDICTOR_COLUMNS = [
    "age", "sex", "race_ethnicity", "poverty_income", "education",
    "bmi", "waist_circumference", "tyg_index",
    "serum_uric_acid", "creatinine", "hdl_cholesterol", "ldl_cholesterol",
    "fasting_glucose",
    "hypertension", "diabetes", "diuretic_use", "nsaid_use",
    "alcohol_intake", "dietary_fiber", "physical_activity", "smoking",
]
CATEGORICAL_COLUMNS = ["sex", "race_ethnicity", "education",
                        "hypertension", "diabetes", "diuretic_use",
                        "nsaid_use", "smoking"]
NUMERIC_COLUMNS = [c for c in PREDICTOR_COLUMNS if c not in CATEGORICAL_COLUMNS]


# =============================================================================
# 4. PREPROCESSING PIPELINE
# =============================================================================

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLUMNS),
        ("cat", categorical_pipe, CATEGORICAL_COLUMNS),
    ])


# =============================================================================
# 5. CANDIDATE MODELS
# =============================================================================

def get_candidate_models(random_state=42):
    base_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            eval_metric="logloss", random_state=random_state, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=-1,
            class_weight="balanced", random_state=random_state, n_jobs=-1),
        "SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=random_state,
                      max_iter=5000, dual=False),
            method="sigmoid", cv=3, n_jobs=-1),
    }
    # Stacking ensemble combining the tree-based models with an LR meta-learner
    stack = StackingClassifier(
        estimators=[
            ("rf", base_models["RandomForest"]),
            ("xgb", base_models["XGBoost"]),
            ("lgb", base_models["LightGBM"]),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        stack_method="predict_proba",
        n_jobs=-1,
    )
    base_models["StackingEnsemble"] = stack
    return base_models


# =============================================================================
# 6. INTERNAL VALIDATION (on the 2007-2016 development set only)
# =============================================================================

def internal_cross_validation(X_dev, y_dev, preprocessor, models, cfg):
    """5-fold stratified CV on the development set. Returns a results table."""
    cv = StratifiedKFold(n_splits=cfg["n_cv_folds"], shuffle=True,
                          random_state=cfg["random_state"])
    scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision"}

    results = []
    fitted_pipelines = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        scores = cross_validate(pipe, X_dev, y_dev, cv=cv, scoring=scoring,
                                 n_jobs=-1, return_train_score=False)
        results.append({
            "model": name,
            "cv_auc_roc_mean": scores["test_roc_auc"].mean(),
            "cv_auc_roc_std":  scores["test_roc_auc"].std(),
            "cv_auc_pr_mean":  scores["test_average_precision"].mean(),
            "cv_auc_pr_std":   scores["test_average_precision"].std(),
        })
        # Refit on the FULL development set -> this is the "locked" model
        pipe.fit(X_dev, y_dev)
        fitted_pipelines[name] = pipe
        print(f"  {name:20s}  CV AUC-ROC = {scores['test_roc_auc'].mean():.3f} "
              f"(+/-{scores['test_roc_auc'].std():.3f})")

    return pd.DataFrame(results).sort_values("cv_auc_roc_mean", ascending=False), \
           fitted_pipelines


# =============================================================================
# 7. TEMPORAL EXTERNAL VALIDATION (locked model, no refitting)
# =============================================================================

def temporal_external_validation(fitted_pipelines, X_test, y_test):
    """
    Apply each LOCKED model (already fit on the development set only) to the
    held-out temporal test set exactly once. No re-tuning, no refitting.
    """
    rows = []
    for name, pipe in fitted_pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]

        auc_roc = roc_auc_score(y_test, y_prob)
        auc_pr  = average_precision_score(y_test, y_prob)
        brier   = brier_score_loss(y_test, y_prob)

        rows.append({
            "model": name,
            "temporal_auc_roc": auc_roc,
            "temporal_auc_pr": auc_pr,
            "temporal_brier_score": brier,
        })
        print(f"  {name:20s}  Temporal AUC-ROC = {auc_roc:.3f} | "
              f"AUC-PR = {auc_pr:.3f} | Brier = {brier:.3f}")

    return pd.DataFrame(rows).sort_values("temporal_auc_roc", ascending=False)


def compare_internal_vs_temporal(internal_df, temporal_df):
    """Merge internal CV results with temporal-test results to show drift."""
    merged = internal_df.merge(temporal_df, on="model")
    merged["auc_drop"] = merged["cv_auc_roc_mean"] - merged["temporal_auc_roc"]
    return merged.sort_values("auc_drop", ascending=True)


# =============================================================================
# 8. CALIBRATION CURVE + DECISION CURVE ANALYSIS
# =============================================================================

def plot_calibration_curve(y_true, y_prob, model_name, out_dir: Path):
    import matplotlib.pyplot as plt
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of gout")
    ax.set_title(f"Calibration curve - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"calibration_{model_name}.png", dpi=150)
    plt.close(fig)


def decision_curve_analysis(y_true, y_prob, thresholds=np.arange(0.01, 0.51, 0.01)):
    """
    Simple decision curve analysis: net benefit of using the model at each
    risk threshold, vs. "treat all" and "treat none" strategies.
    """
    n = len(y_true)
    prevalence = y_true.mean()
    net_benefit_model, net_benefit_all = [], []

    for pt in thresholds:
        predicted_positive = y_prob >= pt
        tp = np.sum((predicted_positive) & (y_true == 1))
        fp = np.sum((predicted_positive) & (y_true == 0))
        nb_model = (tp / n) - (fp / n) * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        net_benefit_model.append(nb_model)
        net_benefit_all.append(nb_all)

    return pd.DataFrame({
        "threshold": thresholds,
        "net_benefit_model": net_benefit_model,
        "net_benefit_treat_all": net_benefit_all,
        "net_benefit_treat_none": 0.0,
    })


# =============================================================================
# 9. SUBGROUP FAIRNESS CHECK
# =============================================================================

def subgroup_auc(y_true, y_prob, subgroup_labels: pd.Series):
    """Compute AUC-ROC separately within each subgroup (e.g. sex, race)."""
    rows = []
    for group_value in subgroup_labels.dropna().unique():
        mask = subgroup_labels == group_value
        if y_true[mask].nunique() < 2:
            continue
        rows.append({
            "subgroup": group_value,
            "n": mask.sum(),
            "auc_roc": roc_auc_score(y_true[mask], y_prob[mask]),
        })
    return pd.DataFrame(rows)


def temporal_validation_by_outcome_type(fitted_pipelines, X_test, y_test, outcome_type):
    """
    DIAGNOSTIC: since the 2017-2023 temporal test set mixes two different
    gout definitions (self-report for 2017-2018, medication-proxy for
    2021-2023), a model's overall temporal AUC could look bad -- or
    anomalously bad, like an AUC well below 0.5 -- simply because it
    performs very differently on one subgroup than the other. This computes
    each model's AUC separately within each outcome-definition subgroup, so
    you can see whether a model's collapse is concentrated in the
    medication-proxy portion (a different clinical construct: "currently
    treated for gout" rather than "ever diagnosed with gout") rather than
    reflecting genuine failure to generalize to self-reported gout over time.
    """
    y_test = pd.Series(y_test).reset_index(drop=True)
    outcome_type = pd.Series(outcome_type).reset_index(drop=True)
    rows = []

    for name, pipe in fitted_pipelines.items():
        y_prob = pd.Series(pipe.predict_proba(X_test)[:, 1]).reset_index(drop=True)
        for group_value in outcome_type.unique():
            mask = outcome_type == group_value
            n = mask.sum()
            if y_test[mask].nunique() < 2:
                rows.append({
                    "model": name, "outcome_type": group_value, "n": n,
                    "gout_prevalence": y_test[mask].mean() if n > 0 else np.nan,
                    "auc_roc": np.nan,
                    "note": "only one class present, AUC undefined",
                })
                continue
            auc = roc_auc_score(y_test[mask], y_prob[mask])
            rows.append({
                "model": name, "outcome_type": group_value, "n": n,
                "gout_prevalence": y_test[mask].mean(),
                "auc_roc": auc, "note": "",
            })

    result = pd.DataFrame(rows)
    print("\n  Temporal AUC-ROC by outcome-definition subgroup:")
    print(result.to_string(index=False))
    return result


# =============================================================================
# 10. SHAP INTERPRETABILITY
# =============================================================================

def run_shap_analysis(pipe, X_sample, out_dir: Path, model_name: str):
    import shap
    import matplotlib.pyplot as plt

    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    X_transformed = preprocessor.transform(X_sample)

    # TreeExplainer for tree-based models; KernelExplainer as a fallback
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, X_transformed[:100])
        shap_values = explainer.shap_values(X_transformed[:200])

    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{model_name}.png", dpi=150)
    plt.close()


# =============================================================================
# 11. MAIN PIPELINE
# =============================================================================

def main():
    cfg = CONFIG
    out_dir = cfg["output_dir"]

    print("=" * 70)
    print("STEP 1: Loading development set (NHANES", cfg["development_cycles"][0],
          "-", cfg["development_cycles"][-1], ")")
    print("=" * 70)
    dev_raw = load_multiple_cycles(cfg["development_cycles"], cfg["data_dir"], cfg=cfg)
    dev_raw = define_outcome(dev_raw)
    dev = build_features(dev_raw).dropna(subset=["gout"])

    print("\n" + "=" * 70)
    print("STEP 2: Loading temporal external test set (NHANES",
          cfg["temporal_test_cycles"][0], "-", cfg["temporal_test_cycles"][-1], ")")
    print("=" * 70)
    test_raw = load_multiple_cycles(cfg["temporal_test_cycles"], cfg["data_dir"], cfg=cfg)
    test_raw = define_outcome(test_raw)
    test = build_features(test_raw).dropna(subset=["gout"])

    X_dev, y_dev = dev[PREDICTOR_COLUMNS], dev["gout"]
    X_test, y_test = test[PREDICTOR_COLUMNS], test["gout"]

    print(f"\nDevelopment set: n={len(X_dev)}, gout prevalence={y_dev.mean():.3f}")
    print(f"Temporal test set: n={len(X_test)}, gout prevalence={y_test.mean():.3f}")

    print("\n" + "=" * 70)
    print("STEP 3: Internal 5-fold cross-validation on development set")
    print("=" * 70)
    preprocessor = build_preprocessor()
    models = get_candidate_models(cfg["random_state"])
    internal_results, fitted_pipelines = internal_cross_validation(
        X_dev, y_dev, preprocessor, models, cfg)
    internal_results.to_csv(out_dir / "internal_cv_results.csv", index=False)
    print("\nInternal CV results:\n", internal_results)

    print("\n" + "=" * 70)
    print("STEP 4: TEMPORAL EXTERNAL VALIDATION (locked models, single pass)")
    print("=" * 70)
    temporal_results = temporal_external_validation(fitted_pipelines, X_test, y_test)
    temporal_results.to_csv(out_dir / "temporal_external_results.csv", index=False)
    print("\nTemporal external validation results:\n", temporal_results)

    print("\n" + "=" * 70)
    print("STEP 4b: DIAGNOSTIC - temporal AUC split by outcome-definition subgroup")
    print("(self-reported gout [2017-2018] vs. medication-proxy gout [2021-2023])")
    print("=" * 70)
    outcome_type_breakdown = temporal_validation_by_outcome_type(
        fitted_pipelines, X_test, y_test, test["gout_outcome_type"])
    outcome_type_breakdown.to_csv(out_dir / "temporal_auc_by_outcome_type.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 5: Comparing internal vs. temporal performance (drift)")
    print("=" * 70)
    comparison = compare_internal_vs_temporal(internal_results, temporal_results)
    comparison.to_csv(out_dir / "internal_vs_temporal_comparison.csv", index=False)
    print(comparison[["model", "cv_auc_roc_mean", "temporal_auc_roc", "auc_drop"]])

    # Use the best model by internal CV AUC-ROC for downstream calibration/SHAP
    best_model_name = internal_results.iloc[0]["model"]
    best_pipe = fitted_pipelines[best_model_name]
    y_prob_test = best_pipe.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print(f"STEP 6: Calibration curve + Decision Curve Analysis ({best_model_name})")
    print("=" * 70)
    plot_calibration_curve(y_test.values, y_prob_test, best_model_name, out_dir)
    dca = decision_curve_analysis(y_test.values, y_prob_test)
    dca.to_csv(out_dir / f"decision_curve_{best_model_name}.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 7: Subgroup fairness (sex, race/ethnicity)")
    print("=" * 70)
    for subgroup_col in ["sex", "race_ethnicity"]:
        sub_df = subgroup_auc(y_test.reset_index(drop=True),
                               pd.Series(y_prob_test),
                               test[subgroup_col].reset_index(drop=True))
        sub_df.to_csv(out_dir / f"subgroup_auc_{subgroup_col}.csv", index=False)
        print(f"\n{subgroup_col}:\n", sub_df)

    print("\n" + "=" * 70)
    print(f"STEP 8: SHAP interpretability ({best_model_name})")
    print("=" * 70)
    run_shap_analysis(best_pipe, X_dev.sample(min(500, len(X_dev)),
                       random_state=cfg["random_state"]), out_dir, best_model_name)

    print("\n" + "=" * 70)
    print("STEP 9: Saving trained model for deployment")
    print("=" * 70)
    import joblib
    model_bundle = {
        "pipeline": best_pipe,
        "model_name": best_model_name,
        "predictor_columns": PREDICTOR_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "internal_auc_roc": float(internal_results.iloc[0]["cv_auc_roc_mean"]),
        "temporal_auc_roc": float(
            temporal_results.loc[temporal_results["model"] == best_model_name, "temporal_auc_roc"].iloc[0]
        ),
        "gout_prevalence_dev": float(y_dev.mean()),
    }
    model_path = out_dir / "gout_model.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"  Saved trained '{best_model_name}' pipeline to: {model_path.resolve()}")
    print("  This file is what the web app will load to make predictions.")

    print("\nAll done. Results saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
