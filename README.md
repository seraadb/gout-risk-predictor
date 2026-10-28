# Gout Risk Predictor

Temporal external validation of machine learning models for gout risk prediction using NHANES data, plus a public web app built on the best model.

**Live app:** https://gout-risk-predictor-gzxqjwjvfx4qkkfpuh57jy.streamlit.app

> **Educational use only.** This tool gives a population-level risk estimate and general lifestyle information. It is not a medical device and is not a substitute for diagnosis or advice from a qualified clinician.

## What this project does

Most NHANES-based gout models are checked only with a random split of one pooled survey window. This project trains on earlier survey cycles, locks the models, and tests them once, without refitting, on a later cycle.

| | NHANES cycles |
|---|---|
| Development (5-fold cross-validation) | 2007–2008, 2009–2010, 2011–2012, 2013–2014, 2015–2016 |
| Temporal external test (used once) | 2017–2018 |

Six models were compared: logistic regression, random forest, XGBoost, LightGBM, a calibrated linear SVM, and a stacking ensemble.

## Results

| Model | Internal CV AUC-ROC | Temporal external AUC-ROC |
|---|---|---|
| Logistic Regression | 0.8360 | 0.7923 |
| Random Forest | 0.8265 | 0.7972 |
| **XGBoost (deployed)** | **0.8405** | **0.8043** (95% CI 0.782–0.827) |
| LightGBM | 0.8214 | 0.7793 |
| SVM (calibrated linear) | 0.8357 | 0.7912 |
| Stacking Ensemble | 0.8372 | 0.7946 |

Calibration, decision curve analysis, SHAP interpretation and subgroup AUC by sex and race/ethnicity are reported in the paper.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `gout_prediction_pipeline.py` | Data loading, feature building, training, validation |
| `download_nhanes.py` | Downloads the NHANES public-use files |
| `regenerate_shap.py` | Rebuilds the SHAP summary plot with feature names |
| `bootstrap_ci.py` | 95% bootstrap confidence interval for the temporal AUC-ROC |
| `requirements.txt` | Python dependencies |
| `outputs/gout_model.pkl` | Saved, locked model used by the app |

## Reproducing the results

1. Install Python 3.12 and the dependencies: `pip install -r requirements.txt`
2. Download the NHANES public-use files into `nhanes_raw/` (see `download_nhanes.py`).
3. Run `python gout_prediction_pipeline.py`. Results are written to `outputs/`.
4. Run the app locally with `streamlit run app.py`.

A fixed random seed (42) is used throughout.

## Data

The NHANES public-use files are published by the CDC National Center for Health Statistics and are not included in this repository. They are used under the [NCHS Data User Agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html): statistical analysis only, with no attempt to identify any person.

Centers for Disease Control and Prevention, National Center for Health Statistics, *National Health and Nutrition Examination Survey (NHANES) public-use data files and documentation*, Hyattsville, MD. https://www.cdc.gov/nchs/nhanes/

## Authors

- Seraa Datta Bhowmik, B.Tech Health Science and Technology, School of Healthcare Science and Engineering (SHINE), Vellore Institute of Technology, Vellore
- M. Monica Subashini, Professor, Department of Control and Automation, School of Electrical Engineering (SELECT), Vellore Institute of Technology, Vellore

## License

Copyright (c) 2026 Seraa Datta Bhowmik. All rights reserved. No license is granted to copy, modify, or redistribute this code without written permission from the author.
