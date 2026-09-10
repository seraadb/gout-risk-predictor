"""
Gout Risk Predictor - Web App
=============================================================================
A Streamlit web app that loads the trained model from
gout_prediction_pipeline.py and lets anyone enter their health information
to get a gout risk prediction, with an explanation of what's driving it.

USAGE:
    streamlit run app.py

This opens a local link (usually http://localhost:8501) in your browser
immediately. See the deployment guide for turning this into a public,
shareable link.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Gout Risk Predictor",
    page_icon="\U0001FA78",
    layout="centered",
)

MODEL_PATH = Path("outputs/gout_model.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()

st.title("\U0001FA78 Gout Risk Predictor")
st.caption(
    "A machine learning tool estimating gout risk from demographic, "
    "metabolic, laboratory, and lifestyle factors, trained and temporally "
    "validated on NHANES survey data (2007-2018)."
)

if bundle is None:
    st.error(
        "No trained model found at `outputs/gout_model.pkl`. Run "
        "`python gout_prediction_pipeline.py` first to train and save a "
        "model, then restart this app."
    )
    st.stop()

pipeline = bundle["pipeline"]
model_name = bundle["model_name"]

with st.expander("About this model", expanded=False):
    st.markdown(f"""
    - **Algorithm:** {model_name}
    - **Trained on:** NHANES 2007-2016 (development set)
    - **Temporally validated on:** NHANES 2017-2018 (unseen, later data)
    - **Internal AUC-ROC:** {bundle['internal_auc_roc']:.3f}
    - **Temporal external AUC-ROC:** {bundle['temporal_auc_roc']:.3f}
    - **Baseline gout prevalence in training data:** {bundle['gout_prevalence_dev']*100:.1f}%

    This tool is for **educational and research demonstration purposes
    only** and is not a substitute for professional medical diagnosis.
    """)

st.divider()
st.subheader("Enter your information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographic**")
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=45)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 2)], format_func=lambda x: x[0])[1]
    race_ethnicity = st.selectbox(
        "Race/Ethnicity",
        options=[
            ("Mexican American", 1), ("Other Hispanic", 2),
            ("Non-Hispanic White", 3), ("Non-Hispanic Black", 4),
            ("Non-Hispanic Asian", 6), ("Other/Multi-racial", 7),
        ],
        format_func=lambda x: x[0],
    )[1]
    poverty_income = st.slider("Income-to-poverty ratio", 0.0, 5.0, 2.0, 0.1)
    education = st.selectbox(
        "Education level",
        options=[
            ("Less than 9th grade", 1), ("9-11th grade", 2),
            ("High school graduate", 3), ("Some college", 4),
            ("College graduate or above", 5),
        ],
        format_func=lambda x: x[0],
    )[1]

    st.markdown("**Metabolic**")
    bmi = st.number_input("BMI (kg/m\u00b2)", min_value=10.0, max_value=70.0, value=27.0, step=0.1)
    waist = st.number_input("Waist circumference (cm)", min_value=40.0, max_value=200.0, value=95.0, step=0.5)

with col2:
    st.markdown("**Laboratory**")
    uric_acid = st.number_input("Serum uric acid (mg/dL)", min_value=1.0, max_value=15.0, value=5.5, step=0.1)
    creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.2, max_value=10.0, value=0.9, step=0.1)
    hdl = st.number_input("HDL cholesterol (mg/dL)", min_value=10.0, max_value=150.0, value=50.0, step=1.0)
    ldl = st.number_input("LDL cholesterol (mg/dL)", min_value=10.0, max_value=300.0, value=110.0, step=1.0)
    fasting_glucose = st.number_input("Fasting glucose (mg/dL)", min_value=50.0, max_value=400.0, value=95.0, step=1.0)
    triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=20.0, max_value=1000.0, value=120.0, step=1.0)

    st.markdown("**Comorbidity & Medications**")
    hypertension = st.selectbox("Diagnosed with hypertension?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    diabetes = st.selectbox("Diagnosed with diabetes?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    diuretic_use = st.selectbox("Currently taking a diuretic?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    nsaid_use = st.selectbox("Regularly taking NSAIDs?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

st.markdown("**Lifestyle**")
col3, col4 = st.columns(2)
with col3:
    alcohol_intake = st.number_input("Alcoholic drinks per day (average)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
    dietary_fiber = st.number_input("Dietary fiber intake (g/day)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
with col4:
    physical_activity = st.selectbox("Regular vigorous physical activity?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    smoking = st.selectbox("Smoking status", options=[("Never smoked", 2), ("Current/former smoker", 1)], format_func=lambda x: x[0])[1]

st.divider()

if st.button("Predict gout risk", type="primary", use_container_width=True):
    tyg_index = np.log((triglycerides * fasting_glucose) / 2) if triglycerides > 0 and fasting_glucose > 0 else np.nan

    input_row = pd.DataFrame([{
        "age": age, "sex": sex, "race_ethnicity": race_ethnicity,
        "poverty_income": poverty_income, "education": education,
        "bmi": bmi, "waist_circumference": waist, "tyg_index": tyg_index,
        "serum_uric_acid": uric_acid, "creatinine": creatinine,
        "hdl_cholesterol": hdl, "ldl_cholesterol": ldl,
        "fasting_glucose": fasting_glucose,
        "hypertension": hypertension, "diabetes": diabetes,
        "diuretic_use": diuretic_use, "nsaid_use": nsaid_use,
        "alcohol_intake": alcohol_intake, "dietary_fiber": dietary_fiber,
        "physical_activity": physical_activity, "smoking": smoking,
    }])
    input_row = input_row[bundle["predictor_columns"]]

    risk_prob = float(pipeline.predict_proba(input_row)[0, 1])

    st.subheader("Result")
    risk_pct = risk_prob * 100

    if risk_prob < 0.05:
        risk_level, color = "Low", "green"
    elif risk_prob < 0.15:
        risk_level, color = "Moderate", "orange"
    else:
        risk_level, color = "Elevated", "red"

    st.metric("Estimated gout risk", f"{risk_pct:.1f}%")
    st.markdown(f"**Risk category:** :{color}[{risk_level}]")
    st.progress(min(risk_prob, 1.0))

    st.caption(
        f"For comparison, baseline gout prevalence in the training "
        f"population was {bundle['gout_prevalence_dev']*100:.1f}%."
    )

    st.info(
        "This is a research/educational estimate based on population "
        "patterns, not a medical diagnosis. Please consult a healthcare "
        "professional for any health concerns.",
        icon="\u2139\ufe0f",
    )

st.divider()
st.caption(
    "Temporal External Validation of Machine Learning Models for Gout "
    "Risk Prediction Using NHANES Data \u2014 Seraa Datta Bhowmik (24BHT0029)"
)
