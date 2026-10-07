"""
Gout Risk Predictor - Web App
=============================================================================
A Streamlit web app that loads the trained model from
gout_prediction_pipeline.py and lets anyone enter their health information
to get a gout risk prediction, with an explanation of what's driving it.

USAGE:
    streamlit run app.py

This opens a local link (usually http://localhost:8501) in your browser
immediately.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Gout Risk Predictor",
    page_icon="\u2695",
    layout="wide",
)

MODEL_PATH = Path("outputs/gout_model.pkl")

# =============================================================================
# VISUAL IDENTITY
# Deep ink navy + teal, a serif display face for headings, a clean sans for
# body/labels. Matches the palette used across the project's slides and
# paper, so the site, deck, and report read as one consistent project.
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');

html { font-size: 23px; color-scheme: light only; }

:root {
    --ink: #0B2027;
    --teal: #045C64;
    --teal-light: #028090;
    --seafoam: #00A896;
    --paper: #F5F8F7;
    --line: #D8E3E1;
    --text: #16302E;
    --muted: #5C7A76;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color-scheme: light only; }
.stApp { background-color: var(--paper) !important; }
[data-testid="stAppViewContainer"] { background-color: var(--paper) !important; }
.block-container { padding-top: 2.2rem; padding-left: 3rem; padding-right: 3rem; max-width: 96vw; width: 96vw; }

h1, h2, h3, .serif { font-family: 'Source Serif 4', serif; color: var(--ink); }

#MainMenu, footer, header { visibility: hidden; }

/* ---------------------------------------------------------------- Hero ---- */
.hero {
    position: relative;
    background: var(--ink);
    border-radius: 14px;
    padding: 2.6rem 2.6rem 2.1rem 2.6rem;
    margin-bottom: 2.2rem;
    box-shadow: 0 12px 32px rgba(11, 32, 39, 0.18);
}
@media (max-width: 640px) { .hero { padding: 1.8rem 1.5rem; } }

.hero-datetime {
    position: absolute;
    top: 1.6rem;
    right: 2rem;
    text-align: right;
    color: #8FB3AE !important;
    font-size: 1.05rem;
    line-height: 1.4;
}
@media (max-width: 640px) { .hero-datetime { position: static; text-align: left; margin-bottom: 1rem; } }

.hero-eyebrow { color: var(--seafoam) !important; font-size: 1.15rem; font-weight: 500; margin-bottom: 0.8rem; margin-top: 0.3rem; }
.hero-title { color: #FFFFFF !important; font-size: 3.4rem; font-weight: 600; line-height: 1.15; margin: 0 0 1rem 0; font-family: 'Source Serif 4', serif; }
.hero-desc { color: #D7ECE8 !important; font-size: 1.4rem; max-width: 56ch; line-height: 1.55; margin: 0 0 2rem 0; }

.stat-strip { display: flex; flex-wrap: wrap; gap: 0; border-top: 1px solid #1E3A40; padding-top: 1.3rem; }
.stat-strip .stat { flex: 1 1 0; min-width: 220px; padding: 0 2rem; border-left: 1px solid #1E3A40; }
.stat-strip .stat:first-child { padding-left: 0; border-left: none; }
.stat-num { font-family: 'Source Serif 4', serif; font-size: 2.4rem; color: #FFFFFF !important; font-weight: 600; white-space: nowrap; }
.stat-num-text { font-family: 'Source Serif 4', serif; font-size: 1.7rem; color: #FFFFFF !important; font-weight: 600; white-space: nowrap; }
.stat-label { color: #9FC6C0 !important; font-size: 1.1rem; line-height: 1.35; margin-top: 0.3rem; }

/* ------------------------------------------------------ Form sections ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px;
    padding: 0.4rem 0.7rem 0.7rem 0.7rem;
    margin-bottom: 1.3rem;
}
.section-label { font-family: 'Source Serif 4', serif; font-size: 1.7rem; color: var(--ink) !important; margin: 0.6rem 0 1rem 0.1rem; }

.stButton > button {
    background: var(--teal);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.85rem 1rem;
}
.stButton > button:hover { background: var(--teal-light); color: white !important; }
.stButton > button p { font-size: 1.3rem !important; }

/* -------------------------------------------------------- Result card ---- */
.result-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 2rem 2.2rem 1.9rem 2.2rem;
    margin-top: 0.6rem;
    box-shadow: 0 8px 24px rgba(11, 32, 39, 0.08);
}
.result-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 1.3rem; flex-wrap: wrap; gap: 0.6rem; }
.result-pct { font-family: 'Source Serif 4', serif; font-size: 4.2rem; color: var(--ink); font-weight: 600; }
.result-badge { font-size: 1.25rem; font-weight: 600; padding: 0.45rem 1.1rem; border-radius: 20px; align-self: center; }
.badge-low { background: #DFF3ED; color: #0A6B52; }
.badge-moderate { background: #FBECD5; color: #92600B; }
.badge-elevated { background: #FBE1DE; color: #A3341F; }

.gauge-track { position: relative; height: 12px; border-radius: 6px; background: linear-gradient(90deg, #0A6B52 0%, #C7A233 50%, #A3341F 100%); margin: 0.5rem 0 0.6rem 0; }
.gauge-marker { position: absolute; top: -7px; width: 4px; height: 26px; background: var(--ink); border-radius: 2px; }
.gauge-labels { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.95rem; }

.footnote { color: var(--muted) !important; font-size: 1.1rem; line-height: 1.6; }
.precaution-list { margin: 0 0 1rem 0; padding-left: 1.4rem; color: var(--text); font-size: 1.15rem; line-height: 1.75; }
.precaution-list li { margin-bottom: 0.5rem; }

/* ------------------------------------------------- Widget label sizing --- */
.block-container label p,
.block-container [data-testid="stWidgetLabel"] p {
    color: var(--text) !important;
    font-size: 1.2rem !important;
    font-weight: 500;
}
.stNumberInput input,
.stTextInput input,
.stNumberInput div[data-baseweb="input"],
.stTextInput div[data-baseweb="input"],
[data-baseweb="select"] > div,
[data-baseweb="input"],
[data-baseweb="base-input"] {
    background-color: #FCFDFD !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    font-size: 1.22rem !important;
}
[data-baseweb="select"] div { color: var(--text) !important; }
[data-baseweb="select"] svg { fill: var(--text) !important; }
[data-baseweb="slider"] [role="slider"] { background-color: var(--teal) !important; border-color: var(--teal) !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; }
div[data-baseweb="popover"] { background-color: #FFFFFF !important; }
div[data-baseweb="popover"] li { color: var(--text) !important; background-color: #FFFFFF !important; }
.streamlit-expanderHeader p { font-size: 1.25rem !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


def estimate_calorie_target(age, sex, active):
    """Rough adult daily calorie target, adapted from USDA Dietary
    Guidelines estimated-calorie-requirement tables (age/sex/activity
    level only -- no weight collected by this form, so treat as a
    ballpark, not an individualized prescription)."""
    if sex == 1:  # Male
        if age <= 30:
            return 3000 if active else 2600
        elif age <= 50:
            return 2800 if active else 2400
        else:
            return 2400 if active else 2200
    else:  # Female
        if age <= 30:
            return 2400 if active else 2000
        elif age <= 50:
            return 2200 if active else 1800
        else:
            return 2000 if active else 1600


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()

if bundle is None:
    st.error(
        "No trained model found at `outputs/gout_model.pkl`. Run "
        "`python gout_prediction_pipeline.py` first to train and save a "
        "model, then restart this app."
    )
    st.stop()

pipeline = bundle["pipeline"]
model_name = bundle["model_name"]

# ---------------------------------------------------------------- Hero ----
now = datetime.now()
st.markdown(f"""
<div class="hero">
    <div class="hero-datetime">{now.strftime('%A, %B %d, %Y')}<br/>{now.strftime('%I:%M:%S %p')}</div>
    <div class="hero-eyebrow">Temporal External Validation &middot; NHANES 2007&ndash;2018</div>
    <div class="hero-title">Gout Risk Predictor</div>
    <div class="hero-desc">Estimates gout risk from demographic, metabolic, laboratory, and
    lifestyle factors, using a model trained on a decade of NHANES data and
    tested, unmodified, on a later, unseen survey cycle.</div>
    <div class="stat-strip">
        <div class="stat">
            <div class="stat-num">{bundle['internal_auc_roc']:.3f}</div>
            <div class="stat-label">Internal AUC-ROC<br/>(2007&ndash;2016)</div>
        </div>
        <div class="stat">
            <div class="stat-num">{bundle['temporal_auc_roc']:.3f}</div>
            <div class="stat-label">Temporal external AUC-ROC<br/>(2017&ndash;2018)</div>
        </div>
        <div class="stat">
            <div class="stat-num-text">{model_name}</div>
            <div class="stat-label">Deployed algorithm<br/>(best of 6 compared)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("About this model"):
    other_models = ["Logistic Regression", "Random Forest", "XGBoost",
                     "LightGBM", "SVM", "Stacking Ensemble"]
    other_models = [m for m in other_models if m.replace(" ", "") != model_name.replace(" ", "")]
    st.markdown(f"""
    Six machine learning algorithms were trained and internally validated
    on NHANES 2007&ndash;2016 (Logistic Regression, Random Forest, XGBoost,
    LightGBM, SVM, and a Stacking Ensemble). **{model_name}** was selected
    for deployment here after also comparing all six on a temporal
    external test set &mdash; NHANES 2017&ndash;2018, a survey cycle none
    of the models saw during training.

    Algorithms also compared during research: {", ".join(other_models)}.

    Baseline gout prevalence in the training population was
    **{bundle['gout_prevalence_dev']*100:.1f}%**.

    This tool is for educational and research demonstration purposes only
    and is not a substitute for professional medical diagnosis.
    """)

# ------------------------------------------------------------- Form -----
with st.container(border=True):
    st.markdown('<div class="section-label">Demographic</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
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
    with col2:
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

with st.container(border=True):
    st.markdown('<div class="section-label">Metabolic</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        bmi = st.number_input("BMI (kg/m\u00b2)", min_value=10.0, max_value=70.0, value=27.0, step=0.1)
    with col2:
        waist = st.number_input("Waist circumference (cm)", min_value=40.0, max_value=200.0, value=95.0, step=0.5)

with st.container(border=True):
    st.markdown('<div class="section-label">Laboratory</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        uric_acid = st.number_input("Serum uric acid (mg/dL)", min_value=1.0, max_value=15.0, value=5.5, step=0.1)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.2, max_value=10.0, value=0.9, step=0.1)
        hdl = st.number_input("HDL cholesterol (mg/dL)", min_value=10.0, max_value=150.0, value=50.0, step=1.0)
    with col2:
        ldl = st.number_input("LDL cholesterol (mg/dL)", min_value=10.0, max_value=300.0, value=110.0, step=1.0)
        fasting_glucose = st.number_input("Fasting glucose (mg/dL)", min_value=50.0, max_value=400.0, value=95.0, step=1.0)
        triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=20.0, max_value=1000.0, value=120.0, step=1.0)

with st.container(border=True):
    st.markdown('<div class="section-label">Comorbidity & Medications</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        hypertension = st.selectbox("Diagnosed with hypertension?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        diabetes = st.selectbox("Diagnosed with diabetes?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with col2:
        diuretic_use = st.selectbox("Currently taking a diuretic?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        nsaid_use = st.selectbox("Regularly taking NSAIDs?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

with st.container(border=True):
    st.markdown('<div class="section-label">Lifestyle</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        alcohol_intake = st.number_input("Alcoholic drinks per day (average)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
        dietary_fiber = st.number_input("Dietary fiber intake (g/day)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    with col2:
        physical_activity = st.selectbox("Regular vigorous physical activity?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        smoking = st.selectbox("Smoking status", options=[("Never smoked", 2), ("Current/former smoker", 1)], format_func=lambda x: x[0])[1]

st.write("")
predict_clicked = st.button("Predict gout risk", type="primary", use_container_width=True)

# ------------------------------------------------------------ Result ----
if predict_clicked:
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
    risk_pct = risk_prob * 100

    if risk_prob < 0.05:
        risk_level, badge_class = "Low", "badge-low"
    elif risk_prob < 0.15:
        risk_level, badge_class = "Moderate", "badge-moderate"
    else:
        risk_level, badge_class = "Elevated", "badge-elevated"

    marker_pos = min(max(risk_pct, 0), 100)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-top">
            <div class="result-pct">{risk_pct:.1f}%</div>
            <div class="result-badge {badge_class}">{risk_level} risk</div>
        </div>
        <div class="gauge-track">
            <div class="gauge-marker" style="left: calc({marker_pos}% - 2px);"></div>
        </div>
        <div class="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
        <p class="footnote" style="margin-top: 1.1rem;">
            For comparison, baseline gout prevalence in the training
            population was {bundle['gout_prevalence_dev']*100:.1f}%.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "This is a research/educational estimate based on population "
        "patterns, not a medical diagnosis. Please consult a healthcare "
        "professional for any health concerns.",
        icon="\u2139\ufe0f",
    )

    # ----------------------------------------------------- Caused by ----
    general_causes = [
        "Hyperuricemia (chronically elevated blood uric acid) is the "
        "primary underlying cause of gout; excess uric acid can "
        "crystallize as monosodium urate in and around joints.",
        "Reduced kidney excretion of uric acid, including from impaired "
        "kidney function.",
        "Obesity and excess body fat, which increase uric acid "
        "production and reduce its clearance.",
        "Use of certain medications, particularly diuretics "
        "(“water pills”), which can raise uric acid levels.",
        "High alcohol intake, especially beer and spirits.",
        "A diet high in purines (red meat, organ meats, certain "
        "seafood) and in fructose or sugary drinks.",
        "Family history and genetic predisposition affecting how the "
        "body handles uric acid.",
        "Other metabolic conditions such as hypertension, insulin "
        "resistance, and type 2 diabetes, which frequently co-occur "
        "with hyperuricemia.",
    ]

    contributing = []
    if uric_acid >= 6.8:
        contributing.append(
            f"Entered serum uric acid ({uric_acid:.1f} mg/dL) is at or "
            "above 6.8 mg/dL, the solubility threshold above which "
            "urate crystals can form."
        )
    if bmi >= 30:
        contributing.append(
            f"Entered BMI ({bmi:.1f} kg/m²) falls in the obese "
            "range, a well-established gout risk factor."
        )
    if diuretic_use == 1:
        contributing.append(
            "Current diuretic use was indicated — this medication "
            "class is known to raise uric acid levels."
        )
    if alcohol_intake >= 2:
        contributing.append(
            f"Entered alcohol intake ({alcohol_intake:.1f} drinks/day) "
            "is above moderate levels associated with higher gout risk."
        )
    if hypertension == 1:
        contributing.append(
            "A hypertension diagnosis was indicated; hypertension "
            "commonly co-occurs with and compounds hyperuricemia."
        )
    if diabetes == 1:
        contributing.append(
            "A diabetes diagnosis was indicated; insulin resistance is "
            "mechanistically linked to reduced uric acid excretion."
        )
    if creatinine >= 1.3:
        contributing.append(
            f"Entered creatinine ({creatinine:.1f} mg/dL) is on the "
            "higher side, suggesting reduced kidney clearance of uric "
            "acid."
        )

    causes_html = "".join(f"<li>{c}</li>" for c in general_causes)
    if contributing:
        contributing_html = "".join(f"<li>{c}</li>" for c in contributing)
    else:
        contributing_html = (
            "<li>None of the major modifiable risk factors captured by "
            "this form were flagged for the values you entered.</li>"
        )

    st.markdown(f"""
    <div class="result-card" style="margin-top: 1.2rem;">
        <div class="section-label" style="margin-top: 0;">What causes gout?</div>
        <ul class="precaution-list">{causes_html}</ul>
        <p class="footnote" style="margin-bottom: 0.6rem; font-weight: 600; color: var(--ink);">
            Based on your entered values, the following may be contributing:
        </p>
        <ul class="precaution-list">{contributing_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------- Precautions -----
    general_precautions = [
        "Limit high-purine foods: red meat, organ meats, and certain "
        "seafood (sardines, anchovies, mussels, scallops).",
        "Limit alcohol, especially beer and spirits, which are more "
        "strongly linked to gout flares than wine.",
        "Avoid sugary drinks and foods high in fructose/high-fructose "
        "corn syrup.",
        "Stay well hydrated with water throughout the day.",
        "Include low-fat dairy products, which are associated with "
        "lower gout risk in research.",
        "Maintain a healthy weight through gradual, sustainable changes "
        "rather than rapid or crash dieting.",
        "Get regular, moderate physical activity.",
    ]
    elevated_precautions = [
        "Consult a healthcare provider about checking your serum uric "
        "acid level and discussing whether urate-lowering therapy is "
        "appropriate.",
        "Review current medications with a doctor \u2014 some diuretics "
        "can raise uric acid levels.",
        "Ask about managing related conditions (hypertension, diabetes, "
        "kidney function), which often occur alongside elevated gout risk.",
    ]

    precaution_items = general_precautions + (elevated_precautions if risk_level == "Elevated" else [])
    precaution_list_html = "".join(f"<li>{p}</li>" for p in precaution_items)

    st.markdown(f"""
    <div class="result-card" style="margin-top: 1.2rem;">
        <div class="section-label" style="margin-top: 0;">Dietary &amp; lifestyle precautions</div>
        <ul class="precaution-list">{precaution_list_html}</ul>
        <p class="footnote">
            These are general, evidence-based precautions associated with
            lower gout risk in the population, not a personalized
            treatment plan. Always consult a healthcare professional
            before making significant dietary or medication changes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------ Nutrition pie chart --
    calorie_target = estimate_calorie_target(age, sex, physical_activity == 1)
    fiber_target_g = round(calorie_target / 1000 * 14)

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="section-label">Daily nutrition targets</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="footnote">Estimated calorie need: '
            f'<strong style="color: var(--ink);">{calorie_target} kcal/day</strong> '
            f'(rough estimate from age, sex, and activity level only).</p>',
            unsafe_allow_html=True,
        )

        diet_labels = ["Carbohydrates", "Protein", "Fats", "Fiber", "Vitamins & Minerals"]
        diet_shares = [45, 20, 25, 5, 5]
        diet_colors = ["#045C64", "#028090", "#00A896", "#7FB8AE", "#C7A233"]

        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        fig.patch.set_alpha(0)
        wedges, texts, autotexts = ax.pie(
            diet_shares, labels=diet_labels, autopct="%1.0f%%",
            colors=diet_colors, startangle=90,
            textprops={"fontsize": 11, "color": "#16302E"},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax.axis("equal")
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.pyplot(fig, use_container_width=True)

        fiber_gap = dietary_fiber - fiber_target_g
        if fiber_gap < 0:
            fiber_note = (
                f"Your reported fiber intake is about {abs(fiber_gap):.0f} g/day "
                f"below the suggested ~{fiber_target_g} g/day for your calorie "
                "level — higher fiber intake has been associated with lower "
                "gout risk in NHANES-based research."
            )
        else:
            fiber_note = (
                f"Your reported fiber intake already meets or exceeds the "
                f"suggested ~{fiber_target_g} g/day for your calorie level."
            )
        st.markdown(
            f'<p class="footnote">{fiber_note} This chart illustrates a '
            "general, gout-conscious plate composition — moderate protein, "
            "complex carbohydrates over refined sugar, healthy fats, and "
            "enough fiber, vitamins, and minerals from vegetables and fruit. "
            "Limit high-purine protein sources (red/organ meats, certain "
            "seafood) within the protein share. This is a general "
            "illustration, not a personalized diet plan — consult a "
            "registered dietitian for one.</p>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------- Physical activity guide --
    with st.container(border=True):
        st.markdown('<div class="section-label">Recommended physical activities</div>', unsafe_allow_html=True)

        base_exercises = [
            ("Brisk walking", "30 minutes, most days of the week",
             "Low-impact aerobic activity; easy on joints and a strong first choice."),
            ("Swimming / water aerobics", "20–30 minutes, 2–3 times/week",
             "Minimal joint stress while still building cardiovascular fitness."),
            ("Cycling (stationary or outdoor)", "20–30 minutes, 3–4 times/week",
             "Low-impact aerobic option that's gentle on the knees and ankles."),
            ("Bodyweight strength (squats, lunges, wall push-ups)", "2 sets of 10–15 reps, 2 times/week",
             "Builds muscle support around joints; skip reps that cause joint pain."),
            ("Yoga / stretching", "15–20 minutes, daily",
             "Improves joint mobility and flexibility; go gently around affected joints."),
        ]
        higher_impact = [
            ("Running / jogging", "20–30 minutes, up to 3 times/week",
             "Higher-impact option for general fitness — best once risk is low and there's no joint pain; avoid during a flare."),
        ]

        exercises = base_exercises + (higher_impact if risk_level == "Low" else [])

        exercise_html = "".join(
            f'<li><strong style="color: var(--ink);">{name}</strong> — {freq}<br/>'
            f'<span style="color: var(--muted); font-size: 1.05rem;">{note}</span></li>'
            for name, freq, note in exercises
        )
        st.markdown(f'<ul class="precaution-list">{exercise_html}</ul>', unsafe_allow_html=True)

        if risk_level != "Low":
            st.markdown(
                '<p class="footnote">Given your estimated risk level, favor the '
                "low-impact activities above (walking, swimming, cycling, gentle "
                "strength work, yoga) over running or other high-impact training, "
                "and avoid exercising an actively inflamed joint during a flare.</p>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p class="footnote">General adult guideline: about 150 minutes/week '
            "of moderate aerobic activity (or 75 minutes/week of vigorous "
            "activity), plus muscle-strengthening activity on 2 or more "
            "days/week. Build up gradually and consult a healthcare "
            "professional before starting a new exercise routine.</p>",
            unsafe_allow_html=True,
        )

st.write("")
st.markdown(
    '<p class="footnote">Temporal External Validation of Machine Learning '
    'Models for Gout Risk Prediction Using NHANES Data &mdash; '
    'Seraa Datta Bhowmik (24BHT0029)</p>',
    unsafe_allow_html=True,
)
