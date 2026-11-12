"""
Employee Attrition Risk - Streamlit app
---------------------------------------
Needs only ONE model file next to this app.py:
    decision_tree_model.pkl   (the tuned DecisionTreeClassifier, 51 input features)

The preprocessing (one-hot, OverTime encoding, scaling) is rebuilt inside this
file, so dtpickle.pkl is NOT required.

Run locally:  streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Risk",
    page_icon="📉",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 100%); color: white;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .9; font-size: 1rem;}
.result-card {padding: 1.4rem 1.6rem; border-radius: 16px; margin-top: .5rem;
              border: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.08);}
.badge {display: inline-block; padding: .25rem .85rem; border-radius: 999px;
        font-weight: 700; font-size: .9rem; color: white;}
.big-number {font-size: 3rem; font-weight: 800; line-height: 1.1; margin: .4rem 0;}
.bar-bg {width: 100%; height: 16px; border-radius: 999px; background: rgba(128,128,128,.3); overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.small-note {font-size: .85rem; opacity: .75;}
div[data-testid="stFormSubmitButton"] button {width: 100%; font-weight: 700; padding: .6rem 0;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    """Load the saved decision tree (tries the repo file name first)."""
    for name in ("decision_tree_model.pkl", "best_tree_cv_model.pkl"):
        path = BASE_DIR / name
        if path.exists():
            return joblib.load(path)
    return None


model = load_model()

# ----------------------------------------------------------------------------
# Preprocessing settings (taken from the IBM HR Attrition data the model was trained on)
# The order here MUST match the order the model was trained with.
# ----------------------------------------------------------------------------
# 1) One-hot columns (categories in alphabetical order, nothing dropped)
OHE_CATEGORIES = {
    "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "EducationField": ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist",
        "Sales Executive", "Sales Representative",
    ],
}

# 2) Columns scaled with StandardScaler: (mean, std)
SCALER_STATS = {
    "Age": (36.92381, 9.132266),
    "DailyRate": (802.485714, 403.371829),
    "DistanceFromHome": (9.192517, 8.104107),
    "MonthlyRate": (14313.103401, 7115.364617),
    "HourlyRate": (65.891156, 20.322512),
    "NumCompaniesWorked": (2.693197, 2.497159),
    "TotalWorkingYears": (11.279592, 7.778135),
    "TrainingTimesLastYear": (2.79932, 1.288832),
    "YearsAtCompany": (7.008163, 6.124441),
    "YearsInCurrentRole": (4.229252, 3.621904),
    "YearsSinceLastPromotion": (2.187755, 3.221334),
    "YearsWithCurrManager": (4.123129, 3.566922),
    "PercentSalaryHike": (15.209524, 3.658693),
}

# 3) Columns passed through without change
PASS_THROUGH = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction",
    "PercentSalaryHike", "RelationshipSatisfaction", "StockOptionLevel",
    "TrainingTimesLastYear", "WorkLifeBalance", "PerformanceRating",
]


def build_features(row: dict) -> np.ndarray:
    """Turn the user's inputs into the 51 numbers the model expects."""
    values = []

    # One-hot encoding (26 values)
    for col, categories in OHE_CATEGORIES.items():
        values += [1.0 if row[col] == c else 0.0 for c in categories]

    # OverTime: No = 0, Yes = 1 (1 value)
    values.append(1.0 if row["OverTime"] == "Yes" else 0.0)

    # Standard scaling (13 values)
    for col, (mean, std) in SCALER_STATS.items():
        values.append((row[col] - mean) / std)

    # Pass-through (11 values)
    values += [float(row[col]) for col in PASS_THROUGH]

    return np.array(values, dtype=float).reshape(1, -1)  # shape (1, 51)


# ----------------------------------------------------------------------------
# Labels for rating-style inputs
# ----------------------------------------------------------------------------
EDUCATION = {1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor", 4: "4 - Master", 5: "5 - Doctor"}
LEVEL4 = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
WORKLIFE = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
PERFORMANCE = {1: "1 - Low", 2: "2 - Good", 3: "3 - Excellent", 4: "4 - Outstanding"}
JOBLEVEL = {1: "Level 1", 2: "Level 2", 3: "Level 3", 4: "Level 4", 5: "Level 5"}
STOCK = {0: "0 - None", 1: "1", 2: "2", 3: "3"}


def pick(label, options: dict, default, key):
    """Dropdown that shows a nice label but returns the number."""
    keys = list(options.keys())
    return st.selectbox(label, keys, index=keys.index(default), format_func=lambda k: options[k], key=key)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>📉 Employee Attrition Risk Predictor</h1>
  <p>Enter an employee's profile to estimate how likely they are to leave the company.</p>
</div>
""",
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "Model file not found. Keep **decision_tree_model.pkl** in the same folder as app.py "
        "(same place in your GitHub repo)."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("attrition_form"):
    tab1, tab2, tab3, tab4 = st.tabs(
        ["👤 Personal", "💼 Job", "💰 Pay & Experience", "😊 Satisfaction"]
    )

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            Age = st.number_input("Age", 18, 65, 35)
            Gender = st.selectbox("Gender", OHE_CATEGORIES["Gender"])
        with c2:
            MaritalStatus = st.selectbox("Marital status", OHE_CATEGORIES["MaritalStatus"], index=1)
            DistanceFromHome = st.number_input("Distance from home (km)", 1, 50, 8)
        with c3:
            Education = pick("Education level", EDUCATION, 3, "edu")
            EducationField = st.selectbox("Education field", OHE_CATEGORIES["EducationField"], index=1)

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            Department = st.selectbox("Department", OHE_CATEGORIES["Department"], index=1)
            JobRole = st.selectbox("Job role", OHE_CATEGORIES["JobRole"], index=6)
        with c2:
            JobLevel = pick("Job level", JOBLEVEL, 2, "joblevel")
            BusinessTravel = st.selectbox("Business travel", OHE_CATEGORIES["BusinessTravel"], index=2)
        with c3:
            OverTime = st.radio("Works overtime?", ["No", "Yes"], horizontal=True)
            PerformanceRating = pick("Performance rating", PERFORMANCE, 3, "perf")

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            DailyRate = st.number_input("Daily rate", 100, 1500, 800, step=10)
            HourlyRate = st.number_input("Hourly rate", 30, 100, 65)
            MonthlyRate = st.number_input("Monthly rate", 2000, 27000, 14000, step=100)
        with c2:
            PercentSalaryHike = st.number_input("Last salary hike (%)", 0, 30, 15)
            StockOptionLevel = pick("Stock option level", STOCK, 0, "stock")
            TrainingTimesLastYear = st.number_input("Trainings last year", 0, 10, 3)
        with c3:
            TotalWorkingYears = st.number_input("Total working years", 0, 45, 10)
            NumCompaniesWorked = st.number_input("Companies worked before", 0, 15, 2)
            YearsAtCompany = st.number_input("Years at this company", 0, 45, 5)

        c4, c5, c6 = st.columns(3)
        with c4:
            YearsInCurrentRole = st.number_input("Years in current role", 0, 30, 3)
        with c5:
            YearsSinceLastPromotion = st.number_input("Years since last promotion", 0, 30, 1)
        with c6:
            YearsWithCurrManager = st.number_input("Years with current manager", 0, 30, 3)

    with tab4:
        c1, c2, c3 = st.columns(3)
        with c1:
            JobSatisfaction = pick("Job satisfaction", LEVEL4, 3, "js")
            EnvironmentSatisfaction = pick("Environment satisfaction", LEVEL4, 3, "es")
        with c2:
            RelationshipSatisfaction = pick("Relationship satisfaction", LEVEL4, 3, "rs")
            JobInvolvement = pick("Job involvement", LEVEL4, 3, "ji")
        with c3:
            WorkLifeBalance = pick("Work-life balance", WORKLIFE, 3, "wlb")

    submitted = st.form_submit_button("🔍 Predict attrition risk", type="primary")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    inputs = {
        "Age": Age, "BusinessTravel": BusinessTravel, "DailyRate": DailyRate, "Department": Department,
        "DistanceFromHome": DistanceFromHome, "Education": Education, "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction, "Gender": Gender, "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement, "JobLevel": JobLevel, "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction, "MaritalStatus": MaritalStatus, "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked, "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike, "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction, "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears, "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance, "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole, "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
    }

    # Basic sanity checks so the numbers make sense
    problems = []
    if YearsAtCompany > TotalWorkingYears:
        problems.append("Years at this company cannot be more than total working years.")
    if YearsInCurrentRole > YearsAtCompany:
        problems.append("Years in current role cannot be more than years at this company.")
    if YearsWithCurrManager > YearsAtCompany:
        problems.append("Years with current manager cannot be more than years at this company.")
    if YearsSinceLastPromotion > YearsAtCompany:
        problems.append("Years since last promotion cannot be more than years at this company.")
    if TotalWorkingYears > Age - 14:
        problems.append("Total working years look too high for this age.")

    if problems:
        for p in problems:
            st.error(p)
    else:
        features = build_features(inputs)
        prob_leave = float(model.predict_proba(features)[0][1])
        pct = prob_leave * 100

        if pct < 30:
            level, color, message = "LOW RISK", "#16a34a", "This employee is likely to stay."
        elif pct < 50:
            level, color, message = "MEDIUM RISK", "#f59e0b", "Some risk signs - worth keeping an eye on."
        else:
            level, color, message = "HIGH RISK", "#dc2626", "This employee is likely to leave."

        left, right = st.columns([3, 2])
        with left:
            st.markdown(
                f"""
<div class="result-card">
<span class="badge" style="background:{color};">{level}</span>
<div class="big-number" style="color:{color};">{pct:.1f}%</div>
<div class="small-note">Probability of leaving the company</div>
<br>
<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div></div>
<br>
<b>{message}</b>
</div>
""",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("##### Quick summary")
            st.write(f"**Role:** {JobRole} ({Department})")
            st.write(f"**Overtime:** {OverTime}")
            st.write(f"**Years at company:** {YearsAtCompany}")
            st.write(f"**Job satisfaction:** {LEVEL4[JobSatisfaction]}")
            st.write(f"**Work-life balance:** {WORKLIFE[WorkLifeBalance]}")

        with st.expander("See all the values used for this prediction"):
            st.dataframe(
                pd.DataFrame({"Feature": list(inputs.keys()), "Value": [str(v) for v in inputs.values()]}),
                hide_index=True,
            )

        st.caption(
            "This is a decision-tree estimate based on historical HR data. "
            "Use it as a guide for conversations, not as the only basis for HR decisions."
        )
else:
    st.info("Fill in the details above and click **Predict attrition risk**.")
