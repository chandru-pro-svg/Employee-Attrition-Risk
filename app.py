"""
Employee Attrition Risk - Streamlit application
===============================================

Predicts whether an employee is likely to leave, using a trained scikit-learn
DecisionTreeClassifier (``decision_tree_model.pkl``, 51 input features) and explains
every prediction with the exact rules the tree followed.

Preprocessing note
------------------
``dtpickle.pkl`` is a ColumnTransformer that was saved *before* ``.fit()`` was called,
so it holds no learned categories or scaling values. This app therefore recreates the
same preprocessing by hand, in the same column order:

    1. One-hot encoding  -> BusinessTravel, Department, MaritalStatus,
                            EducationField, Gender, JobRole          (26 columns)
    2. Label encoding    -> OverTime (No = 0, Yes = 1)               ( 1 column)
    3. Standard scaling  -> 13 numeric columns                       (13 columns)
    4. Pass-through      -> 11 ordinal / numeric columns             (11 columns)
                                                               total = 51 columns

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
MODEL_PATH = Path(__file__).parent / "decision_tree_model.pkl"
EXPECTED_FEATURES = 51

# Risk thresholds on the predicted probability of leaving.
LOW_RISK_BELOW = 0.35
HIGH_RISK_FROM = 0.65


@dataclass(frozen=True)
class Theme:
    """Colour palette used across the interface."""

    ink: str = "#1B2A41"
    teal: str = "#0F766E"
    teal_dark: str = "#0B5E58"
    paper: str = "#F6F7F9"
    low: str = "#2F855A"
    mid: str = "#B7791F"
    high: str = "#C0392B"
    font: str = "IBM Plex Sans"


THEME = Theme()


@dataclass(frozen=True)
class RiskBand:
    """A named risk level with its colour and short message."""

    label: str
    color: str
    message: str


@dataclass(frozen=True)
class Prediction:
    """Everything the result panel needs to display."""

    probability: float
    band: RiskBand
    steps: list[str]
    warnings: list[str]


# =============================================================================
# 2. FEATURE DEFINITIONS (must follow the original ColumnTransformer order)
# =============================================================================
# OneHotEncoder sorts categories alphabetically, so these lists are alphabetical.
OHE_COLUMNS: dict[str, list[str]] = {
    "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "EducationField": [
        "Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree",
    ],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist",
        "Sales Executive", "Sales Representative",
    ],
}

# StandardScaler columns: (name, mean, standard deviation) from the IBM HR dataset.
SCALED_COLUMNS: list[tuple[str, float, float]] = [
    ("Age", 36.92, 9.14),
    ("DailyRate", 802.49, 403.44),
    ("DistanceFromHome", 9.19, 8.11),
    ("MonthlyRate", 14313.10, 7117.00),
    ("HourlyRate", 65.89, 20.33),
    ("NumCompaniesWorked", 2.69, 2.50),
    ("TotalWorkingYears", 11.28, 7.78),
    ("TrainingTimesLastYear", 2.80, 1.29),
    ("YearsAtCompany", 7.01, 6.13),
    ("YearsInCurrentRole", 4.23, 3.62),
    ("YearsSinceLastPromotion", 2.19, 3.22),
    ("YearsWithCurrManager", 4.12, 3.57),
    ("PercentSalaryHike", 15.21, 3.66),
]

PASS_THROUGH_COLUMNS: list[str] = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction",
    "PercentSalaryHike", "RelationshipSatisfaction", "StockOptionLevel",
    "TrainingTimesLastYear", "WorkLifeBalance", "PerformanceRating",
]

DISPLAY_NAMES: dict[str, str] = {
    "BusinessTravel": "Business travel", "Department": "Department",
    "MaritalStatus": "Marital status", "EducationField": "Education field",
    "Gender": "Gender", "JobRole": "Job role", "OverTime": "Works overtime",
    "Age": "Age", "DailyRate": "Daily rate", "DistanceFromHome": "Distance from home (km)",
    "MonthlyRate": "Monthly rate", "HourlyRate": "Hourly rate",
    "NumCompaniesWorked": "Companies worked at", "TotalWorkingYears": "Total working years",
    "TrainingTimesLastYear": "Trainings last year", "YearsAtCompany": "Years at company",
    "YearsInCurrentRole": "Years in current role",
    "YearsSinceLastPromotion": "Years since last promotion",
    "YearsWithCurrManager": "Years with current manager",
    "PercentSalaryHike": "Last salary hike (%)", "Education": "Education level",
    "EnvironmentSatisfaction": "Environment satisfaction", "JobInvolvement": "Job involvement",
    "JobLevel": "Job level", "JobSatisfaction": "Job satisfaction",
    "RelationshipSatisfaction": "Relationship satisfaction",
    "StockOptionLevel": "Stock option level", "WorkLifeBalance": "Work-life balance",
    "PerformanceRating": "Performance rating",
}


def build_feature_catalog() -> list[dict[str, Any]]:
    """Describe each of the 51 model inputs, in model order."""
    catalog: list[dict[str, Any]] = []
    for column, categories in OHE_COLUMNS.items():
        for category in categories:
            catalog.append({
                "kind": "onehot", "category": category, "group": DISPLAY_NAMES[column],
                "label": f"{DISPLAY_NAMES[column]}: {category}",
            })
    catalog.append({"kind": "overtime", "label": DISPLAY_NAMES["OverTime"]})
    for name, mean, std in SCALED_COLUMNS:
        catalog.append({"kind": "scaled", "label": DISPLAY_NAMES[name], "mean": mean, "std": std})
    for name in PASS_THROUGH_COLUMNS:
        catalog.append({"kind": "raw", "label": DISPLAY_NAMES[name]})
    return catalog


FEATURE_CATALOG = build_feature_catalog()
if len(FEATURE_CATALOG) != EXPECTED_FEATURES:
    raise RuntimeError(f"Feature catalog has {len(FEATURE_CATALOG)} entries, expected {EXPECTED_FEATURES}.")

# =============================================================================
# 3. STYLING
# =============================================================================
_STATIC_CSS = """
html, body, [class*="css"], .stApp { font-family: var(--font), sans-serif; }
.stApp { background: var(--paper); }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1250px; }
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.01em; }

.hero { padding: 0.2rem 0 1.2rem 0; }
.hero h1 { font-size: 2.3rem; font-weight: 700; margin: 0 0 .3rem 0; }
.hero p  { font-size: 1.05rem; color: #4A5568; max-width: 46rem; margin: 0; }

.card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.card h4 { margin: 0 0 .6rem 0; font-size: 1.05rem; }

.verdict { border-left: 6px solid var(--c); padding: .2rem 0 .2rem 1rem; }
.verdict .label { font-size: 1.55rem; font-weight: 700; color: var(--c); }
.verdict .sub { color: #4A5568; margin-top: .15rem; }

.step { display: flex; gap: .7rem; padding: .55rem 0; border-bottom: 1px solid #EDF2F7; }
.step:last-child { border-bottom: none; }
.step .n { flex: 0 0 1.7rem; height: 1.7rem; border-radius: 50%; background: var(--teal);
           color: #fff; font-weight: 600; font-size: .85rem;
           display: flex; align-items: center; justify-content: center; }
.step .t { color: var(--ink); font-size: .95rem; }

.placeholder { color: #718096; text-align: center; padding: 3rem 1rem; }

div[data-testid="stFormSubmitButton"] button { width: 100%; background: var(--teal); color: #fff;
    border: none; font-weight: 600; padding: .6rem 0; border-radius: 8px; }
div[data-testid="stFormSubmitButton"] button:hover { background: var(--teal-dark); color: #fff; }
div[data-testid="stFormSubmitButton"] button:focus-visible { outline: 3px solid #99E0D8; }
"""


def inject_styles() -> None:
    """Load the web font and apply the custom stylesheet."""
    variables = (
        f":root {{ --ink: {THEME.ink}; --teal: {THEME.teal}; --teal-dark: {THEME.teal_dark}; "
        f"--paper: {THEME.paper}; --font: '{THEME.font}'; }}"
    )
    font_import = (
        "@import url('https://fonts.googleapis.com/css2?"
        "family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');"
    )
    st.markdown(f"<style>{font_import}\n{variables}\n{_STATIC_CSS}</style>", unsafe_allow_html=True)


# =============================================================================
# 4. MODEL LOADING AND PREPROCESSING
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model() -> DecisionTreeClassifier:
    """Load the trained decision tree once and reuse it across reruns."""
    return joblib.load(MODEL_PATH)


def encode_employee(values: dict[str, Any]) -> np.ndarray:
    """Convert the form answers into the (1, 51) array the tree expects."""
    row: list[float] = []
    for column, categories in OHE_COLUMNS.items():
        row.extend(1.0 if values[column] == category else 0.0 for category in categories)
    row.append(1.0 if values["OverTime"] == "Yes" else 0.0)
    row.extend((values[name] - mean) / std for name, mean, std in SCALED_COLUMNS)
    row.extend(float(values[name]) for name in PASS_THROUGH_COLUMNS)
    return np.array([row])


def validate_inputs(values: dict[str, Any]) -> list[str]:
    """Return plain-language warnings for combinations that look inconsistent."""
    issues: list[str] = []
    if values["YearsAtCompany"] > values["TotalWorkingYears"]:
        issues.append("Years at company is higher than total working years.")
    if values["YearsInCurrentRole"] > values["YearsAtCompany"]:
        issues.append("Years in current role is higher than years at company.")
    if values["YearsWithCurrManager"] > values["YearsAtCompany"]:
        issues.append("Years with current manager is higher than years at company.")
    if values["TotalWorkingYears"] > values["Age"] - 14:
        issues.append("Total working years looks too high for this age.")
    return issues


# =============================================================================
# 5. PREDICTION AND EXPLANATION
# =============================================================================
def classify_risk(probability: float) -> RiskBand:
    """Map a probability to a named risk level."""
    if probability < LOW_RISK_BELOW:
        return RiskBand("Low risk", THEME.low, "This profile looks stable.")
    if probability < HIGH_RISK_FROM:
        return RiskBand("Moderate risk", THEME.mid, "Worth a check-in conversation.")
    return RiskBand("High risk", THEME.high, "This profile resembles employees who left.")


def describe_split(feature_index: int, threshold: float, went_left: bool) -> str:
    """Explain one tree split in plain words, using real units instead of scaled ones."""
    feature = FEATURE_CATALOG[feature_index]
    if feature["kind"] == "onehot":
        verb = "is not" if went_left else "is"
        return f"{feature['group']} {verb} {feature['category']}"
    if feature["kind"] == "overtime":
        return "Does not work overtime" if went_left else "Works overtime"

    if feature["kind"] == "scaled":
        threshold = threshold * feature["std"] + feature["mean"]  # back to real units
    cut = math.floor(threshold)  # every numeric feature here is a whole number
    return f"{feature['label']} ≤ {cut}" if went_left else f"{feature['label']} ≥ {cut + 1}"


def explain_path(model: DecisionTreeClassifier, x: np.ndarray) -> list[str]:
    """List the rules this employee passed through, from the root to the leaf."""
    tree = model.tree_
    steps: list[str] = []
    for node in model.decision_path(x).indices:
        if tree.children_left[node] == -1:  # reached a leaf
            continue
        feature_index = tree.feature[node]
        went_left = x[0, feature_index] <= tree.threshold[node]
        steps.append(describe_split(feature_index, tree.threshold[node], went_left))
    return steps


def predict(model: DecisionTreeClassifier, values: dict[str, Any]) -> Prediction:
    """Run the model for one employee and package the result."""
    x = encode_employee(values)
    probability = float(model.predict_proba(x)[0, 1])
    return Prediction(
        probability=probability,
        band=classify_risk(probability),
        steps=explain_path(model, x),
        warnings=validate_inputs(values),
    )


def list_leaf_rules(model: DecisionTreeClassifier) -> pd.DataFrame:
    """Describe every possible outcome of the tree as one readable row."""
    tree = model.tree_
    rows: list[dict[str, Any]] = []

    def walk(node: int, conditions: list[str]) -> None:
        if tree.children_left[node] == -1:
            counts = tree.value[node][0]
            rows.append({
                "Rule": " AND ".join(conditions),
                "Prediction": "Leaves" if counts[1] / counts.sum() >= 0.5 else "Stays",
                "Who left in training data": f"{counts[1] / counts.sum():.0%}",
                "Employees": int(tree.n_node_samples[node]),
            })
            return
        feature_index, threshold = tree.feature[node], tree.threshold[node]
        walk(tree.children_left[node], conditions + [describe_split(feature_index, threshold, True)])
        walk(tree.children_right[node], conditions + [describe_split(feature_index, threshold, False)])

    walk(0, [])
    return pd.DataFrame(rows)


# =============================================================================
# 6. CHARTS
# =============================================================================
def build_gauge(probability: float, color: str) -> go.Figure:
    """Semi-circular gauge showing the risk percentage."""
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "color": THEME.ink}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 35, 65, 100]},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#EDF2F7", "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "#E6F4EA"},
                {"range": [35, 65], "color": "#FBF0D9"},
                {"range": [65, 100], "color": "#FBE3E0"},
            ],
        },
    ))
    figure.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0),
                         paper_bgcolor="rgba(0,0,0,0)", font={"family": THEME.font})
    return figure


def build_importance_chart(model: DecisionTreeClassifier) -> tuple[go.Figure, int]:
    """Horizontal bar chart of the inputs the tree actually uses."""
    series = pd.Series(model.feature_importances_, index=[f["label"] for f in FEATURE_CATALOG])
    series = series[series > 0].sort_values()
    figure = go.Figure(go.Bar(
        x=series.values * 100, y=series.index, orientation="h", marker_color=THEME.teal,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    figure.update_layout(
        height=60 + 44 * len(series), margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Importance (%)", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font={"family": THEME.font, "color": THEME.ink},
    )
    return figure, len(series)


# =============================================================================
# 7. USER INTERFACE
# =============================================================================
def render_sidebar(model: DecisionTreeClassifier) -> None:
    with st.sidebar:
        st.markdown("### Model facts")
        st.markdown(
            f"- **Algorithm:** Decision Tree Classifier\n"
            f"- **Depth:** {model.get_depth()} levels, {model.get_n_leaves()} outcomes\n"
            f"- **Inputs:** {model.n_features_in_} (after encoding)\n"
            f"- **Predicts:** will the employee leave (yes / no)"
        )
        st.markdown("### How to use")
        st.markdown(
            "1. Fill in the employee profile\n"
            "2. Select **Predict attrition risk**\n"
            "3. Read the score and the rules behind it"
        )
        st.info("This is a decision-support tool. It should not be the only basis for HR decisions.")


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Employee attrition risk</h1>
          <p>Estimate how likely an employee is to leave, and see the exact rules
          the model used to get there.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_extra_details() -> dict[str, Any]:
    """Optional fields the current tree does not use. Returned as a dict of answers."""
    st.caption("The current model does not use these fields, but they are kept so the "
               "form matches the full dataset.")
    scale4 = {1: "1 · Low", 2: "2 · Medium", 3: "3 · High", 4: "4 · Very high"}
    balance = {1: "1 · Bad", 2: "2 · Good", 3: "3 · Better", 4: "4 · Best"}

    c1, c2 = st.columns(2)
    values: dict[str, Any] = {
        "BusinessTravel": c1.selectbox("Business travel", OHE_COLUMNS["BusinessTravel"], index=2),
        "Department": c2.selectbox("Department", OHE_COLUMNS["Department"], index=1),
        "MaritalStatus": c1.selectbox("Marital status", OHE_COLUMNS["MaritalStatus"], index=1),
        "EducationField": c2.selectbox("Education field", OHE_COLUMNS["EducationField"], index=1),
        "Gender": c1.selectbox("Gender", OHE_COLUMNS["Gender"], index=1),
        "Education": c2.select_slider("Education level", options=[1, 2, 3, 4, 5], value=3),
        "DailyRate": c1.number_input("Daily rate", 100, 1500, 800),
        "HourlyRate": c2.number_input("Hourly rate", 30, 100, 66),
        "MonthlyRate": c1.number_input("Monthly rate", 2000, 27000, 14000),
        "PercentSalaryHike": c2.number_input("Last salary hike (%)", 11, 25, 14),
        "DistanceFromHome": c1.number_input("Distance from home (km)", 1, 29, 9),
        "TrainingTimesLastYear": c2.number_input("Trainings last year", 0, 6, 3),
        "YearsAtCompany": c1.number_input("Years at company", 0, 40, 5),
        "YearsInCurrentRole": c2.number_input("Years in current role", 0, 18, 3),
        "YearsSinceLastPromotion": c1.number_input("Years since last promotion", 0, 15, 1),
        "YearsWithCurrManager": c2.number_input("Years with current manager", 0, 17, 3),
    }
    values["EnvironmentSatisfaction"] = st.select_slider(
        "Environment satisfaction", [1, 2, 3, 4], 3, format_func=scale4.get)
    values["JobInvolvement"] = st.select_slider(
        "Job involvement", [1, 2, 3, 4], 3, format_func=scale4.get)
    values["JobSatisfaction"] = st.select_slider(
        "Job satisfaction", [1, 2, 3, 4], 3, format_func=scale4.get)
    values["RelationshipSatisfaction"] = st.select_slider(
        "Relationship satisfaction", [1, 2, 3, 4], 3, format_func=scale4.get)
    values["WorkLifeBalance"] = st.select_slider(
        "Work-life balance", [1, 2, 3, 4], 3, format_func=balance.get)
    values["PerformanceRating"] = st.select_slider("Performance rating", [3, 4], 3)
    return values


def render_form() -> tuple[bool, dict[str, Any]]:
    """Draw the employee form. Returns (was_submitted, answers)."""
    with st.form("employee_form"):
        st.markdown("#### Key factors")
        c1, c2 = st.columns(2)
        values: dict[str, Any] = {
            "OverTime": c1.radio("Works overtime", ["No", "Yes"], horizontal=True),
            "JobRole": c2.selectbox("Job role", OHE_COLUMNS["JobRole"], index=7),
        }
        c1, c2 = st.columns(2)
        values["Age"] = c1.number_input("Age", 18, 60, 35)
        values["TotalWorkingYears"] = c2.number_input("Total working years", 0, 40, 10)

        c1, c2 = st.columns(2)
        values["NumCompaniesWorked"] = c1.number_input("Companies worked at", 0, 9, 2)
        values["JobLevel"] = c2.select_slider("Job level", options=[1, 2, 3, 4, 5], value=2)
        values["StockOptionLevel"] = st.select_slider("Stock option level", options=[0, 1, 2, 3], value=1)

        with st.expander("More employee details (optional)"):
            values.update(render_extra_details())

        submitted = st.form_submit_button("Predict attrition risk")
    return submitted, values


def render_result(prediction: Prediction | None) -> None:
    """Draw the result panel (or a placeholder before the first prediction)."""
    if prediction is None:
        st.markdown(
            '<div class="card placeholder"><h4>No prediction yet</h4>'
            "Fill in the profile and select <b>Predict attrition risk</b>.</div>",
            unsafe_allow_html=True,
        )
        return

    band = prediction.band
    st.markdown(
        f'<div class="card"><div class="verdict" style="--c:{band.color}">'
        f'<div class="label">{band.label}</div><div class="sub">{band.message}</div></div></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_gauge(prediction.probability, band.color), key="gauge")
    st.caption("Share of similar employees in the training data who left.")

    for warning in prediction.warnings:
        st.warning(warning)

    steps_html = "".join(
        f'<div class="step"><div class="n">{i}</div><div class="t">{step}</div></div>'
        for i, step in enumerate(prediction.steps, start=1)
    )
    st.markdown(f'<div class="card"><h4>Why this result</h4>{steps_html}</div>', unsafe_allow_html=True)


def render_predict_tab(model: DecisionTreeClassifier) -> None:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        submitted, values = render_form()

    if submitted:
        st.session_state["prediction"] = predict(model, values)
        st.balloons()  # celebrate every time a prediction is made

    with right:
        render_result(st.session_state.get("prediction"))


def render_model_tab(model: DecisionTreeClassifier) -> None:
    st.markdown("#### What the model pays attention to")
    chart, used = build_importance_chart(model)
    st.plotly_chart(chart, key="importance")
    st.caption(f"Only {used} of the {model.n_features_in_} inputs are used by this tree. "
               "All other fields have no effect on the result.")

    st.markdown("#### Every outcome the tree can give")
    st.dataframe(list_leaf_rules(model), hide_index=True)


def render_about_tab() -> None:
    st.markdown(
        """
#### About this tool
This app wraps a scikit-learn **Decision Tree** trained to predict employee attrition
(the IBM HR Analytics dataset layout). A decision tree is a set of yes/no questions, so every
result can be explained step by step.

#### Reading the score
The score is the share of training employees in the same group who left. Because a tree of this
size has only a few groups, you will see the same few values repeat. The training data was
balanced between leavers and stayers, so the score is best read as a **relative** signal, not as
the true chance that someone resigns.

#### Limits
- Trained on one historical dataset; your organisation may differ.
- Several fields (gender, marital status, age) are sensitive. Do not use them to treat people unfairly.
- Use the result to start a conversation, not to make a decision on its own.
        """
    )


# =============================================================================
# 8. ENTRY POINT
# =============================================================================
def main() -> None:
    st.set_page_config(
        page_title="Employee Attrition Risk", page_icon="🧭",
        layout="wide", initial_sidebar_state="expanded",
    )
    inject_styles()

    try:
        model = load_model()
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH.name}. Keep it in the same folder as app.py.")
        st.stop()
    except Exception as error:  # noqa: BLE001 - show a friendly message instead of a traceback
        st.error(f"The model could not be loaded: {error}")
        st.stop()

    render_sidebar(model)
    render_header()

    tab_predict, tab_model, tab_about = st.tabs(["Predict", "How the model decides", "About"])
    with tab_predict:
        render_predict_tab(model)
    with tab_model:
        render_model_tab(model)
    with tab_about:
        render_about_tab()


main()