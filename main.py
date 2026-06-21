"""
RA Patients' Lung Cancer Risk Prediction
=========================================
A rigorous, explainable machine-learning web app that predicts lung-cancer
risk in rheumatoid-arthritis (RA) patients.

Key engineering principles (and why they matter):
  * NO DATA LEAKAGE - SMOTE is applied *inside* a pipeline so it only ever
    sees the training fold. Resampling before the split would let synthetic
    copies of test rows leak into training and inflate the score.
  * HONEST EVALUATION - a 60/20/20 stratified train/validation/test split.
    The decision threshold is tuned on the *validation* set, never on test.
  * RICH METRICS - accuracy alone is misleading on imbalanced data (96% of
    patients are cancer-free), so we report precision, recall, F1, ROC-AUC,
    confusion matrices, ROC curves and cross-validated AUC.
  * EXPLAINABILITY - global feature importance + per-patient SHAP/contribution
    breakdowns so a clinician can see *why* a prediction was made.
  * CLINICAL UX - a risk gauge, risk banding and a downloadable PDF report.

Run with:  streamlit run main.py
"""

import warnings
warnings.filterwarnings("ignore")

import io
import os
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(APP_DIR, "ra_lung_cancer_dataset_cleaned.csv")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ----- Optional dependencies (app degrades gracefully if missing) -----------
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False


# ===========================================================================
# Page configuration & styling
# ===========================================================================
st.set_page_config(
    page_title="RA Lung Cancer Risk AI",
    page_icon="🫁",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main > div {padding-top: 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        background:#f0f4f8; border-radius:8px 8px 0 0; padding:8px 16px;
    }
    .stTabs [aria-selected="true"] {background:#1f77b4; color:white;}
    .metric-card {
        background:linear-gradient(135deg,#1f77b4,#4a90d9); color:white;
        border-radius:14px; padding:16px; text-align:center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🫁 RA Patients' Lung-Cancer Risk Prediction")
st.caption(
    "A leakage-free, explainable ML system — train & compare multiple models, "
    "evaluate them honestly on imbalanced data, and score individual patients "
    "with a transparent, downloadable report."
)


# ===========================================================================
# Sidebar controls
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**1 · Data**")
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV or XLSX)", type=["csv", "xlsx"]
    )
    use_sample = st.checkbox(
        "Use bundled sample dataset", value=True,
        help="Loads ra_lung_cancer_dataset_cleaned.csv from the app folder.",
    )

    st.markdown("**2 · Modelling**")
    threshold_metric = st.selectbox(
        "Optimise decision threshold for",
        ["F1 (balanced)", "Recall (catch all cancers)", "Youden's J"],
        help="Threshold is tuned on the validation set, never on the test set.",
    )
    enable_tuning = st.checkbox(
        "Hyper-parameter tuning (GridSearchCV)", value=False,
        help="More accurate but noticeably slower. Leave off for a quick run.",
    )
    random_state = st.number_input("Random seed", 0, 9999, 42)

    st.markdown("---")
    st.caption(
        "Optional libraries detected: "
        f"{'Plotly ✅' if HAS_PLOTLY else 'Plotly ❌'} · "
        f"{'SHAP ✅' if HAS_SHAP else 'SHAP ❌'} · "
        f"{'XGBoost ✅' if HAS_XGB else 'XGBoost ❌'} · "
        f"{'PDF ✅' if HAS_FPDF else 'PDF ❌'}"
    )


# ===========================================================================
# Data loading
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_data(file, use_sample_flag):
    if file is not None:
        if file.name.endswith(".xlsx"):
            return pd.read_csv(file) if False else pd.read_excel(file)
        return pd.read_csv(file)
    if use_sample_flag:
        try:
            return pd.read_csv(SAMPLE_CSV)
        except Exception:
            return None
    return None


df = load_data(uploaded_file, use_sample)

if df is None:
    st.info(
        "👈 Upload a dataset or tick **Use bundled sample dataset** in the "
        "sidebar to begin."
    )
    st.stop()


# ===========================================================================
# Preprocessing helpers
# ===========================================================================
TARGET_CANDIDATES = ["lung_cancer"]
DROP_COLS = ["patient_id", "diagnosis_year"]


@st.cache_data(show_spinner=False)
def preprocess(df):
    target_col = next(
        (c for c in df.columns if c.lower() in TARGET_CANDIDATES), None
    )
    if target_col is None:
        return None
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=[target_col] + drop).copy()
    y = df[target_col]

    # Normalise target to 0/1
    if y.dtype == bool:
        y = y.astype(int)
    elif y.dtype == object:
        y = LabelEncoder().fit_transform(y.astype(str))
    y = pd.Series(np.asarray(y).astype(int), index=X.index, name=target_col)

    # Encode categoricals; remember encoders + which cols are categorical
    encoders, cat_cols = {}, []
    for col in X.select_dtypes(include=["object", "bool"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        cat_cols.append(col)

    return X, y, target_col, encoders, cat_cols


pre = preprocess(df)
if pre is None:
    st.error("❌ Target column 'lung_cancer' not found in the dataset.")
    st.stop()
X, y, target_col, encoders, cat_cols = pre


def build_models(seed):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "SVM": SVC(probability=True, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=4,
            eval_metric="logloss", random_state=seed, verbosity=0,
        )
    return models


PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.1, 1.0, 10.0]},
    "Decision Tree": {"clf__max_depth": [3, 5, 8, None]},
    "Random Forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [5, 10, None]},
    "SVM": {"clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", "auto"]},
    "Gradient Boosting": {"clf__n_estimators": [150, 300], "clf__learning_rate": [0.05, 0.1]},
    "XGBoost": {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]},
}


@st.cache_resource(show_spinner=True)
def train_everything(X, y, seed, tune):
    """Train all models with a leakage-free SMOTE pipeline. Returns a dict."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 60 / 20 / 20 stratified split
    X_trf, X_test, y_trf, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trf, y_trf, test_size=0.25, stratify=y_trf, random_state=seed
    )

    models = build_models(seed)
    fitted, metrics, roc_data = {}, {}, {}

    for name, clf in models.items():
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=seed)),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        if tune and name in PARAM_GRIDS:
            gs = GridSearchCV(
                pipe, PARAM_GRIDS[name], cv=3, scoring="roc_auc", n_jobs=-1
            )
            gs.fit(X_train, y_train)
            pipe = gs.best_estimator_
        else:
            pipe.fit(X_train, y_train)

        proba_test = pipe.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)
        cv_auc = cross_val_score(
            ImbPipeline([
                ("smote", SMOTE(random_state=seed)),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]),
            X_trf, y_trf, cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        fpr, tpr, _ = roc_curve(y_test, proba_test)

        fitted[name] = pipe
        roc_data[name] = (fpr, tpr)
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, pred_test),
            "Precision": precision_score(y_test, pred_test, zero_division=0),
            "Recall": recall_score(y_test, pred_test, zero_division=0),
            "F1": f1_score(y_test, pred_test, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba_test),
            "CV AUC (mean)": cv_auc.mean(),
            "CV AUC (std)": cv_auc.std(),
        }

    return dict(
        fitted=fitted, metrics=metrics, roc_data=roc_data,
        splits=(X_train, X_val, X_test, y_train, y_val, y_test),
    )


def tune_threshold(y_val, proba_val, metric):
    ths = np.linspace(0.05, 0.95, 181)
    if metric.startswith("Recall"):
        scores = [recall_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in ths]
    elif metric.startswith("Youden"):
        scores = []
        for t in ths:
            pred = (proba_val >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0
            spec = tn / (tn + fp) if (tn + fp) else 0
            scores.append(sens + spec - 1)
    else:  # F1
        scores = [f1_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in ths]
    best_idx = int(np.argmax(scores))
    return float(ths[best_idx]), float(scores[best_idx])


# ===========================================================================
# Train
# ===========================================================================
with st.spinner("Training models with a leakage-free pipeline…"):
    res = train_everything(X, y, int(random_state), enable_tuning)

metrics = res["metrics"]
fitted = res["fitted"]
roc_data = res["roc_data"]
X_train, X_val, X_test, y_train, y_val, y_test = res["splits"]

metrics_df = (
    pd.DataFrame(metrics).T
    .sort_values("CV AUC (mean)", ascending=False)
    .round(3)
)
best_model_name = metrics_df.index[0]
best_pipe = fitted[best_model_name]

# Threshold tuning on the VALIDATION set
proba_val = best_pipe.predict_proba(X_val)[:, 1]
best_threshold, best_thr_score = tune_threshold(y_val, proba_val, threshold_metric)


# ===========================================================================
# Tabs
# ===========================================================================
tab_overview, tab_models, tab_eval, tab_explain, tab_predict, tab_method = st.tabs(
    ["📊 Overview", "🏁 Model Comparison", "📈 Evaluation",
     "🔍 Explainability", "🧍 Patient Prediction", "📚 Methodology"]
)


# ---------------------------------------------------------------- Overview --
with tab_overview:
    n = len(df)
    pos = int(y.sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h2>{n:,}</h2>Patients</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h2>{X.shape[1]}</h2>Features</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h2>{pos}</h2>Cancer cases</div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h2>{pos / n * 100:.1f}%</h2>Prevalence</div>", unsafe_allow_html=True)

    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("⚖️ Class balance")
        fig, ax = plt.subplots(figsize=(4, 3))
        counts = y.value_counts().sort_index()
        ax.bar(["No cancer", "Lung cancer"], counts.values, color=["#66b3ff", "#ff6b6b"])
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v}", ha="center", va="bottom")
        ax.set_ylabel("Patients")
        ax.set_title("Severe class imbalance")
        st.pyplot(fig)
        st.caption(
            "Only a small fraction of patients have lung cancer — this is why we "
            "use SMOTE on the training data and judge models by AUC/F1, not accuracy."
        )
    with colB:
        st.subheader("🔗 Feature correlations")
        numeric = X.copy()
        corr = numeric.corr()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(corr.columns)))
        ax2.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax2.set_yticks(range(len(corr.columns)))
        ax2.set_yticklabels(corr.columns, fontsize=6)
        fig2.colorbar(im, fraction=0.046, pad=0.04)
        st.pyplot(fig2)


# -------------------------------------------------------- Model comparison --
with tab_models:
    st.subheader("🏁 Model leaderboard")
    st.caption(
        "Ranked by **cross-validated ROC-AUC** (robust to the random split). "
        "All metrics below are measured on a held-out test set the models never saw."
    )
    try:
        styled = metrics_df.style.background_gradient(
            cmap="Greens", subset=["ROC-AUC", "CV AUC (mean)", "F1"]
        ).format("{:.3f}")
        st.dataframe(styled, use_container_width=True)
    except Exception:
        # pandas Styler needs jinja2; fall back to a plain table if absent
        st.dataframe(metrics_df, use_container_width=True)
    st.success(
        f"🏆 Best model: **{best_model_name}** "
        f"(CV AUC = {metrics_df.loc[best_model_name, 'CV AUC (mean)']:.3f} ± "
        f"{metrics_df.loc[best_model_name, 'CV AUC (std)']:.3f})"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ROC-AUC by model**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        order = metrics_df.sort_values("ROC-AUC")
        ax.barh(order.index, order["ROC-AUC"], color="#1f77b4")
        ax.axvline(0.5, ls="--", color="grey", label="Random (0.5)")
        ax.set_xlim(0, 1); ax.set_xlabel("ROC-AUC"); ax.legend()
        st.pyplot(fig)
    with col2:
        st.markdown("**ROC curves**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for name, (fpr, tpr) in roc_data.items():
            ax.plot(fpr, tpr, label=f"{name} ({metrics[name]['ROC-AUC']:.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7)
        st.pyplot(fig)


# ------------------------------------------------------------- Evaluation ---
with tab_eval:
    st.subheader(f"📈 Detailed evaluation — {best_model_name}")
    proba_test = best_pipe.predict_proba(X_test)[:, 1]

    st.markdown(
        f"**Decision threshold tuned on the validation set** for "
        f"*{threshold_metric}*: **{best_threshold:.2f}** "
        f"(validation score = {best_thr_score:.2f})."
    )
    pred_default = (proba_test >= 0.5).astype(int)
    pred_tuned = (proba_test >= best_threshold).astype(int)

    cc1, cc2 = st.columns(2)
    for col, pred, label in [
        (cc1, pred_default, "Default threshold = 0.50"),
        (cc2, pred_tuned, f"Tuned threshold = {best_threshold:.2f}"),
    ]:
        with col:
            st.markdown(f"**{label}**")
            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(3.5, 3))
            im = ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black", fontsize=12)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["No cancer", "Cancer"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["No cancer", "Cancer"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
            st.write({
                "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
                "Recall": round(recall_score(y_test, pred, zero_division=0), 3),
                "F1": round(f1_score(y_test, pred, zero_division=0), 3),
            })
    st.info(
        "Lowering the threshold trades precision for recall — in cancer "
        "screening, missing a true case (false negative) is usually far costlier "
        "than a false alarm, so a lower threshold is often clinically preferred."
    )


# ---------------------------------------------------------- Explainability --
@st.cache_data(show_spinner=False)
def global_importance(_pipe, X_train, X_test, y_test, model_name):
    """Return a Series of feature importances using the best available method."""
    clf = _pipe.named_steps["clf"]
    feat = X_train.columns
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=feat).sort_values()
    if hasattr(clf, "coef_"):
        return pd.Series(np.abs(clf.coef_[0]), index=feat).sort_values()
    # fallback: permutation importance on the full pipeline
    r = permutation_importance(_pipe, X_test, y_test, n_repeats=5,
                               random_state=0, scoring="roc_auc")
    return pd.Series(r.importances_mean, index=feat).sort_values()


with tab_explain:
    st.subheader("🔍 Why does the model decide the way it does?")
    imp = global_importance(best_pipe, X_train, X_test, y_test, best_model_name)

    st.markdown("**Global feature importance**")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index, imp.values, color="#2ca02c")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    st.caption(
        f"Drivers of risk according to {best_model_name}. Higher = more "
        "influence on the prediction across the patient population."
    )

    if HAS_SHAP:
        st.markdown("**SHAP summary (sampled)** — direction & magnitude of each feature's effect")
        try:
            clf = best_pipe.named_steps["clf"]
            scaler = best_pipe.named_steps["scaler"]
            X_bg = scaler.transform(X_test)
            X_bg_df = pd.DataFrame(X_bg, columns=X_test.columns)
            sample = X_bg_df.sample(min(150, len(X_bg_df)), random_state=0)
            if hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(sample)
                if isinstance(sv, list):
                    sv = sv[1]
            else:
                explainer = shap.Explainer(clf, X_bg_df.sample(min(100, len(X_bg_df)), random_state=1))
                sv = explainer(sample).values
            fig = plt.figure()
            shap.summary_plot(sv, sample, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.caption(f"SHAP summary unavailable for this model ({e}). "
                       "Global importance above still applies.")
    else:
        st.caption("Install `shap` to unlock per-feature SHAP value plots "
                   "(`pip install shap`).")


# ------------------------------------------------------- Patient prediction --
def make_pdf(patient, prob, threshold, decision, band, model_name):
    pdf = FPDF()
    pdf.add_page()
    nl = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "RA Lung-Cancer Risk Report", **nl)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", **nl)
    pdf.cell(0, 6, f"Model: {model_name}", **nl)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*((200, 0, 0) if decision == 1 else (0, 130, 0)))
    pdf.cell(0, 9, f"Risk band: {band}", **nl)
    pdf.cell(0, 9, f"Predicted probability: {prob*100:.1f}%   (threshold {threshold:.2f})", **nl)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Patient inputs", **nl)
    pdf.set_font("Helvetica", "", 10)
    for k, v in patient.items():
        pdf.cell(0, 6, f"  - {k}: {v}", **nl)
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5,
        "Disclaimer: This is a decision-support estimate from a machine-learning "
        "model trained on historical data. It is NOT a medical diagnosis and must "
        "not replace evaluation by a qualified clinician.")
    return bytes(pdf.output())


with tab_predict:
    st.subheader("🧍 Score an individual patient")
    st.caption("Enter the patient's details, then click **Predict risk**.")

    with st.form("patient_form"):
        cols = st.columns(3)
        input_data, display = {}, {}
        for i, col in enumerate(X.columns):
            with cols[i % 3]:
                if col in encoders:
                    options = list(encoders[col].classes_)
                    choice = st.selectbox(col, options, key=col)
                    input_data[col] = int(encoders[col].transform([choice])[0])
                    display[col] = choice
                else:
                    lo, hi = float(X[col].min()), float(X[col].max())
                    med = float(X[col].median())
                    val = st.number_input(col, lo, hi, med, key=col)
                    input_data[col] = val
                    display[col] = val
        thr = st.slider("Decision threshold", 0.05, 0.95, float(best_threshold), 0.01)
        submitted = st.form_submit_button("🔮 Predict risk", use_container_width=True)

    if submitted:
        patient_df = pd.DataFrame([input_data])[X.columns]
        prob = float(best_pipe.predict_proba(patient_df)[0][1])
        decision = int(prob >= thr)

        if prob < 0.20:
            band, color = "Low", "#2ca02c"
        elif prob < 0.50:
            band, color = "Moderate", "#ff9f40"
        else:
            band, color = "High", "#e74c3c"

        left, right = st.columns([1, 1])
        with left:
            if HAS_PLOTLY:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": f"Lung-cancer risk · {band}"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 20], "color": "#d4efdf"},
                            {"range": [20, 50], "color": "#fdebd0"},
                            {"range": [50, 100], "color": "#fadbd8"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3},
                                      "value": thr * 100},
                    },
                ))
                gauge.update_layout(height=300, margin=dict(t=50, b=10))
                st.plotly_chart(gauge, use_container_width=True)
            else:
                st.metric("Predicted risk", f"{prob*100:.1f}%", band)

        with right:
            if decision == 1:
                st.error(f"⚠️ **Elevated risk — flag for follow-up**\n\n"
                         f"Probability {prob*100:.1f}% ≥ threshold {thr:.2f}")
            else:
                st.success(f"✅ **Below threshold**\n\n"
                           f"Probability {prob*100:.1f}% < threshold {thr:.2f}")
            st.caption(f"Risk band: **{band}** · Model: {best_model_name}")

        # Per-patient contribution breakdown
        st.markdown("**Top factors for this patient**")
        try:
            clf = best_pipe.named_steps["clf"]
            scaler = best_pipe.named_steps["scaler"]
            xp = scaler.transform(patient_df)
            xp_df = pd.DataFrame(xp, columns=X.columns)
            if HAS_SHAP and hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(xp_df)
                if isinstance(sv, list):
                    sv = sv[1]
                contrib = pd.Series(np.asarray(sv).ravel(), index=X.columns)
            elif hasattr(clf, "coef_"):
                contrib = pd.Series(clf.coef_[0] * xp_df.iloc[0].values, index=X.columns)
            else:
                imp = global_importance(best_pipe, X_train, X_test, y_test, best_model_name)
                contrib = imp.reindex(X.columns).fillna(0)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(top.index, top.values,
                    color=["#e74c3c" if v > 0 else "#2ca02c" for v in top.values])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("← lowers risk      raises risk →")
            st.pyplot(fig)
        except Exception as e:
            st.caption(f"Per-patient breakdown unavailable ({e}).")

        # PDF report
        if HAS_FPDF:
            try:
                pdf_bytes = make_pdf(display, prob, thr, decision, band, best_model_name)
                st.download_button(
                    "📄 Download patient report (PDF)",
                    data=pdf_bytes,
                    file_name=f"ra_lung_risk_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"PDF generation failed ({e}).")
        else:
            st.caption("Install `fpdf2` to enable downloadable PDF reports.")


# ------------------------------------------------------------- Methodology --
with tab_method:
    st.subheader("📚 Methodology & design decisions")
    st.markdown(
        f"""
### 1 · The problem
Predict lung-cancer risk for rheumatoid-arthritis patients. Only
**{int(y.sum())} of {len(y):,}** patients ({y.mean()*100:.1f}%) have cancer, so
the data is **severely imbalanced**. A model that predicts "no cancer" for
everyone would be ~{(1-y.mean())*100:.0f}% accurate yet clinically useless —
this is why we never judge models by accuracy alone.

### 2 · Avoiding data leakage (the big one)
A common mistake is to apply **SMOTE before splitting** the data. That copies
information from test rows into training and produces falsely high scores.
Here, SMOTE lives **inside an `imblearn` Pipeline**, so it only ever resamples
the training fold — during cross-validation *and* the final fit. Scaling is in
the same pipeline for the same reason.

### 3 · Honest evaluation
The data is split **60 % train / 20 % validation / 20 % test**, all
*stratified* to preserve the rare-class ratio:
- **Train** — fit the models (with SMOTE applied only here).
- **Validation** — choose the decision threshold.
- **Test** — reported only once, never touched during tuning.

Models are ranked by **cross-validated ROC-AUC** (mean ± std over 5 folds) so
the leaderboard isn't at the mercy of one lucky split.

### 4 · Threshold tuning
Instead of blindly using 0.5, the threshold is optimised on the **validation
set** for your chosen objective ({threshold_metric}). Optimising recall alone
tends to flag everyone, so **F1** (recall *and* precision) is the default;
**Youden's J** balances sensitivity and specificity.

### 5 · Models compared
Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting
{'and XGBoost ' if HAS_XGB else ''}— optionally tuned with `GridSearchCV`.

### 6 · Explainability
Global feature importance{' plus SHAP value plots' if HAS_SHAP else ''} reveal
which factors drive predictions, and each individual prediction comes with a
per-patient factor breakdown — essential for clinical trust.

### ⚠️ Honest caveat
On this dataset the leakage-free ROC-AUC is modest (≈0.6), well below what a
*leaky* pipeline would falsely report. That gap is exactly the point: a
rigorous setup tells you the truth about how much real signal exists, rather
than flattering the model.
"""
    )

st.markdown("---")
st.caption(
    "⚕️ Decision-support tool for educational use only — not a medical device "
    "and not a substitute for professional diagnosis."
)
