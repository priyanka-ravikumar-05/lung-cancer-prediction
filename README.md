# 🫁 RA Patients' Lung-Cancer Risk Prediction

An interactive, **leakage-free and explainable** machine-learning web app that
estimates lung-cancer risk in rheumatoid-arthritis (RA) patients, compares
several models honestly, and scores individual patients with a transparent,
downloadable report.

Built with Streamlit + scikit-learn + imbalanced-learn.

---

## ✨ What it does

- **Upload** any compatible CSV/XLSX (a bundled sample is included).
- **Trains and compares** Logistic Regression, Decision Tree, Random Forest,
  SVM, Gradient Boosting (and XGBoost if installed), with optional
  `GridSearchCV` hyper-parameter tuning.
- **Evaluates rigorously** with precision, recall, F1, ROC-AUC, cross-validated
  AUC, confusion matrices and ROC curves — not just accuracy.
- **Explains predictions** with global feature importance and per-patient
  contribution breakdowns (SHAP when available).
- **Scores a single patient** with a risk gauge, risk band, adjustable decision
  threshold and a one-click **PDF report**.

## 🔬 Why it's methodologically sound

This project deliberately fixes the most common mistakes in student ML projects:

1. **No data leakage.** SMOTE (which balances the rare cancer class) runs
   *inside* an `imblearn` Pipeline, so it only ever sees the training fold —
   during cross-validation and the final fit. Resampling before the split would
   copy information from test rows into training and produce *falsely high*
   scores.
2. **Honest split.** Data is divided **60% train / 20% validation / 20% test**,
   all *stratified* to preserve the ~3.7% cancer prevalence. The decision
   threshold is tuned on the **validation** set; the **test** set is reported
   only once.
3. **Right metrics.** With such an imbalanced target, a model that predicts "no
   cancer" for everyone is ~96% accurate yet useless — so models are ranked by
   **cross-validated ROC-AUC** and judged on F1/recall/precision.
4. **Sensible threshold.** Instead of a blind 0.5 cut-off, the threshold is
   optimised for F1 (default), recall, or Youden's J.

> ⚠️ **Honest result:** on this dataset the leakage-free ROC-AUC is modest
> (~0.6) — well below what a *leaky* pipeline would falsely report. That gap is
> the whole point: a rigorous setup tells you how much real signal exists rather
> than flattering the model.

## 🚀 Run it

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. launch
streamlit run main.py
```

The app opens in your browser. Keep **"Use bundled sample dataset"** ticked to
load `ra_lung_cancer_dataset_cleaned.csv`, or upload your own file with a
`lung_cancer` column.

## 📁 Files

| File | Purpose |
|------|---------|
| `main.py` | The Streamlit application |
| `ra_lung_cancer_dataset_cleaned.csv` | Sample dataset (2,500 patients) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## ⚕️ Disclaimer

This is a decision-support and educational tool only. It is **not** a medical
device and must not replace evaluation by a qualified clinician.
