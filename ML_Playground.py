import io
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Optional imports
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------- Page setup ----------
st.set_page_config(page_title="ML Expert Playground", page_icon="🧠", layout="wide")

st.title("🧠 ML Expert Playground — Streamlit")
st.write(
    "Upload tabular data, do preprocessing & feature engineering, handle imbalance, run 5-fold CV, inspect interactive plots, and download a trained pipeline."
)

st.sidebar.header("Quick install")
st.sidebar.code("pip install streamlit scikit-learn pandas numpy plotly imbalanced-learn shap xgboost")

# ---------- Helpers ----------

def load_demo(name: str) -> pd.DataFrame:
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes, fetch_california_housing

    if name == "Iris (classification)":
        data = load_iris(as_frame=True)
    elif name == "Wine (classification)":
        data = load_wine(as_frame=True)
    elif name == "Breast Cancer (classification)":
        data = load_breast_cancer(as_frame=True)
    elif name == "Diabetes (regression)":
        data = load_diabetes(as_frame=True)
    elif name == "California Housing (regression)":
        data = fetch_california_housing(as_frame=True)
    else:
        raise ValueError("Unknown demo")
    df = data.frame.copy()
    if "target" not in df.columns:
        df["target"] = data.target
    return df


def infer_task(y: pd.Series) -> str:
    nunique = y.nunique(dropna=True)
    if y.dtype.kind in "ifu":
        if nunique <= max(2, int(0.05 * len(y))):
            return "Classification"
        return "Regression"
    else:
        return "Classification"


def build_preprocessor(X: pd.DataFrame, apply_poly: bool) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if apply_poly and len(num_cols) > 0:
        numeric_steps.insert(1, ("poly", PolynomialFeatures(degree=2, include_bias=False)))

    from sklearn.pipeline import Pipeline as SkPipeline

    numeric = SkPipeline(numeric_steps)
    categorical = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)])
    return pre


def get_models(task: str) -> Dict[str, object]:
    if task == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_jobs=-1),
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "RandomForestRegressor": RandomForestRegressor(n_jobs=-1),
        }


def scoring_for_task(task: str):
    if task == "Classification":
        return {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "roc_auc": "roc_auc",
        }
    else:
        return {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}

# ---------- UI: Data ----------

st.header("1) Data")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload CSV", type=["csv"] )
    demo = st.selectbox("or pick a demo", ["—", "Iris (classification)", "Wine (classification)", "Breast Cancer (classification)", "Diabetes (regression)", "California Housing (regression)"], index=0)
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source = "uploaded"
    elif demo != "—":
        df = load_demo(demo)
        source = f"demo: {demo}"
    else:
        st.info("Upload data or choose a demo to continue")
        st.stop()

with col2:
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", f"{df.shape[1]:,}")
    st.caption(f"Source: {source}")

st.write(df.head(100))

# ---------- UI: Target & task ----------

st.header("2) Target & Task")
cols = st.columns(3)
with cols[0]:
    target_col = st.selectbox("Select target", df.columns, index=len(df.columns)-1)
with cols[1]:
    task_auto = infer_task(df[target_col])
    task_choice = st.selectbox("Task", ["Auto", "Classification", "Regression"], index=0)
with cols[2]:
    test_size = st.slider("Test size (%)", 5, 40, 20)

if task_choice == "Auto":
    task = task_auto
else:
    task = task_choice

st.caption(f"Inferred task: **{task_auto}** — using: **{task}**")

X = df.drop(columns=[target_col])
y = df[target_col]
if task == "Classification" and y.dtype.kind not in "iu":
    y = y.astype("category").cat.codes

# ---------- UI: Preprocessing & FE ----------

st.header("3) Preprocessing & Feature Engineering")
col_a, col_b = st.columns(2)
with col_a:
    apply_poly = st.checkbox("Add polynomial interaction features (degree=2)", value=False)
    do_scaling = st.checkbox("Scale numeric features", value=True)
    handle_missing = st.selectbox("Missing value strategy (numeric)", ["median", "mean", "drop"], index=0)
with col_b:
    imbalance_method = st.selectbox("Handle class imbalance", ["None", "SMOTE (oversample)", "Random Undersample"], index=0)
    feature_selection = st.selectbox("Feature selection", ["None", "SelectKBest", "RFE (wrapper)"], index=0)
    k_select = st.slider("K (for SelectKBest)", 1, max(1, X.shape[1]), min(10, max(1, X.shape[1])))

pre = build_preprocessor(X, apply_poly=apply_poly)

# ---------- UI: Model & CV ----------

st.header("4) Model & 5-fold Cross-validation")
models = get_models(task)
model_name = st.selectbox("Model", list(models.keys()))
model = models[model_name]

cv = 5
random_state = st.number_input("random_state", value=42, step=1)

# ---------- Prepare pipeline building ----------

steps = [("pre", pre)]

# imbalance step handled after preprocessing (on numeric arrays) — implement in training loop

if feature_selection == "SelectKBest":
    score_func = f_classif if task == "Classification" else f_regression
    steps.append(("select", SelectKBest(score_func=score_func, k=k_select)))
elif feature_selection == "RFE":
    # RFE requires estimator; use model
    steps.append(("select", RFE(estimator=model, n_features_to_select=min(k_select, X.shape[1]))))

steps.append(("model", model))
pipe = Pipeline(steps)

# ---------- Train / CV ----------

if st.button("Run 5-fold CV & Train"):
    with st.spinner("Running cross-validation and training…"):
        # Prepare folds
        if task == "Classification":
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(random_state))
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=int(random_state))

        scoring = scoring_for_task(task)

        # We will perform cross_validate on the pipeline but need to handle imbalance with a wrapper if selected
        # Simple approach: for imbalance methods, perform manual CV loop to apply SMOTE/undersample on train fold

        if imbalance_method != "None" and IMBLEARN_AVAILABLE:
            scores = {name: [] for name in scoring.keys()}
            fold = 0
            for train_idx, test_idx in splitter.split(X, y):
                fold += 1
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                # Fit preprocessor on train
                pre_fit = pre.fit(X_tr)
                X_tr_trans = pre_fit.transform(X_tr)
                X_te_trans = pre_fit.transform(X_te)

                # handle imbalance
                if imbalance_method == "SMOTE":
                    sm = SMOTE(random_state=int(random_state))
                    X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_trans, y_tr)
                elif imbalance_method == "Random Undersample":
                    ru = RandomUnderSampler(random_state=int(random_state))
                    X_tr_bal, y_tr_bal = ru.fit_resample(X_tr_trans, y_tr)
                else:
                    X_tr_bal, y_tr_bal = X_tr_trans, y_tr

                # If feature selection is RFE, need to fit selector on balanced X_tr_bal
                model_clone = pickle.loads(pickle.dumps(model))
                if feature_selection == "RFE":
                    sel = RFE(estimator=model_clone, n_features_to_select=min(k_select, X_tr_bal.shape[1]))
                    X_tr_bal = sel.fit_transform(X_tr_bal, y_tr_bal)
                    X_te_trans = sel.transform(X_te_trans)

                # fit model
                model_clone.fit(X_tr_bal, y_tr_bal)
                y_pred = model_clone.predict(X_te_trans)

                # compute metrics
                if task == "Classification":
                    scores["accuracy"].append(accuracy_score(y_te, y_pred))
                    scores["f1_weighted"].append(f1_score(y_te, y_pred, average="weighted"))
                    try:
                        if hasattr(model_clone, "predict_proba"):
                            y_proba = model_clone.predict_proba(X_te_trans)[:, 1] if len(np.unique(y)) == 2 else None
                            if y_proba is not None:
                                scores["roc_auc"].append(roc_auc_score(y_te, y_proba))
                            else:
                                scores["roc_auc"].append(np.nan)
                        else:
                            scores["roc_auc"].append(np.nan)
                    except Exception:
                        scores["roc_auc"].append(np.nan)
                else:
                    # regression metrics
                    mae = mean_absolute_error(y_te, y_pred)
                    rmse = mean_squared_error(y_te, y_pred, squared=False)
                    r2 = r2_score(y_te, y_pred)
                    scores["MAE"].append(mae)
                    scores["RMSE"].append(rmse)
                    scores["R2"].append(r2)

            # show results
            st.header("Cross-validation results")
            cv_summary = {k: (np.nanmean(v), np.nanstd(v)) for k, v in scores.items()}
            cv_df = pd.DataFrame({"metric": list(cv_summary.keys()), "mean": [cv_summary[k][0] for k in cv_summary], "std": [cv_summary[k][1] for k in cv_summary]})
            st.table(cv_df)

        else:
            # use sklearn cross_validate directly on pipeline
            cv_results = cross_validate(pipe, X, y, cv=splitter, scoring=list(scoring.values()), return_train_score=False)
            st.header("Cross-validation results")
            cv_df = pd.DataFrame({k: v for k, v in cv_results.items() if "test_" in k})
            # rename columns
            cv_df.columns = [c.replace("test_", "") for c in cv_df.columns]
            st.write(cv_df.describe().loc[["mean", "std"]].T)

        st.success("Cross-validation finished")

    # Fit final pipeline on full data (optionally with imbalance handling)
    with st.spinner("Fitting final pipeline on full dataset…"):
        # Fit preprocessor
        pre_fit = pre.fit(X)
        X_full = pre_fit.transform(X)
        y_full = y

        if imbalance_method != "None" and IMBLEARN_AVAILABLE:
            if imbalance_method == "SMOTE":
                X_full, y_full = SMOTE(random_state=int(random_state)).fit_resample(X_full, y_full)
            elif imbalance_method == "Random Undersample":
                X_full, y_full = RandomUnderSampler(random_state=int(random_state)).fit_resample(X_full, y_full)

        # If RFE selected, fit selector
        if feature_selection == "RFE":
            model_clone = pickle.loads(pickle.dumps(model))
            sel = RFE(estimator=model_clone, n_features_to_select=min(k_select, X_full.shape[1]))
            X_full = sel.fit_transform(X_full, y_full)
            final_model = model_clone
            final_model.fit(X_full, y_full)

            # Create final pipeline with pre + sel + model to save
            final_pipe = (pre_fit, sel, final_model)
        else:
            final_model = pickle.loads(pickle.dumps(model))
            final_model.fit(X_full, y_full)
            final_pipe = Pipeline([("pre", pre_fit), ("model", final_model)])

        st.success("Final model trained on full data")

        # Save pipeline
        buf = io.BytesIO()
        try:
            pickle.dump(final_pipe, buf)
            st.download_button("Download trained pipeline (.pkl)", data=buf.getvalue(), file_name="expert_pipeline.pkl")
        except Exception as e:
            st.warning(f"Could not pickle pipeline: {e}")

        # ---------- Interactive plots ----------
        st.header("Interactive data behaviour plots")
        # Target distribution
        fig = px.histogram(df, x=target_col, nbins=50, title="Target distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap (numeric only)
        num_df = df.select_dtypes(include=["number"]).corr()
        fig2 = px.imshow(num_df, title="Numeric feature correlation (Pearson)")
        st.plotly_chart(fig2, use_container_width=True)

        # PCA scatter of first 2 components
        try:
            pca = PCA(n_components=2)
            X_prep = pre_fit.transform(X if feature_selection != "RFE" else X)
            X_pca = pca.fit_transform(X_prep)
            pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "target": y_full})
            fig3 = px.scatter(pca_df, x="PC1", y="PC2", color="target", title="PCA (2 components)")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.info(f"PCA plot skipped: {e}")

        # Feature importance for tree models or coefficients for linear
        try:
            if hasattr(final_model, "feature_importances_"):
                # get feature names from preprocessor
                try:
                    # attempt to get feature names (simple approach)
                    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
                    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                    feat_names = num_cols + cat_cols
                except Exception:
                    feat_names = [f"f{i}" for i in range(len(final_model.feature_importances_))]
                fi = pd.DataFrame({"feature": feat_names[: len(final_model.feature_importances_)], "importance": final_model.feature_importances_})
                fi = fi.sort_values("importance", ascending=False).head(30)
                fig4 = px.bar(fi, x="importance", y="feature", orientation="h", title="Feature Importances")
                st.plotly_chart(fig4, use_container_width=True)
            elif hasattr(final_model, "coef_"):
                coef = final_model.coef_
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                feat_names = X.select_dtypes(include=["number"]).columns.tolist()
                fi = pd.DataFrame({"feature": feat_names[: len(coef)], "coef": coef})
                fi = fi.sort_values("coef", ascending=False).head(30)
                fig5 = px.bar(fi, x="coef", y="feature", orientation="h", title="Top coefficients")
                st.plotly_chart(fig5, use_container_width=True)
        except Exception as e:
            st.info(f"Feature importance not available: {e}")

        # SHAP explanations (if available)
        if SHAP_AVAILABLE:
            st.subheader("SHAP explainability (sample)")
            try:
                explainer = shap.Explainer(final_model, X_full)
                shap_values = explainer(X_full[:100])
                st.pyplot(shap.summary_plot(shap_values, feature_names=None, show=False))
            except Exception as e:
                st.info(f"SHAP failed: {e}")

st.caption("This expert playground shows best-practice steps: preprocessing, optional polynomial features, handling imbalance (SMOTE/undersampling), feature selection (SelectKBest/RFE), 5-fold CV, and interactive plots. For heavy training or GPU models, train offline and use this app for inference/demos.")
