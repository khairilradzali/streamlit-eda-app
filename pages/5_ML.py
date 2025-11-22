# pages/5_ML.py
import streamlit as st
import pandas as pd
import numpy as np
from utils import render_footer
from utils import get_df, train_test_split_xy, to_csv_bytes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io

st.set_page_config(page_title="ML Modeling", layout="wide")
st.title("🤖 Automatic ML Modeling")

df = get_df()
if df is None:
    st.info("Upload a dataset first (Home page).")
else:
    st.subheader("Select target column (what you want to predict)")
    target = st.selectbox("Target column", ["--select--"] + list(df.columns))
    if target and target != "--select--":
        st.write("Target sample values:")
        st.write(df[target].value_counts().head(10))

        problem_type = st.selectbox("Override problem type (auto-detected)", ["Auto", "Classification", "Regression"])
        # auto-detect
        if problem_type == "Auto":
            unique = df[target].nunique(dropna=True)
            if df[target].dtype.kind in 'O' or unique <= 10:
                task = "classification"
            else:
                task = "regression"
        else:
            task = problem_type.lower()

        st.info(f"Detected task: **{task}**")

        # simple train/test split on numeric features only (user should use cleaning page to prepare)
        test_size = st.slider("Test size (fraction)", 0.05, 0.5, 0.2)
        scale = st.checkbox("Scale numeric features (StandardScaler)", value=True)

        X_train, X_test, y_train, y_test = train_test_split_xy(df, target, test_size=test_size, scale=scale)

        st.write("Features used (numeric only):", X_train.columns.tolist())

        if st.button("Train models"):
            if task == "classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=500),
                    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42)
                }

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    if task == "classification":
                        acc = accuracy_score(y_test, preds)
                        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                        # roc only for binary
                        roc = None
                        if len(np.unique(y_test.dropna())) == 2:
                            try:
                                prob = model.predict_proba(X_test)[:,1]
                                roc = roc_auc_score(y_test, prob)
                            except:
                                roc = None
                        results[name] = {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "roc_auc":roc}
                    else:
                        mse = mean_squared_error(y_test, preds)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, preds)
                        results[name] = {"rmse":rmse, "r2":r2}
                    # save model to session for download / reuse
                    st.session_state[f"model_{name}"] = model
                except Exception as e:
                    st.error(f"Training failed for {name}")
                    st.exception(e)

            st.subheader("Model results")
            st.write(results)

            # show feature importances for tree-based
            for name, model in models.items():
                if hasattr(model, "feature_importances_"):
                    st.write(f"Feature importances — {name}")
                    importances = model.feature_importances_
                    fi = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
                    st.dataframe(fi.head(20))

            # Predict with best model (naively pick RandomForest if available)
            preferred = "RandomForest" if task=="classification" else "RandomForestRegressor"
            if preferred in st.session_state:
                chosen = st.session_state[ "model_" + (preferred if task=="regression" else preferred) ]
            else:
                # fallback to first model
                chosen = st.session_state[list(st.session_state.keys())[-1]]

            preds = chosen.predict(X_test)
            out_df = X_test.copy()
            out_df[f"pred_{target}"] = preds
            st.subheader("Sample predictions (test set)")
            st.dataframe(out_df.head())

            # download preds
            st.download_button("Download predictions CSV", data=to_csv_bytes(out_df.reset_index()), file_name="predictions.csv", mime="text/csv")

            # allow download of model
            buf = io.BytesIO()
            joblib.dump(chosen, buf)
            buf.seek(0)
            st.download_button("Download trained model (joblib)", data=buf, file_name="model.joblib")

# Footer
render_footer()