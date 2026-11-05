"""Page 6: Single Customer Simulator, Real-time Risk Gauge, Local SHAP, and Batch Scoring."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from app.utils import apply_custom_styles, init_session_state, render_sidebar_status
from src.explain.shap_utils import build_shap_explainer, explain_single_customer
from src.explain.insights import get_retention_playbook
from src.data.loader import load_dataset

st.set_page_config(page_title="RetenSight AI · Predict Churn", page_icon=":material/track_changes:", layout="wide")
apply_custom_styles()
init_session_state()
render_sidebar_status()

st.title("Real-Time Churn Scoring & Customer Simulator")
st.markdown(
    "Score individual accounts with **instant local SHAP attribution**, "
    "or batch-score thousands of customer records with 1-click prioritized CSV export."
)

exp_results = st.session_state.get("experiment_results")
raw_df = st.session_state.get("raw_df")
target_col = st.session_state.get("target_col")
id_col = st.session_state.get("id_col")

if exp_results is None or raw_df is None:
    st.warning("⚠️ No trained models available. Please train models in **Model Training** first.")
    st.stop()

best_name = exp_results["best_model_name"]
preprocessor = exp_results["preprocessor"]
estimator = exp_results["models"][best_name]["estimator"]
feature_names = exp_results["feature_names"]
threshold = float(st.session_state.get("decision_threshold", 0.50))

tab_single, tab_batch = st.tabs(["👤 Single Account Simulator", "📁 Batch Scoring & CSV Export"])

with tab_single:
    st.subheader("1️⃣ Single Account Churn Risk Simulator")
    st.markdown("Adjust customer parameters below to simulate live churn probability and underlying drivers.")

    feature_cols = [c for c in raw_df.columns if c != target_col and c != id_col]
    
    cols_grid = st.columns(3)
    user_inputs = {}

    for idx, col in enumerate(feature_cols):
        grid_col = cols_grid[idx % 3]
        series = raw_df[col]
        
        with grid_col:
            if pd.api.types.is_numeric_dtype(series):
                min_v = float(series.min())
                max_v = float(series.max())
                median_v = float(series.median())
                user_inputs[col] = st.number_input(
                    f"{col}:",
                    min_value=min_v,
                    max_value=max_v,
                    value=median_v,
                    step=1.0 if pd.api.types.is_integer_dtype(series) else 0.5,
                )
            else:
                unique_vals = list(series.dropna().unique())
                user_inputs[col] = st.selectbox(
                    f"{col}:",
                    unique_vals,
                    index=0,
                )

    if st.button("⚡ Score Account & Decompose Risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([user_inputs])
        
        proc_vec = preprocessor.transform(input_df)
        
        if hasattr(estimator, "predict_proba"):
            prob = float(estimator.predict_proba(proc_vec)[0, 1])
        else:
            prob = float(estimator.predict(proc_vec)[0])

        prob_pct = prob * 100
        is_churn = prob >= threshold

        if prob >= 0.60:
            badge_html = '<span class="badge-high">🚨 HIGH ATTRITION RISK</span>'
        elif prob >= 0.30:
            badge_html = '<span class="badge-med">⚠️ MODERATE RISK</span>'
        else:
            badge_html = '<span class="badge-low">✅ LOYAL CUSTOMER</span>'

        st.markdown("---")
        st.subheader("2️⃣ RetenSight Risk Assessment")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Churn Probability", f"{prob_pct:.1f}%")
        with m2:
            st.markdown(f"**Risk Tier:**<br/>{badge_html}", unsafe_allow_html=True)
        with m3:
            status_label = "CHURN PREDICTED" if is_churn else "RETAINED"
            st.metric(f"Classification ($t={threshold:.2f}$)", status_label)

        st.markdown("#### 🔍 Explainability: Why is this customer at risk?")
        explainer = st.session_state.get("explainer")
        if explainer is None:
            explainer = build_shap_explainer(estimator, exp_results["X_train_proc"], model_name=best_name)
            st.session_state["explainer"] = explainer

        local_exp = explain_single_customer(explainer, proc_vec, feature_names)

        c_exp_left, c_exp_right = st.columns([1, 1], gap="large")
        with c_exp_left:
            st.markdown("**Key Risk Drivers (Top Contributors):**")
            for bullet in local_exp["bullet_reasons"]:
                st.markdown(bullet)

        with c_exp_right:
            st.markdown("**Prescriptive Retention Intervention:**")
            top_feats = local_exp["risk_increasing"]["Feature"].tolist()
            actions = get_retention_playbook(top_feats)
            for act in actions:
                st.info(f"💡 **{act['action']}** ({act['target_team']})\n\n{act['detail']}")

with tab_batch:
    st.subheader("📁 Batch Customer Scoring & CSV Export")
    st.markdown("Upload new customer accounts without labels to batch-score the entire cohort in seconds.")

    batch_file = st.file_uploader("Upload CSV or Excel file for scoring", type=["csv", "xlsx"])
    if batch_file is not None:
        batch_df = load_dataset(batch_file)
        st.write(f"Loaded **{len(batch_df):,} accounts** for scoring.")

        if st.button("⚡ Score All Accounts Now", type="primary"):
            with st.spinner("Executing transformations and generating probabilities..."):
                clean_batch = batch_df.copy()
                if id_col and id_col in clean_batch.columns:
                    clean_features = clean_batch.drop(columns=[id_col], errors="ignore")
                else:
                    clean_features = clean_batch

                if target_col in clean_features.columns:
                    clean_features = clean_features.drop(columns=[target_col], errors="ignore")

                batch_proc = preprocessor.transform(clean_features)
                
                if hasattr(estimator, "predict_proba"):
                    batch_probs = estimator.predict_proba(batch_proc)[:, 1]
                else:
                    batch_probs = estimator.predict(batch_proc).astype(float)

                results_df = batch_df.copy()
                results_df["Churn_Probability"] = np.round(batch_probs * 100, 1)
                results_df["Predicted_Churn"] = np.where(batch_probs >= threshold, "Yes", "No")
                results_df["Risk_Band"] = np.where(
                    batch_probs >= 0.60, "High", np.where(batch_probs >= 0.30, "Medium", "Low")
                )

                # Prioritize at-risk accounts first
                results_df = results_df.sort_values(by="Churn_Probability", ascending=False).reset_index(drop=True)

                st.success(f"✓ Scored {len(results_df):,} customer accounts in <0.3s!")
                
                h_count = int((results_df["Risk_Band"] == "High").sum())
                st.markdown(f"**High-Risk Accounts Flagged:** `{h_count:,}` ({h_count / len(results_df) * 100:.1f}%)")
                
                st.dataframe(results_df.head(25), use_container_width=True)

                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Scored Accounts CSV",
                    data=csv_data,
                    file_name="retensight_scored_at_risk_customers.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
