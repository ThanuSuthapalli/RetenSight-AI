"""Page 5: Model Explainability, Global SHAP Analysis, and Feature Ranking."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app.utils import apply_custom_styles, init_session_state, render_sidebar_status
from src.explain.shap_utils import build_shap_explainer, compute_global_shap
from src.viz.plots import plot_feature_importance_bar

st.set_page_config(page_title="RetenSight AI · Explainability", page_icon=":material/visibility:", layout="wide")
apply_custom_styles()
init_session_state()
render_sidebar_status()

st.title("Model Explainability & Global SHAP Insights")
st.markdown(
    "Deconstruct black-box predictions using **Shapley Additive exPlanations (SHAP)**. "
    "Identify which customer attributes exert the strongest pull toward churn across your entire customer base."
)

exp_results = st.session_state.get("experiment_results")

if exp_results is None:
    st.warning("⚠️ No trained models available. Please train models in **Model Training** first.")
    st.stop()

best_name = exp_results["best_model_name"]
model_names = list(exp_results["models"].keys())
selected_model_name = st.selectbox("Select Model to Explain:", model_names, index=model_names.index(best_name))

estimator = exp_results["models"][selected_model_name]["estimator"]
feature_names = exp_results["feature_names"]
X_train_proc = exp_results["X_train_proc"]
X_test_proc = exp_results["X_test_proc"]

# Cache SHAP calculations in session state for instant retrieval (<0.01s)
cached_shap = st.session_state.get("global_shap_data")
if cached_shap is None or st.session_state.get("shap_model_name") != selected_model_name:
    with st.spinner("Calculating SHAP values (<1s lightning mode)..."):
        explainer = build_shap_explainer(estimator, X_train_proc, model_name=selected_model_name)
        cached_shap = compute_global_shap(explainer, X_test_proc, feature_names, max_samples=150)
        st.session_state["global_shap_data"] = cached_shap
        st.session_state["explainer"] = explainer
        st.session_state["shap_model_name"] = selected_model_name

col_bar, col_table = st.columns([3, 2], gap="large")

with col_bar:
    st.subheader("1️⃣ Global Churn Drivers (Mean |SHAP|)")
    fig_bar = plot_feature_importance_bar(cached_shap["importance_df"], top_n=12)
    st.pyplot(fig_bar)
    st.caption("Attributes at the top exert the highest directional impact on customer churn risk.")

with col_table:
    st.subheader("2️⃣ Driver Impact Ranking")
    st.dataframe(
        cached_shap["importance_df"].head(15).rename(columns={
            "Feature": "Customer Attribute",
            "Mean_Abs_SHAP": "Mean |SHAP| Impact",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
st.subheader("3️⃣ Interpreting SHAP Values in Retention Strategy")
st.markdown(
    """
    <div class="retensight-card">
        <strong>💡 Understanding SHAP Impact:</strong>
        <p style="margin-top: 0.5rem; color: #475569; font-size: 0.92rem; line-height: 1.6;">
            SHAP values quantify the exact marginal contribution of each attribute toward churn relative to the baseline.<br/>
            • <strong style="color: #DC2626;">Positive (+) SHAP:</strong> Increases likelihood of churn (e.g. Month-to-month contract, payment friction).<br/>
            • <strong style="color: #059669;">Negative (-) SHAP:</strong> Buffers retention and creates platform stickiness (e.g. 2-year contract, multi-year tenure).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
