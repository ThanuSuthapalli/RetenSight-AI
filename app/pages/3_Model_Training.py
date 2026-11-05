"""Page 3: Model Training Control Center, Tuning, and Experiment Tracker."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from app.utils import apply_custom_styles, init_session_state, render_sidebar_status
from src.models.train import train_models_pipeline
from src.models.registry import save_experiment_run

st.set_page_config(page_title="RetenSight AI · Model Training", page_icon=":material/psychology:", layout="wide")
apply_custom_styles()
init_session_state()
render_sidebar_status()

st.title("Automated Model Tournament & Validation Lab")
st.markdown(
    "Train, tune, and evaluate classification models using **Stratified Cross-Validation**, "
    "automated class imbalance weighting, and leak-free pipelines in seconds."
)

X = st.session_state.get("cleaned_X")
y = st.session_state.get("cleaned_y")

if X is None or y is None:
    st.warning("⚠️ No dataset validated. Please load or upload data on the **Home** page first.")
    st.stop()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1️⃣ Model Tournament Selection")
    selected_models = []
    
    c_lr = st.checkbox("Logistic Regression (Ultra-Fast Linear Baseline)", value=True)
    c_rf = st.checkbox("Random Forest Classifier (Ensemble Bagging Trees)", value=True)
    c_xgb = st.checkbox("XGBoost Classifier (Regularized Gradient Boosting)", value=True)

    if c_lr:
        selected_models.append("Logistic Regression")
    if c_rf:
        selected_models.append("Random Forest")
    if c_xgb:
        selected_models.append("XGBoost")

    if not selected_models:
        st.error("Select at least one model to train.")

    st.subheader("2️⃣ Performance & Validation Strategy")
    t_mode = st.radio(
        "Tuning Speed Mode:",
        ["⚡ Quick Mode (Lightning Fast: ~10-15s total)", "🔬 Thorough Mode (Expanded search: ~1-2 min)"],
        index=0,
        help="Quick Mode uses 3-fold Stratified CV for instant results; Thorough Mode uses 5-fold CV.",
    )
    quick_mode = "Quick" in t_mode

    imbalance_mode = st.selectbox(
        "Class Imbalance Handling:",
        [
            ("balanced_weights", "Balanced Class Weights (Recommended: penalizes missed churners)"),
            ("none", "None (Uniform Class Weights)"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    metric_choice = st.selectbox(
        "Cross-Validation Objective:",
        ["roc_auc", "f1", "recall", "accuracy"],
        index=0,
        help="Target metric optimized during hyperparameter tuning.",
    )

with col_right:
    st.subheader("3️⃣ Execution Control")
    st.write(
        f"Ready to train on **{len(X):,} customer records** across **{X.shape[1]} attributes** "
        f"with an 80/20 stratified train/test split."
    )

    if st.button("🚀 Launch Model Tournament", type="primary", use_container_width=True, disabled=not selected_models):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, msg: str):
            progress_bar.progress(pct)
            status_text.text(f"⏳ {msg}")

        try:
            exp_results = train_models_pipeline(
                X=X,
                y=y,
                selected_models=selected_models,
                enable_feature_engineering=st.session_state.get("enable_feature_engineering", True),
                quick_mode=quick_mode,
                class_imbalance=imbalance_mode,
                scoring_metric=metric_choice,
                progress_callback=update_progress,
            )

            st.session_state["experiment_results"] = exp_results
            st.session_state["global_shap_data"] = None
            st.session_state["explainer"] = None

            # Auto-save run
            ds_name = st.session_state.get("dataset_name", "dataset").lower().replace(" ", "_")
            saved_meta = save_experiment_run(exp_results, dataset_name=ds_name)
            st.session_state["last_saved_run"] = saved_meta

            st.success(
                f"🎉 Training finished in **{exp_results['total_duration_sec']:.1f} seconds**! "
                f"Champion Model: **{exp_results['best_model_name']}**"
            )
            st.rerun()

        except Exception as e:
            st.error(f"Training failed: {e}")

# Display Results if Available
exp_results = st.session_state.get("experiment_results")
if exp_results is not None:
    st.markdown("---")
    st.subheader(f"🏆 RetenSight Leaderboard · Champion: {exp_results['best_model_name']}")
    
    st.dataframe(
        exp_results["comparison_table"].style.highlight_max(
            subset=["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1-Score", "Accuracy"],
            color="#D1FAE5",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div style="text-align: right; margin-top: 1rem;">
            <em>Next Step: Inspect detailed curves & trade-offs in <strong>Model Comparison</strong>.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
