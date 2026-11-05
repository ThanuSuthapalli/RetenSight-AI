"""Page 4: Model Comparison, Overlaid Evaluation Curves, and Dynamic Threshold Slider."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from app.utils import apply_custom_styles, init_session_state, render_sidebar_status
from src.models.evaluate import (
    compute_metrics_at_threshold,
    get_roc_curve_data,
    get_pr_curve_data,
)
from src.viz.plots import plot_confusion_matrix, plot_roc_curves, plot_pr_curves

st.set_page_config(page_title="RetenSight AI · Model Comparison", page_icon=":material/leaderboard:", layout="wide")
apply_custom_styles()
init_session_state()
render_sidebar_status()

st.title("Model Tournament Leaderboard & Threshold Tuner")
st.markdown(
    "Benchmark models on the held-out test set, evaluate discrimination power, "
    "and adjust the **decision threshold** to optimize recall versus false alarm costs."
)

exp_results = st.session_state.get("experiment_results")

if exp_results is None:
    st.warning("⚠️ No trained models available. Please train models in **Model Training** first.")
    st.stop()

# 1. Summary Leaderboard Table
st.subheader("1️⃣ Test Set Evaluation Leaderboard")
st.dataframe(
    exp_results["comparison_table"].style.highlight_max(
        subset=["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1-Score", "Accuracy"],
        color="#D1FAE5",
    ),
    use_container_width=True,
    hide_index=True,
)

best_name = exp_results["best_model_name"]
best_auc = exp_results["comparison_table"].iloc[0]["ROC-AUC"]
st.success(
    f"🏆 **Selected Champion Model:** **{best_name}** (ROC-AUC: {best_auc}) — "
    f"Demonstrated superior ranking performance on unseen customer data."
)

st.markdown("---")

# 2. Overlaid Evaluation Curves
st.subheader("2️⃣ Overlaid ROC and Precision-Recall Curves")
col_roc, col_pr = st.columns(2)

roc_data_dict = {}
pr_data_dict = {}

y_test = exp_results["y_test"]

for m_name, m_info in exp_results["models"].items():
    y_prob = m_info["y_prob_test"]
    roc_data_dict[m_name] = get_roc_curve_data(y_test, y_prob)
    pr_data_dict[m_name] = get_pr_curve_data(y_test, y_prob)

with col_roc:
    fig_roc = plot_roc_curves(roc_data_dict)
    st.pyplot(fig_roc)
    st.caption("ROC Curve measures True Positive Rate vs False Positive Rate across all thresholds.")

with col_pr:
    baseline_churn = float(y_test.mean())
    fig_pr = plot_pr_curves(pr_data_dict, baseline_pos_rate=baseline_churn)
    st.pyplot(fig_pr)
    st.caption("Precision-Recall curve illustrates alarm precision at high-recall operating points.")

st.markdown("---")

# 3. Dynamic Threshold Tuning
st.subheader("3️⃣ Interactive Decision Threshold Optimizer")
st.markdown(
    "Fine-tune the classification threshold to match your business economics. "
    "A lower threshold catches more churners (higher recall); a higher threshold reduces false alarms."
)

model_names = list(exp_results["models"].keys())
selected_model_name = st.selectbox("Select Model to Tune:", model_names, index=model_names.index(best_name))
selected_prob = exp_results["models"][selected_model_name]["y_prob_test"]

threshold = st.slider(
    "Decision Probability Threshold ($t$):",
    min_value=0.05,
    max_value=0.95,
    value=float(st.session_state.get("decision_threshold", 0.50)),
    step=0.01,
    help="Default is 0.50. Adjust to see live confusion matrix and financial trade-offs.",
)
st.session_state["decision_threshold"] = threshold

metrics_t = compute_metrics_at_threshold(y_test, selected_prob, threshold=threshold)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Recall (Caught Churners)", f"{metrics_t['recall']:.1%}")
c2.metric("Precision (Alarm Accuracy)", f"{metrics_t['precision']:.1%}")
c3.metric("F1-Score", f"{metrics_t['f1']:.3f}")
c4.metric("Accuracy", f"{metrics_t['accuracy']:.1%}")

col_cm, col_biz = st.columns([1, 1], gap="large")

with col_cm:
    st.markdown(f"#### Confusion Matrix ($t = {threshold:.2f}$)")
    fig_cm = plot_confusion_matrix(metrics_t["confusion_matrix"], model_name=f"{selected_model_name} (t={threshold:.2f})")
    st.pyplot(fig_cm)

with col_biz:
    st.markdown("#### Operational Trade-Offs")
    st.write(
        f"""
        - 👥 **Held-out Test Cohort:** {metrics_t['total_samples']:,}
        - 🚨 **Actual Churners:** {metrics_t['actual_churners']:,}
        - ✅ **Caught by RetenSight AI:** {metrics_t['caught_churners']:,} ({metrics_t['recall']:.1%})
        - ❌ **Missed Attrition:** {metrics_t['missed_churners']:,}
        - ⚠️ **False Alarms:** {metrics_t['false_alarms']:,}
        """
    )
    if threshold < 0.40:
        st.info("💡 **Aggressive Retention Strategy:** Prioritizes catching every churner; ideal for low-cost digital perks or emails.")
    elif threshold > 0.60:
        st.info("💡 **Targeted High-Touch Strategy:** Preserves budget for expensive VIP discounts by only acting on high-certainty accounts.")
    else:
        st.info("💡 **Balanced Strategy:** Neutral threshold trading off retention coverage and intervention expense.")
