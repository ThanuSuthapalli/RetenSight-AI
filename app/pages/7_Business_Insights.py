"""Page 7: Executive Business Insights, Revenue at Risk, and Retention Playbooks."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from app.utils import apply_custom_styles, init_session_state, render_sidebar_status
from src.explain.insights import (
    generate_executive_insights,
    calculate_revenue_at_risk,
    get_retention_playbook,
)

st.set_page_config(page_title="RetenSight AI · Business Insights", page_icon=":material/trending_up:", layout="wide")
apply_custom_styles()
init_session_state()
render_sidebar_status()

st.title("Executive Churn Insights & Revenue at Risk")
st.markdown(
    "Translate complex machine learning findings into **executive business KPIs**, "
    "quantify potential annualized revenue loss, and execute targeted retention playbooks."
)

df = st.session_state.get("raw_df")
target_col = st.session_state.get("target_col")
pos_label = st.session_state.get("positive_label")
exp_results = st.session_state.get("experiment_results")

if df is None or target_col is None:
    st.warning("⚠️ No dataset loaded. Please navigate to the **Home** page to load data first.")
    st.stop()

# 1. Revenue at Risk Financial Exposure
st.subheader("Revenue at Risk Financial Exposure")

if exp_results is not None:
    best_name = exp_results["best_model_name"]
    y_prob = exp_results["models"][best_name]["y_prob_test"]
    rev_data = calculate_revenue_at_risk(exp_results["X_test_df"], y_prob)
    
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predictive Monthly Loss", f"₹{rev_data['high_risk_monthly_loss']:,.2f}")
    r2.metric("Annualized Exposure", f"₹{rev_data['high_risk_annual_loss']:,.2f}")
    r3.metric("High-Risk Accounts", f"{rev_data['high_risk_count']:,}")
    r4.metric("Share of Test Accounts", f"{rev_data['high_risk_pct']}%")
    
    st.caption(f"Financial exposure calculated using champion model ({best_name}) on column: <code>{rev_data['revenue_column_used']}</code>")
else:
    # Compute historical revenue exposure directly from dataset
    cols_lower = {str(c).lower(): c for c in df.columns}
    rev_col = (
        cols_lower.get("monthlycharges")
        or cols_lower.get("monthly_charges")
        or cols_lower.get("balance")
        or cols_lower.get("revenue")
    )
    churn_mask = df[target_col].astype(str).str.strip() == str(pos_label).strip()
    churn_cnt = int(churn_mask.sum())
    churn_pct = (churn_cnt / max(len(df), 1)) * 100

    if rev_col is not None:
        churned_rev = pd.to_numeric(df.loc[churn_mask, rev_col], errors="coerce").fillna(0)
        monthly_loss = float(churned_rev.sum())
        annual_loss = monthly_loss * 12
        rev_used_label = rev_col
    else:
        # Benchmark fallback ₹50/customer/month
        monthly_loss = float(churn_cnt * 50.0)
        annual_loss = monthly_loss * 12
        rev_used_label = "Estimated Benchmark (₹50/mo per account)"

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Historical Monthly Loss", f"₹{monthly_loss:,.2f}")
    r2.metric("Annualized Churn Exposure", f"₹{annual_loss:,.2f}")
    r3.metric("Churned Accounts", f"{churn_cnt:,}")
    r4.metric("Cohort Churn Rate", f"{churn_pct:.1f}%")
    
    st.caption(
        f"Historical financial loss from actual churners using: <code>{rev_used_label}</code>. "
        "Train a model in <strong>Model Training</strong> to quantify <em>future predictive</em> revenue at risk.",
        unsafe_allow_html=True,
    )

st.markdown("---")

# 2. Executive Insight Cards
st.subheader("Automated Strategic Insights")
insights = generate_executive_insights(df, target_col=target_col, positive_label=pos_label)

cols = st.columns(2)
for idx, card in enumerate(insights):
    target_c = cols[idx % 2]
    with target_c:
        st.markdown(
            f"""
            <div class="retensight-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.15rem;"><strong>{card['title']}</strong></span>
                    <span class="badge-med">{card['badge']}</span>
                </div>
                <div style="font-size: 1.45rem; font-weight: 800; color: #2563EB; margin-bottom: 0.5rem;">
                    {card['metric']}
                </div>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                    {card['description']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# 3. Prescriptive Retention Playbook
st.subheader("Prescriptive Retention Playbook")
st.markdown("Actionable cross-functional retention campaigns mapped directly from the top attrition drivers:")

top_drivers = ["contract", "support", "tenure", "payment", "charge"]
if exp_results and st.session_state.get("global_shap_data"):
    top_drivers = st.session_state["global_shap_data"].get("top_features", top_drivers)

playbook = get_retention_playbook(top_drivers[:4])

for idx, action in enumerate(playbook, 1):
    with st.expander(f"Campaign #{idx}: {action['action']} · Team: {action['target_team']}", expanded=True):
        st.write(action["detail"])
        st.markdown(f"**Action Owner:** `{action['target_team']}`")
