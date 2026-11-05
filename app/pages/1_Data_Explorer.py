"""Page 1: Exploratory Data Analysis, Distributions, and Segment Breakdown."""

from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from app.utils import apply_custom_styles, init_session_state, render_sidebar_status
from src.viz.plots import plot_segment_churn_bar, plot_correlation_heatmap
from src.explain.insights import compute_segment_churn_stats

st.set_page_config(page_title="RetenSight AI · Data Explorer", page_icon=":material/analytics:", layout="wide")
apply_custom_styles()
init_session_state()
render_sidebar_status()

st.title("Customer Cohort & Churn Exploratory Analysis")
st.markdown("Analyze baseline attrition rates, identify high-risk customer segments, and inspect feature correlations.")

df = st.session_state.get("raw_df")
target_col = st.session_state.get("target_col")
pos_label = st.session_state.get("positive_label")

if df is None or target_col is None:
    st.warning("⚠️ No dataset loaded. Please navigate to the **Home** page to load data first.")
    st.stop()

# Overall KPI Banner
y_series = df[target_col].dropna()
churn_count = int((y_series.astype(str).str.strip() == str(pos_label).strip()).sum())
total_count = len(y_series)
churn_rate = (churn_count / max(total_count, 1)) * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Cohort Size", f"{total_count:,}")
k2.metric("Churned Accounts", f"{churn_count:,}")
k3.metric("Baseline Churn Rate", f"{churn_rate:.1f}%")
k4.metric("Retention Rate", f"{100 - churn_rate:.1f}%")

st.markdown("---")

tab_segments, tab_numeric, tab_corr = st.tabs([
    "👥 Segment Churn Analysis",
    "📈 Numeric Distributions",
    "🔥 Correlation Matrix",
])

with tab_segments:
    st.subheader("Segment Churn Disparity")
    st.markdown("Select a categorical attribute to pinpoint customer plans with elevated attrition.")
    
    cat_cols = [c for c in df.columns if c != target_col and df[c].nunique() <= 15]
    if cat_cols:
        col_select = st.selectbox("Select Segment Attribute:", cat_cols, index=0)
        stats = compute_segment_churn_stats(df, target_col, pos_label, col_select)
        
        c_left, c_right = st.columns([3, 2], gap="medium")
        with c_left:
            fig = plot_segment_churn_bar(stats, col_select)
            st.pyplot(fig)
        with c_right:
            st.markdown(f"**Segment Data: `{col_select}`**")
            st.dataframe(
                stats.rename(columns={
                    "total_customers": "Total Accounts",
                    "churned_customers": "Churned Accounts",
                    "churn_rate_pct": "Churn Rate (%)",
                }),
                use_container_width=True,
            )
    else:
        st.info("No categorical attributes detected with <= 15 unique levels.")

with tab_numeric:
    st.subheader("Feature Density by Churn Outcome")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    if num_cols:
        selected_num = st.selectbox("Select Numeric Metric:", num_cols, index=0)
        fig, ax = plt.subplots(figsize=(7, 3.8), dpi=100)
        
        y_binary = (df[target_col].astype(str).str.strip() == str(pos_label).strip()).map({True: "Churned", False: "Retained"})
        plot_df = pd.DataFrame({
            selected_num: pd.to_numeric(df[selected_num], errors="coerce"),
            "Status": y_binary,
        }).dropna()
        
        sns.kdeplot(
            data=plot_df,
            x=selected_num,
            hue="Status",
            common_norm=False,
            fill=True,
            palette={"Churned": "#EF4444", "Retained": "#2563EB"},
            alpha=0.4,
            ax=ax,
        )
        ax.set_title(f"Density Distribution: {selected_num}", fontsize=11, fontweight="bold")
        st.pyplot(fig)
    else:
        st.info("No numeric columns available.")

with tab_corr:
    st.subheader("Pearson Correlation Heatmap")
    fig = plot_correlation_heatmap(df)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("Insufficient numeric features to compute correlation matrix.")
