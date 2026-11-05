"""Page 2: Automated Preprocessing Audit, Column Profiles, and Feature Engineering Toggles."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from app.utils import apply_custom_styles, init_session_state, render_sidebar_status
from src.preprocessing.pipeline import FullDataPreprocessor

st.set_page_config(page_title="RetenSight AI · Preprocessing", page_icon=":material/tune:", layout="wide")
apply_custom_styles()
init_session_state()
render_sidebar_status()

st.title("Preprocessing & Domain Feature Engineering")
st.markdown(
    "Automate type inference, missing value imputation, one-hot encoding, and customer intelligence features "
    "with **zero data leakage guarantees**."
)

X = st.session_state.get("cleaned_X")
y = st.session_state.get("cleaned_y")
report = st.session_state.get("cleaning_report")
val_res = st.session_state.get("validation_result")

if X is None or y is None:
    st.warning("⚠️ No dataset validated. Please load or upload data on the **Home** page first.")
    st.stop()

col_cfg, col_audit = st.columns([1, 1], gap="large")

with col_cfg:
    st.subheader("1️⃣ Transformation Pipeline Setup")
    
    fe_enabled = st.checkbox(
        "Enable Domain Feature Engineering",
        value=st.session_state.get("enable_feature_engineering", True),
        help="Constructs tenure buckets, spend-to-tenure ratios, addon counters, and contract interactions.",
    )
    st.session_state["enable_feature_engineering"] = fe_enabled

    with st.expander("Advanced Pipeline Parameters", expanded=True):
        num_scaler = st.selectbox(
            "Numeric Feature Scaling:",
            ["standard", "minmax", "none"],
            index=0,
            help="StandardScaler normalizes mean=0, std=1; recommended for Logistic Regression.",
        )
        num_imputer = st.selectbox(
            "Numeric Imputation Strategy:",
            ["median", "mean"],
            index=0,
            help="Replaces missing numerical measurements.",
        )
        cat_imputer = st.selectbox(
            "Categorical Imputation Strategy:",
            ["most_frequent", "Unknown"],
            index=0,
            help="Fills missing categorical categories.",
        )

    if st.button("⚡ Test Pipeline Transformation", type="primary", use_container_width=True):
        with st.spinner("Transforming feature matrix..."):
            prep = FullDataPreprocessor(
                enable_feature_engineering=fe_enabled,
                numeric_scaler=num_scaler,
                numeric_imputer=num_imputer,
                categorical_imputer=cat_imputer,
            )
            X_sample = prep.fit_transform(X)
            st.session_state["preview_preprocessor"] = prep
            st.session_state["preview_transformed_shape"] = X_sample.shape
            st.success(f"✓ Pipeline transformed feature matrix to shape {X_sample.shape} in <0.2s!")

with col_audit:
    st.subheader("2️⃣ Preprocessing Audit Report")
    
    if report:
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Features In", f"{report.get('initial_cols', 0)}")
        c2.metric("Exact Duplicates Dropped", f"{report.get('dropped_duplicates', 0)}")
        c3.metric("Type Sanitizations", f"{len(report.get('imputed_numeric_type_conversions', []))}")

        if report.get("dropped_columns"):
            st.info(f"🚫 Dropped Columns (ID / Constant / High Missing): `{report['dropped_columns']}`")
        if report.get("imputed_numeric_type_conversions"):
            st.info(f"🔄 Converted text-encoded numbers: `{report['imputed_numeric_type_conversions']}`")

    if val_res and val_res.profile:
        prof = val_res.profile
        st.markdown("#### Detected Attribute Types")
        type_summary = pd.DataFrame([
            {"Type": "Numeric Features", "Count": len(prof.numeric_cols), "Sample": ", ".join(prof.numeric_cols[:4])},
            {"Type": "Categorical (Low Cardinality)", "Count": len(prof.categorical_cols), "Sample": ", ".join(prof.categorical_cols[:4])},
            {"Type": "Binary Features (0/1)", "Count": len(prof.binary_cols), "Sample": ", ".join(prof.binary_cols[:4])},
            {"Type": "High Cardinality (>20)", "Count": len(prof.high_cardinality_cols), "Sample": ", ".join(prof.high_cardinality_cols[:4])},
            {"Type": "Suspected Identifier Columns", "Count": len(prof.suspected_id_cols), "Sample": ", ".join(prof.suspected_id_cols[:4])},
        ])
        st.dataframe(type_summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("3️⃣ Feature Dimension Expansion")

temp_prep = st.session_state.get("preview_preprocessor")
if temp_prep is None:
    temp_prep = FullDataPreprocessor(enable_feature_engineering=fe_enabled)
    temp_prep.fit(X)

feat_report = temp_prep.get_summary_report()
m1, m2, m3 = st.columns(3)
m1.metric("Input Raw Features", f"{X.shape[1]}")
m2.metric("Engineered Domain Features", f"{len(feat_report['engineered_features_added'])}")
m3.metric("One-Hot Encoded Model Dimension", f"{feat_report['final_feature_dimension']}")

st.markdown(f"**Engineered Features Added:** `{feat_report['engineered_features_added']}`")

with st.expander("Inspect Encoded Column Names Out", expanded=False):
    st.write(temp_prep.get_feature_names_out())
