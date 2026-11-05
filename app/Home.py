"""Home Page: Dataset Ingestion, Demo Loader, and Schema Configuration."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from app.utils import apply_custom_styles, init_session_state, render_sidebar_status
from src.data.loader import load_dataset, load_sample_dataset, get_dataset_summary, preview_data
from src.data.validator import validate_dataset, clean_and_prepare_df

st.set_page_config(
    page_title="RetenSight AI · Overview",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_styles()
init_session_state()
render_sidebar_status()

# Cached sample dataset loaders for instant response time (< 0.05s)
@st.cache_data(show_spinner=False)
def get_cached_sample(name: str):
    return load_sample_dataset(name)

# RetenSight Hero Banner
st.markdown(
    """
    <div class="retensight-hero">
        <div class="retensight-hero-title">⚡ RetenSight AI</div>
        <p class="retensight-hero-subtitle">
            Next-generation customer intelligence & churn prediction platform. Turn raw customer data 
            into explainable attrition risk models, quantified revenue-at-risk analysis, and automated retention playbooks.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 1️⃣ Ingest Customer Data")
    
    tab_sample, tab_upload = st.tabs(["🚀 Instant Demo Datasets", "📂 Upload Custom CSV / Excel"])
    
    with tab_sample:
        st.write("Test RetenSight AI immediately with calibrated reference data:")
        sample_choice = st.selectbox(
            "Choose Reference Dataset:",
            ["Telco Customer Churn (7,043 rows · Telecom)", "Bank Customer Churn (10,000 rows · Financial)"],
            index=0,
        )
        if st.button("⚡ Load Demo Dataset", type="primary", use_container_width=True):
            with st.spinner("Loading reference dataset..."):
                if "Telco" in sample_choice:
                    df = get_cached_sample("telco")
                    ds_name = "Telco Customer Churn"
                    target_default = "Churn"
                    pos_default = "Yes"
                    id_default = "customerID"
                else:
                    df = get_cached_sample("bank")
                    ds_name = "Bank Customer Churn"
                    target_default = "Exited"
                    pos_default = 1
                    id_default = "CustomerId"

                st.session_state["raw_df"] = df
                st.session_state["dataset_name"] = ds_name
                st.session_state["target_col"] = target_default
                st.session_state["positive_label"] = pos_default
                st.session_state["id_col"] = id_default
                
                # Auto validate
                val_res = validate_dataset(
                    df,
                    target_col=target_default,
                    positive_label=pos_default,
                    id_col=id_default,
                )
                st.session_state["validation_result"] = val_res

                # Auto clean
                X, y, report = clean_and_prepare_df(
                    df,
                    target_col=target_default,
                    positive_label=pos_default,
                    id_col=id_default,
                )
                st.session_state["cleaned_X"] = X
                st.session_state["cleaned_y"] = y
                st.session_state["cleaning_report"] = report
                st.success(f"✓ Loaded {ds_name} successfully!")
                st.rerun()

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload customer file (CSV or XLSX up to 50MB)",
            type=["csv", "xlsx", "xls"],
            help="Upload raw customer records containing historical churn outcomes.",
        )
        if uploaded_file is not None:
            try:
                with st.spinner("Parsing file..."):
                    df = load_dataset(uploaded_file)
                    st.session_state["raw_df"] = df
                    st.session_state["dataset_name"] = uploaded_file.name
                    st.success(f"✓ Uploaded {uploaded_file.name} ({len(df):,} records)")
            except Exception as e:
                st.error(f"Error parsing file: {e}")

with col_right:
    st.markdown("### 2️⃣ Schema & Target Mapping")
    df = st.session_state.get("raw_df")
    
    if df is not None:
        cols = list(df.columns)
        
        default_target_idx = 0
        for i, c in enumerate(cols):
            if c.lower() in ["churn", "churned", "exited", "target", "left", "is_churn"]:
                default_target_idx = i
                break

        selected_target = st.selectbox(
            "Select Target Column (Binary Churn):",
            cols,
            index=default_target_idx,
            help="Column specifying whether customer churned.",
        )
        st.session_state["target_col"] = selected_target

        unique_targets = list(df[selected_target].dropna().unique())
        
        default_pos_idx = 0
        for i, val in enumerate(unique_targets):
            if str(val).lower() in ["yes", "1", "true", "churn", "positive", "y"]:
                default_pos_idx = i
                break

        col_pos, col_id = st.columns(2)
        with col_pos:
            selected_pos = st.selectbox(
                "Positive Class ('Churned'):",
                unique_targets,
                index=default_pos_idx,
                help="The specific value representing customer churn.",
            )
            st.session_state["positive_label"] = selected_pos

        with col_id:
            id_options = ["None"] + cols
            default_id_idx = 0
            for i, c in enumerate(id_options):
                if "id" in c.lower() and c.lower() != selected_target.lower():
                    default_id_idx = i
                    break
            selected_id = st.selectbox(
                "Optional Customer ID Column:",
                id_options,
                index=default_id_idx,
                help="Unique identifier to exclude from ML training.",
            )
            st.session_state["id_col"] = None if selected_id == "None" else selected_id

        if st.button("🔍 Confirm Schema & Prepare Pipeline", type="primary", use_container_width=True):
            with st.spinner("Validating schema and auditing leakage..."):
                val_res = validate_dataset(
                    df,
                    target_col=selected_target,
                    positive_label=selected_pos,
                    id_col=st.session_state["id_col"],
                )
                st.session_state["validation_result"] = val_res

                if val_res.is_valid:
                    X, y, report = clean_and_prepare_df(
                        df,
                        target_col=selected_target,
                        positive_label=selected_pos,
                        id_col=st.session_state["id_col"],
                    )
                    st.session_state["cleaned_X"] = X
                    st.session_state["cleaned_y"] = y
                    st.session_state["cleaning_report"] = report
                    st.success("✓ Schema verified and feature matrix prepared!")
                else:
                    st.error("Validation failed. Check error diagnostics below.")
    else:
        st.info("👈 Please load a demo dataset or upload your customer records to begin.")

# Dataset Overview Section
df = st.session_state.get("raw_df")
if df is not None:
    st.markdown("---")
    st.markdown(f"### 📊 Active Dataset: {st.session_state.get('dataset_name', '')}")
    
    summary = get_dataset_summary(df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Customers", f"{summary['n_rows']:,}")
    m2.metric("Raw Attributes", f"{summary['n_cols']}")
    m3.metric("Memory Ingest", f"{summary['memory_mb']} MB")
    m4.metric("Missing Values", f"{summary['total_missing_cells']:,}")
    m5.metric("Exact Duplicates", f"{summary['duplicate_rows']:,}")

    # Validation message callouts
    val_res = st.session_state.get("validation_result")
    if val_res:
        if val_res.errors:
            for err in val_res.errors:
                st.error(f"❌ {err}")
        if val_res.warnings:
            for warn in val_res.warnings:
                st.warning(f"⚠️ {warn}")

    st.markdown("#### Sample Records (First 10 Accounts)")
    st.dataframe(preview_data(df, 10), use_container_width=True)
