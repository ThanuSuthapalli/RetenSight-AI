"""End-to-end verification script for churn intelligence platform pipeline."""

import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from src.data.loader import load_sample_dataset
from src.data.validator import validate_dataset, clean_and_prepare_df
from src.models.train import train_models_pipeline
from src.explain.shap_utils import build_shap_explainer, compute_global_shap, explain_single_customer
from src.explain.insights import generate_executive_insights, calculate_revenue_at_risk, get_retention_playbook
from src.models.registry import save_experiment_run

def main():
    print("=== Step 1: Loading Telco Sample Dataset ===", flush=True)
    df = load_sample_dataset("telco")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.", flush=True)

    print("\n=== Step 2: Validating & Cleaning Data ===", flush=True)
    val_res = validate_dataset(df, target_col="Churn", positive_label="Yes", id_col="customerID")
    print(f"Dataset Valid: {val_res.is_valid}", flush=True)
    print(f"Positive Label: {val_res.positive_label}, Negative: {val_res.negative_label}", flush=True)
    print(f"Warnings: {val_res.warnings}", flush=True)

    X, y, clean_report = clean_and_prepare_df(
        df,
        target_col="Churn",
        positive_label="Yes",
        id_col="customerID",
    )
    print(f"Cleaned feature matrix: {X.shape}, Target distribution: {y.value_counts().to_dict()}", flush=True)

    print("\n=== Step 3: Training & Tuning Models Pipeline ===", flush=True)
    start_time = time.time()
    exp_results = train_models_pipeline(
        X,
        y,
        selected_models=["Logistic Regression", "Random Forest", "XGBoost"],
        enable_feature_engineering=True,
        quick_mode=True,
        class_imbalance="balanced_weights",
        scoring_metric="roc_auc",
    )
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds.", flush=True)
    print(f"Best Model: {exp_results['best_model_name']}", flush=True)
    print("\nModel Comparison Table:", flush=True)
    print(exp_results["comparison_table"].to_string(), flush=True)

    print("\n=== Step 4: SHAP Explainability Engine ===", flush=True)
    best_model = exp_results["models"][exp_results["best_model_name"]]["estimator"]
    explainer = build_shap_explainer(
        best_model,
        exp_results["X_train_proc"],
        model_name=exp_results["best_model_name"],
    )
    global_shap = compute_global_shap(
        explainer,
        exp_results["X_test_proc"],
        feature_names=exp_results["feature_names"],
        max_samples=250,
    )
    print("Top 5 Global Churn Drivers:", flush=True)
    print(global_shap["importance_df"].head(5).to_string(), flush=True)

    # Local customer explanation
    sample_cust_proc = exp_results["X_test_proc"][0]
    local_exp = explain_single_customer(
        explainer,
        sample_cust_proc,
        feature_names=exp_results["feature_names"],
    )
    print("\nSingle Customer Risk Explanation:", flush=True)
    for bullet in local_exp["bullet_reasons"]:
        print(f"  {bullet}", flush=True)

    print("\n=== Step 5: Business Insights & Revenue at Risk ===", flush=True)
    insights = generate_executive_insights(df, target_col="Churn", positive_label="Yes")
    print(f"Generated {len(insights)} Executive Insight Cards.", flush=True)
    for ins in insights:
        print(f"  • [{ins['category']}] {ins['title']}: {ins['metric']} -> {ins['description'][:80]}...", flush=True)

    y_prob = exp_results["models"][exp_results["best_model_name"]]["y_prob_test"]
    rev_risk = calculate_revenue_at_risk(exp_results["X_test_df"], y_prob)
    print(f"\nRevenue At Risk: ₹{rev_risk['high_risk_monthly_loss']:,.2f}/month (₹{rev_risk['high_risk_annual_loss']:,.2f}/year)", flush=True)
    print(f"High-Risk Customer Count: {rev_risk['high_risk_count']} ({rev_risk['high_risk_pct']}%)", flush=True)

    playbook = get_retention_playbook(global_shap["top_features"][:3])
    print("\nPrescriptive Retention Actions:", flush=True)
    for pb in playbook:
        print(f"  • {pb['action']} ({pb['target_team']}): {pb['detail']}", flush=True)

    print("\n=== Step 6: Model Serialization & Registry ===")
    saved_run = save_experiment_run(exp_results, dataset_name="telco_churn")
    print(f"Model saved successfully to: {saved_run['model_path']}", flush=True)
    print(f"Report saved successfully to: {saved_run['report_path']}", flush=True)

    print("\n>>> ALL PIPELINE TESTS PASSED SUCCESSFULLY! <<<", flush=True)

if __name__ == "__main__":
    main()
