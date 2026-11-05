"""Model persistence, artifact serialization, and metadata logging."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


def ensure_artifact_dirs():
    """Ensure artifacts directories exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_experiment_run(
    experiment_data: Dict[str, Any],
    dataset_name: str = "dataset",
) -> Dict[str, str]:
    """Serialize the trained pipeline, best model, and experiment metadata."""
    ensure_artifact_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_name = experiment_data.get("best_model_name", "model").replace(" ", "_")
    run_id = f"{dataset_name}_{best_model_name}_{timestamp}"

    model_filename = f"{run_id}.joblib"
    report_filename = f"{run_id}.json"

    model_path = MODELS_DIR / model_filename
    report_path = REPORTS_DIR / report_filename

    # Save joblib bundle
    bundle_to_save = {
        "preprocessor": experiment_data["preprocessor"],
        "best_model_name": experiment_data["best_model_name"],
        "estimator": experiment_data["models"][experiment_data["best_model_name"]]["estimator"],
        "feature_names": experiment_data["feature_names"],
        "config": experiment_data.get("config", {}),
        "timestamp": timestamp,
        "run_id": run_id,
    }
    joblib.dump(bundle_to_save, model_path)

    # Save JSON metadata (clean serializable version)
    best_metrics = (
        experiment_data["models"][experiment_data["best_model_name"]].get("test_metrics", {})
    )
    # Ensure serializable types
    clean_metrics = {k: v for k, v in best_metrics.items() if k != "confusion_matrix"}
    
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset_name": dataset_name,
        "best_model_name": experiment_data["best_model_name"],
        "best_params": {
            str(k): str(v)
            for k, v in experiment_data["models"][experiment_data["best_model_name"]].get("best_params", {}).items()
        },
        "test_metrics": clean_metrics,
        "all_models_ranked": (
            experiment_data["comparison_table"].to_dict(orient="records")
            if "comparison_table" in experiment_data
            else []
        ),
        "model_path": str(model_path),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "run_id": run_id,
        "model_path": str(model_path),
        "report_path": str(report_path),
    }


def load_model_bundle(model_path: str | Path) -> Dict[str, Any]:
    """Load a previously serialized model bundle."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path}")
    return joblib.load(path)


def list_saved_models() -> List[Dict[str, Any]]:
    """List all saved models and reports."""
    ensure_artifact_dirs()
    saved = []
    for report_file in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved.append(data)
        except Exception:
            continue
    return saved

