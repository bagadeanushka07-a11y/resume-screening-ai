"""
Evaluate trained models on the held-out test set.

Loads:
    models/{logistic_regression,random_forest,xgboost}.joblib
    models/test_set.npz    (X_test, y_test — scaled)

Produces:
    - Console report with accuracy, precision, recall, F1, ROC-AUC
    - Confusion matrices (saved as PNGs in reports/)
    - ROC curve comparison plot
    - Model comparison bar chart
    - reports/evaluation_results.json

Run:
    python -m src.ml.evaluate
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TEST_SET = MODELS_DIR / "test_set.npz"
RESULTS_JSON = REPORTS_DIR / "evaluation_results.json"


MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
COLORS = {
    "logistic_regression": "#1f77b4",
    "random_forest": "#2ca02c",
    "xgboost": "#d62728",
}


# --- Helpers -----------------------------------------------------------------

def _load_test_set():
    if not TEST_SET.exists():
        raise FileNotFoundError(
            f"Test set not found: {TEST_SET}. Run 'python -m src.ml.train' first."
        )
    data = np.load(TEST_SET)
    return data["X_test"], data["y_test"]


def _load_models() -> Dict[str, object]:
    models = {}
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run training first.")
        models[name] = joblib.load(path)
    return models


def _compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


# --- Plots -------------------------------------------------------------------

def _plot_confusion_matrices(metrics_by_model: Dict, cms: Dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, MODEL_NAMES):
        cm = cms[name]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["No Match", "Match"],
            yticklabels=["No Match", "Match"],
        )
        ax.set_title(f"{DISPLAY_NAMES[name]}\nROC-AUC: {metrics_by_model[name]['roc_auc']}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    out = REPORTS_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved {out}")


def _plot_roc_curves(curves: Dict, metrics_by_model: Dict) -> None:
    plt.figure(figsize=(7, 6))
    for name in MODEL_NAMES:
        fpr, tpr, _ = curves[name]
        plt.plot(
            fpr, tpr, color=COLORS[name], lw=2,
            label=f"{DISPLAY_NAMES[name]} (AUC = {metrics_by_model[name]['roc_auc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = REPORTS_DIR / "roc_curves.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved {out}")


def _plot_model_comparison(metrics_by_model: Dict) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(MODEL_NAMES):
        values = [metrics_by_model[name][m] for m in metrics]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=DISPLAY_NAMES[name],
                      color=COLORS[name], alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = REPORTS_DIR / "model_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved {out}")


# --- Main --------------------------------------------------------------------

def evaluate() -> Dict:
    print("=" * 60)
    print("PHASE 13 — MODEL EVALUATION")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading test set and models...")
    X_test, y_test = _load_test_set()
    models = _load_models()
    print(f"  Test set: {X_test.shape}  ({int(y_test.sum())} positives, "
          f"{len(y_test) - int(y_test.sum())} negatives)")

    metrics_by_model: Dict[str, Dict] = {}
    cms: Dict[str, np.ndarray] = {}
    curves: Dict[str, tuple] = {}

    for name, model in models.items():
        print(f"\n--- {DISPLAY_NAMES[name]} ---")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = _compute_metrics(y_test, y_pred, y_proba)
        metrics_by_model[name] = metrics
        cms[name] = confusion_matrix(y_test, y_pred)
        curves[name] = roc_curve(y_test, y_proba)

        for k, v in metrics.items():
            print(f"  {k:10} {v}")

    # --- Plots ---
    print("\nGenerating plots...")
    _plot_confusion_matrices(metrics_by_model, cms)
    _plot_roc_curves(curves, metrics_by_model)
    _plot_model_comparison(metrics_by_model)

    # --- Pick best model by ROC-AUC ---
    best_model = max(metrics_by_model, key=lambda n: metrics_by_model[n]["roc_auc"])
    print(f"\n🏆 Best model by ROC-AUC: {DISPLAY_NAMES[best_model]} "
          f"({metrics_by_model[best_model]['roc_auc']})")

    # --- Save results ---
    results = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "test_set_size": int(len(y_test)),
        "test_set_positives": int(y_test.sum()),
        "test_set_negatives": int(len(y_test) - y_test.sum()),
        "metrics": metrics_by_model,
        "confusion_matrices": {n: cm.tolist() for n, cm in cms.items()},
        "best_model": best_model,
        "best_model_roc_auc": metrics_by_model[best_model]["roc_auc"],
        "notes": (
            "Metrics computed on the held-out test set (15% of full dataset), "
            "which was never seen during training or SMOTE. "
            "Labels are weak supervision from TF-IDF similarity; the models "
            "predict from independent features (skill match, experience, "
            "education, category, length)."
        ),
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_JSON}")

    print("\n" + "=" * 60)
    print("DONE — Phase 13 complete")
    print("=" * 60)
    return results


if __name__ == "__main__":
    evaluate()