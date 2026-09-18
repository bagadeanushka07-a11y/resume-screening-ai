"""
Tests for src/ml/evaluate.py

Run with: python -m pytest tests/test_evaluate.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


# --- Artifact existence ------------------------------------------------------

def test_evaluation_results_exists():
    assert (REPORTS_DIR / "evaluation_results.json").exists()


def test_roc_curves_plot_exists():
    assert (REPORTS_DIR / "roc_curves.png").exists()


def test_confusion_matrices_plot_exists():
    assert (REPORTS_DIR / "confusion_matrices.png").exists()


def test_model_comparison_plot_exists():
    assert (REPORTS_DIR / "model_comparison.png").exists()


# --- Results content ---------------------------------------------------------

def test_results_has_expected_keys():
    with open(REPORTS_DIR / "evaluation_results.json", encoding="utf-8") as f:
        results = json.load(f)
    for key in ["evaluated_at", "test_set_size", "metrics",
                "confusion_matrices", "best_model", "best_model_roc_auc"]:
        assert key in results


def test_results_has_all_three_models():
    with open(REPORTS_DIR / "evaluation_results.json", encoding="utf-8") as f:
        results = json.load(f)
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        assert name in results["metrics"]
        assert name in results["confusion_matrices"]


def test_each_model_has_all_metrics():
    with open(REPORTS_DIR / "evaluation_results.json", encoding="utf-8") as f:
        results = json.load(f)
    for name, m in results["metrics"].items():
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in m, f"{name} missing {key}"
            assert 0.0 <= m[key] <= 1.0


def test_metrics_are_realistic_not_perfect():
    """Sanity: metrics should NOT be 1.0 — that would indicate leakage."""
    with open(REPORTS_DIR / "evaluation_results.json", encoding="utf-8") as f:
        results = json.load(f)
    for name, m in results["metrics"].items():
        assert m["roc_auc"] < 0.99, f"{name} ROC-AUC suspiciously high: {m['roc_auc']}"
        assert m["accuracy"] < 0.99, f"{name} accuracy suspiciously high: {m['accuracy']}"


def test_best_model_is_valid():
    with open(REPORTS_DIR / "evaluation_results.json", encoding="utf-8") as f:
        results = json.load(f)
    assert results["best_model"] in ["logistic_regression", "random_forest", "xgboost"]


def test_best_model_roc_auc_matches_metrics():
    with open(REPORTS_DIR / "evaluation_results.json", encoding="utf-8") as f:
        results = json.load(f)
    best = results["best_model"]
    assert abs(results["best_model_roc_auc"] - results["metrics"][best]["roc_auc"]) < 1e-6


def test_confusion_matrices_shape():
    with open(REPORTS_DIR / "evaluation_results.json", encoding="utf-8") as f:
        results = json.load(f)
    for name, cm in results["confusion_matrices"].items():
        cm_arr = np.array(cm)
        assert cm_arr.shape == (2, 2), f"{name} confusion matrix shape wrong"
        # Each cell should be a non-negative integer
        assert (cm_arr >= 0).all()
        # Total should equal test set size
        assert cm_arr.sum() == results["test_set_size"]


def test_test_set_size_matches_npz():
    with open(REPORTS_DIR / "evaluation_results.json", encoding="utf-8") as f:
        results = json.load(f)
    data = np.load(MODELS_DIR / "test_set.npz")
    assert results["test_set_size"] == len(data["y_test"])