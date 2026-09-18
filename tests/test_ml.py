"""
Tests for ML pipeline — feature engineering, model loading, prediction.

Run with: python -m pytest tests/test_ml.py -v
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.feature_engineering import extract_features  # noqa: E402

MODELS_DIR = ROOT / "models"


# --- Feature engineering -----------------------------------------------------

def test_extract_features_returns_all_columns():
    features = extract_features(
        "Python, Django, PostgreSQL, Docker",
        "We need a Python engineer with Django and AWS experience.",
    )
    expected = {
        "skill_match_score",
        "skill_match_required",
        "experience_match",
        "education_match",
        "category_match",
    }
    assert expected == set(features.keys())


def test_extract_features_empty_inputs():
    features = extract_features("", "")
    for v in features.values():
        assert v == 0.0


def test_extract_features_none_inputs():
    features = extract_features(None, None)
    for v in features.values():
        assert v == 0.0


def test_extract_features_all_in_range():
    features = extract_features(
        "Experienced Python developer with Django, PostgreSQL, Docker, AWS.",
        "We need 5 years Python with Django and AWS.",
    )
    for k, v in features.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of range"


def test_extract_features_skill_match_positive_when_overlap():
    # Resume and JD share Python and Django
    features = extract_features(
        "Skills: Python, Django, PostgreSQL",
        "Required: Python, Django, AWS, Docker",
    )
    assert features["skill_match_score"] > 0
    assert features["skill_match_required"] > 0


# --- Model loading ----------------------------------------------------------

def test_models_directory_exists():
    assert MODELS_DIR.exists(), "models/ directory not found"


def test_all_three_models_exist():
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        path = MODELS_DIR / f"{name}.joblib"
        assert path.exists(), f"model {name}.joblib not found"


def test_scaler_exists():
    assert (MODELS_DIR / "scaler.joblib").exists()


def test_metadata_exists():
    assert (MODELS_DIR / "metadata.json").exists()


def test_test_set_exists():
    assert (MODELS_DIR / "test_set.npz").exists()


# --- Model prediction --------------------------------------------------------

def _load_model(name):
    return joblib.load(MODELS_DIR / f"{name}.joblib")


def _sample_features():
    """Build a plausible 5-feature vector in the right order."""
    return np.array([[
        0.5,   # skill_match_score
        0.4,   # skill_match_required
        0.8,   # experience_match
        0.8,   # education_match
        0.8,   # category_match
    ]])

@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost"])
def test_model_predicts_binary(model_name):
    model = _load_model(model_name)
    X = _sample_features()
    pred = model.predict(X)
    assert pred[0] in (0, 1)


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost"])
def test_model_predicts_proba(model_name):
    model = _load_model(model_name)
    X = _sample_features()
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0][0] <= 1.0
    assert 0.0 <= proba[0][1] <= 1.0
    assert abs(proba[0].sum() - 1.0) < 1e-6


# --- Metadata check ----------------------------------------------------------

def test_metadata_has_expected_keys():
    import json
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    for key in ["trained_at", "n_samples", "feature_columns",
                "class_distribution", "results"]:
        assert key in meta


def test_metadata_feature_columns_has_no_leak():
    """The feature list must NOT contain tfidf_similarity or length_ratio."""
    import json
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert "tfidf_similarity" not in meta["feature_columns"]
    assert "length_ratio" not in meta["feature_columns"]