"""
Tests for src/ml/explain.py

Run with: python -m pytest tests/test_explain.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.explain import (  # noqa: E402
    FEATURE_COLS,
    FEATURE_LABELS,
    explain_global,
    explain_prediction,
    build_explanation_text,
)


# --- Configuration -----------------------------------------------------------

def test_feature_cols_has_five_features():
    assert len(FEATURE_COLS) == 5


def test_feature_cols_does_not_include_length_ratio():
    """Regression guard — length_ratio was removed for being leaky."""
    assert "length_ratio" not in FEATURE_COLS


def test_feature_cols_does_not_include_tfidf_similarity():
    """Regression guard — tfidf_similarity is the label source."""
    assert "tfidf_similarity" not in FEATURE_COLS


def test_all_features_have_labels():
    for f in FEATURE_COLS:
        assert f in FEATURE_LABELS


# --- Global explanation ------------------------------------------------------

def test_explain_global_returns_expected_keys():
    result = explain_global(sample_size=50)
    assert "features" in result
    assert "n_samples" in result


def test_explain_global_all_features_ranked():
    result = explain_global(sample_size=50)
    assert len(result["features"]) == len(FEATURE_COLS)
    ranks = [f["importance_rank"] for f in result["features"]]
    assert ranks == list(range(1, len(FEATURE_COLS) + 1))


def test_explain_global_features_sorted_descending():
    result = explain_global(sample_size=50)
    shapes = [f["mean_abs_shap"] for f in result["features"]]
    assert shapes == sorted(shapes, reverse=True)


def test_explain_global_skill_match_is_top_feature():
    """
    After removing length_ratio, skill_match_score should be the top
    feature globally — it's the most meaningful signal for resume-JD match.
    """
    result = explain_global(sample_size=200)
    top = result["features"][0]
    assert top["name"] == "skill_match_score", (
        f"Expected skill_match_score at #1, got {top['name']}"
    )


def test_explain_global_all_shap_non_negative():
    result = explain_global(sample_size=50)
    for f in result["features"]:
        assert f["mean_abs_shap"] >= 0.0


# --- Local explanation -------------------------------------------------------

STRONG = {
    "skill_match_score": 0.75,
    "skill_match_required": 0.70,
    "experience_match": 1.0,
    "education_match": 1.0,
    "category_match": 1.0,
}

WEAK = {
    "skill_match_score": 0.1,
    "skill_match_required": 0.05,
    "experience_match": 0.3,
    "education_match": 0.3,
    "category_match": 0.0,
}


def test_explain_prediction_returns_expected_keys():
    result = explain_prediction(STRONG)
    for key in ["prediction", "probability", "base_value", "contributions"]:
        assert key in result


def test_explain_prediction_strong_candidate_high_proba():
    result = explain_prediction(STRONG)
    assert result["probability"] > 0.5


def test_explain_prediction_weak_candidate_low_proba():
    result = explain_prediction(WEAK)
    assert result["probability"] < 0.5


def test_explain_prediction_all_features_present():
    result = explain_prediction(STRONG)
    contrib_features = {c["feature"] for c in result["contributions"]}
    assert contrib_features == set(FEATURE_COLS)


def test_explain_prediction_sorted_by_abs_shap():
    result = explain_prediction(STRONG)
    abs_shaps = [abs(c["shap"]) for c in result["contributions"]]
    assert abs_shaps == sorted(abs_shaps, reverse=True)


def test_explain_prediction_strong_skill_match_drives_prediction():
    """For a strong candidate, skill_match_score should be the top contributor."""
    result = explain_prediction(STRONG)
    top = result["contributions"][0]
    assert top["feature"] == "skill_match_score"
    assert top["shap"] > 0


def test_explain_prediction_handles_missing_keys():
    """Features not provided default to 0.0 — should not crash."""
    result = explain_prediction({})
    assert "prediction" in result
    assert "contributions" in result


def test_explain_prediction_prediction_is_binary():
    result = explain_prediction(STRONG)
    assert result["prediction"] in (0, 1)


# --- Human-readable text -----------------------------------------------------

def test_build_explanation_returns_string():
    text = build_explanation_text(STRONG)
    assert isinstance(text, str)
    assert len(text) > 20


def test_build_explanation_mentions_prediction():
    text = build_explanation_text(STRONG)
    assert "Prediction" in text or "MATCH" in text


def test_build_explanation_mentions_top_factors():
    text = build_explanation_text(STRONG)
    assert "Top contributing factors" in text or "skill" in text.lower()