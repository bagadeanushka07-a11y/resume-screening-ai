"""
SHAP explainability for the trained Logistic Regression model.

Given a feature vector (now 5 features after removing `length_ratio`),
produce:
    - Global feature importance (mean |SHAP| across a sample)
    - Per-prediction local explanations (which features helped/hurt)

Public API:
    - explain_global(sample_size=200) -> dict with global feature importance
    - explain_prediction(features_dict) -> dict with local contributions
    - build_explanation_text(features_dict) -> human-readable summary

Design notes:
    - We explain Logistic Regression — our best-generalizing model from
      Phase 13. Explaining Random Forest or XGBoost would work too, but
      those models overfit and their SHAP values would reflect noise.
    - We removed `length_ratio` from features because SHAP showed it
      dominated the model despite having no semantic meaning for hiring.
      It correlated with the TF-IDF label source. See Phase 17 notes.
    - Inputs to the scaler are passed as DataFrames (with named columns)
      to avoid the sklearn "feature names" UserWarning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"


# --- Features used by the Logistic Regression model (5, no length_ratio) -----
FEATURE_COLS = [
    "skill_match_score",
    "skill_match_required",
    "experience_match",
    "education_match",
    "category_match",
]

# Human-friendly descriptions used in the UI
FEATURE_LABELS = {
    "skill_match_score":    "Overall skill overlap with the JD",
    "skill_match_required": "Overlap with the JD's required skills only",
    "experience_match":     "Your experience vs. the JD's requirement",
    "education_match":      "Your education vs. the JD's requirement",
    "category_match":       "Category alignment (e.g. IT resume vs IT role)",
}


# --- Lazy-loaded artifacts ---------------------------------------------------
_MODEL = None
_SCALER = None
_EXPLAINER = None
_BACKGROUND = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        path = MODELS_DIR / "logistic_regression.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. Run 'python -m src.ml.train' first."
            )
        _MODEL = joblib.load(path)
    return _MODEL


def _load_scaler():
    global _SCALER
    if _SCALER is None:
        path = MODELS_DIR / "scaler.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Scaler not found: {path}")
        _SCALER = joblib.load(path)
    return _SCALER


def _load_background(n: int = 200) -> np.ndarray:
    """Load a background sample for the SHAP explainer (scaled)."""
    global _BACKGROUND
    if _BACKGROUND is None:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATASET_PATH}. "
                "Run 'python -m src.ml.build_dataset' first."
            )
        df = pd.read_csv(DATASET_PATH)
        # Only keep rows with all our features present
        X = df[FEATURE_COLS].dropna()
        if len(X) > n:
            X = X.sample(n=n, random_state=42)
        scaler = _load_scaler()
        # Passing a DataFrame (with column names) avoids the sklearn warning
        _BACKGROUND = scaler.transform(X)
    return _BACKGROUND


def _get_explainer() -> shap.LinearExplainer:
    global _EXPLAINER
    if _EXPLAINER is None:
        model = _load_model()
        background = _load_background()
        _EXPLAINER = shap.LinearExplainer(model, background)
    return _EXPLAINER


# --- Public API --------------------------------------------------------------

def explain_global(sample_size: int = 200) -> Dict:
    """
    Compute global feature importance as mean |SHAP value| across a sample
    of the training data.

    Returns:
        {
            "features": [
                {
                    "name": ..., "label": ..., "mean_abs_shap": ...,
                    "importance_rank": 1..N
                },
                ...
            ],
            "n_samples": int,
        }
    """
    model = _load_model()
    scaler = _load_scaler()

    df = pd.read_csv(DATASET_PATH)[FEATURE_COLS].dropna()
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Pass DataFrame to scaler → no warning
    X_scaled = scaler.transform(df)

    explainer = _get_explainer()
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    mean_abs = np.abs(shap_values).mean(axis=0)

    ranked_indices = np.argsort(mean_abs)[::-1]
    features = []
    for rank, idx in enumerate(ranked_indices, start=1):
        features.append({
            "name": FEATURE_COLS[idx],
            "label": FEATURE_LABELS.get(FEATURE_COLS[idx], FEATURE_COLS[idx]),
            "mean_abs_shap": round(float(mean_abs[idx]), 6),
            "importance_rank": rank,
        })

    return {"features": features, "n_samples": int(len(df))}


def explain_prediction(features_dict: Dict[str, float]) -> Dict:
    """
    Compute SHAP values for a single prediction.

    Args:
        features_dict: e.g.
            {"skill_match_score": 0.5, "skill_match_required": 0.4, ...}

    Returns:
        {
            "prediction": 0 or 1,
            "probability": float,
            "base_value": float,
            "contributions": [
                {"feature": ..., "label": ..., "value": ..., "shap": ...},
                ...
            ],   # sorted by |shap| desc
        }
    """
    model = _load_model()
    scaler = _load_scaler()

    # Build a single-row DataFrame in the correct order — avoids the
    # sklearn "X does not have valid feature names" UserWarning.
    vector_df = pd.DataFrame(
        [{c: float(features_dict.get(c, 0.0)) for c in FEATURE_COLS}],
        columns=FEATURE_COLS,
    )
    vector_scaled = scaler.transform(vector_df)

    explainer = _get_explainer()
    shap_vals = explainer.shap_values(vector_scaled)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.asarray(shap_vals).ravel()

    # Prediction and probability
    proba = float(model.predict_proba(vector_scaled)[0][1])
    prediction = int(model.predict(vector_scaled)[0])

    # Base value
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[0])
    else:
        base = float(base)

    contributions = []
    for i, col in enumerate(FEATURE_COLS):
        contributions.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "value": round(float(features_dict.get(col, 0.0)), 4),
            "shap": round(float(shap_vals[i]), 6),
        })

    contributions.sort(key=lambda c: -abs(c["shap"]))

    return {
        "prediction": prediction,
        "probability": round(proba, 4),
        "base_value": round(base, 4),
        "contributions": contributions,
    }


def build_explanation_text(features_dict: Dict[str, float]) -> str:
    """
    Build a plain-English explanation of a single prediction.
    """
    result = explain_prediction(features_dict)
    lines = []
    prob = result["probability"]
    verdict = "match" if result["prediction"] == 1 else "no match"
    lines.append(
        f"Prediction: {verdict.upper()} (probability of match: {prob:.0%})"
    )

    top = result["contributions"][:3]
    lines.append("")
    lines.append("Top contributing factors:")
    for c in top:
        direction = "supported" if c["shap"] > 0 else "weakened"
        lines.append(
            f"  • {c['label']} (value={c['value']}) {direction} "
            f"the match by {abs(c['shap']):.3f}"
        )

    return "\n".join(lines)