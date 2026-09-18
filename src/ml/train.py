"""
Train three ML models on the weak-supervised resume-JD match dataset.

Pipeline:
    1. Load data/processed/ml_dataset.csv
    2. Stratified split: 70% train / 15% val / 15% test
    3. Fit StandardScaler on train only
    4. Apply SMOTE on train only (fixes 87/13 imbalance)
    5. Train LogisticRegression, RandomForest, XGBoost with class weights
    6. Save each model to models/ with joblib
    7. Log training metrics + save metadata to models/metadata.json

Run:
    python -m src.ml.train
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODELS_DIR / "metadata.json"


# --- Feature columns (exclude label + tracking + overlap) --------------------
FEATURE_COLS = [
    "skill_match_score",
    "skill_match_required",
    "experience_match",
    "education_match",
    "category_match",
    "length_ratio",
]
LABEL_COL = "label"


# --- Helpers -----------------------------------------------------------------

def _load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Run 'python -m src.ml.build_dataset' first."
        )
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=[LABEL_COL])
    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(int).copy()
    return X, y


def _metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


# --- Main training -----------------------------------------------------------

def train_all(random_state: int = 42) -> Dict[str, Dict]:
    print("=" * 60)
    print("PHASE 12 — MODEL TRAINING")
    print("=" * 60)

    # --- Load ---
    print("\nLoading dataset...")
    X, y = _load_dataset()
    print(f"  Samples: {len(X)}")
    print(f"  Features: {list(X.columns)}")
    print(f"  Label counts:\n{y.value_counts().to_string()}")

    # --- Split (stratified) ---
    print("\nSplitting 70/15/15 (stratified)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Train positives: {int(y_train.sum())}  ({y_train.mean()*100:.1f}%)")

    # --- Scale (fit on train only) ---
    print("\nFitting StandardScaler on train...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- SMOTE on training fold only ---
    print("\nApplying SMOTE to training fold only...")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"  Before SMOTE: {len(X_train)} samples (positives: {int(y_train.sum())})")
    print(f"  After SMOTE:  {len(X_train_res)} samples (positives: {int(y_train_res.sum())})")

    # --- Compute scale_pos_weight for XGBoost ---
    n_pos = int(y_train_res.sum())
    n_neg = len(y_train_res) - n_pos
    spw = round(n_neg / max(n_pos, 1), 3)
    print(f"\nXGBoost scale_pos_weight = {spw}")

    # --- Define models ---
    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
            solver="liblinear",
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=random_state,
            use_label_encoder=False,
        ),
    }

    # --- Train and evaluate on validation ---
    print("\nTraining models...")
    results: Dict[str, Dict] = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        model.fit(X_train_res, y_train_res)

        # Validation metrics
        val_pred = model.predict(X_val_scaled)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        val_metrics = _metrics(y_val, val_pred, val_proba)
        print(f"  Validation metrics: {val_metrics}")

        # Train metrics (for overfitting check)
        train_pred = model.predict(X_train_res)
        train_proba = model.predict_proba(X_train_res)[:, 1]
        train_metrics = _metrics(y_train_res, train_pred, train_proba)
        print(f"  Train metrics:      {train_metrics}")

        # Save model
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved to {model_path}")

        results[name] = {
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
            "saved_at": str(model_path),
        }

    # --- Save scaler ---
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved scaler to {scaler_path}")

    # --- Save metadata ---
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_train_after_smote": int(len(X_train_res)),
        "feature_columns": FEATURE_COLS,
        "label_column": LABEL_COL,
        "class_distribution": {
            "train_before_smote": {
                "0": int((y_train == 0).sum()),
                "1": int((y_train == 1).sum()),
            },
            "train_after_smote": {
                "0": int((y_train_res == 0).sum()),
                "1": int((y_train_res == 1).sum()),
            },
            "val": {
                "0": int((y_val == 0).sum()),
                "1": int((y_val == 1).sum()),
            },
            "test": {
                "0": int((y_test == 0).sum()),
                "1": int((y_test == 1).sum()),
            },
        },
        "results": results,
        "notes": (
            "Weak supervision: labels derived from skill overlap >= 0.5 (match) "
            "or <= 0.15 (no match). SMOTE applied to training fold only. "
            "Class weights applied to all models."
        ),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")

    # --- Save test set for Phase 13 evaluation ---
    test_path = MODELS_DIR / "test_set.npz"
    np.savez(test_path, X_test=X_test_scaled, y_test=y_test.to_numpy())
    print(f"Saved test set to {test_path}")

    print("\n" + "=" * 60)
    print("DONE — Phase 12 complete")
    print("=" * 60)
    return results


if __name__ == "__main__":
    train_all()