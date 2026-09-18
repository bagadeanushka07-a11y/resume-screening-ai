"""Model Analytics page — ML performance, methodology, and explainability."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.navbar import render_footer, render_header  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.components.styles import inject_global_styles  # noqa: E402
from app.config.config import PROJECT_ROOT  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def _render_metrics_table() -> None:
    """Load evaluation_results.json and show a metrics table."""
    path = REPORTS_DIR / "evaluation_results.json"
    if not path.exists():
        st.info("Evaluation results not found. Run `python -m src.ml.evaluate` first.")
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    for name, metrics in data.get("metrics", {}).items():
        rows.append({
            "Model": display_names.get(name, name),
            "Accuracy": metrics.get("accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
            "ROC-AUC": metrics.get("roc_auc", 0),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    best = data.get("best_model")
    best_auc = data.get("best_model_roc_auc")
    if best and best_auc is not None:
        st.markdown(
            f"<div class='card' style='text-align:center;'>"
            f"<div class='card-title'>🏆 Best model by ROC-AUC</div>"
            f"<div class='card-value'>{display_names.get(best, best)}</div>"
            f"<div class='card-subtitle'>ROC-AUC = {best_auc:.3f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _show_plot(filename: str, caption: str) -> None:
    path = REPORTS_DIR / filename
    if not path.exists():
        st.info(f"Plot not found: {filename}")
        return
    st.image(str(path), caption=caption, use_column_width=True)


def _render_methodology() -> None:
    st.markdown("### 📖 Methodology")

    with st.expander("1. Dataset & weak supervision", expanded=False):
        st.markdown(
            """
            - **Resumes**: 2,483 real resumes from a public Kaggle dataset.
            - **Job descriptions**: 5,000 LinkedIn job postings.
            - **Pairs**: 3,000 sampled (resume, JD) combinations.
            - **Weak labels**: each pair was labeled `match` or `no match`
              based on TF-IDF cosine similarity between resume and JD text.
              Thresholds were **auto-calibrated from the observed similarity
              distribution** (top 30% → match; bottom 40% → no match;
              middle 30% → skipped as ambiguous) rather than hard-coded,
              so the split adapts to the actual data.
            - **2,101 labeled pairs** remained for training.

            This is **weak supervision** — the labels come from an automated
            heuristic, not human annotation. It scales to thousands of
            examples but introduces label noise, which is why the model's
            ROC-AUC is ~0.70 rather than 0.95+.
            """
        )

    with st.expander("2. Feature engineering", expanded=False):
        st.markdown(
            """
            Five features per pair:

            | Feature | Description |
            |---------|-------------|
            | `skill_match_score` | Overlap of resume skills vs. all JD skills |
            | `skill_match_required` | Overlap vs. JD required skills only |
            | `experience_match` | Resume years of experience vs. JD requirement |
            | `education_match` | Highest resume degree vs. JD requirement |
            | `category_match` | Resume category vs. JD title keywords |

            Features were engineered to be **independent of the label** —
            the label comes from TF-IDF, the features do not use TF-IDF.
            This prevents data leakage.
            """
        )

    with st.expander("3. Model training & class imbalance", expanded=False):
        st.markdown(
            """
            - **Split**: 70% train / 15% validation / 15% test (stratified).
            - **Imbalance**: ~57% no-match / 43% match.
            - **Fix**: SMOTE applied on the training fold only (never
              validation or test).
            - **Models compared**: Logistic Regression, Random Forest, XGBoost.
            - **Regularization**: class weights (`balanced`) on all models,
              XGBoost uses `scale_pos_weight`.
            """
        )

    with st.expander("4. Why Logistic Regression won", expanded=False):
        st.markdown(
            """
            The tree-based models (Random Forest, XGBoost) showed large
            train-validation gaps (~30 AUC points), indicating overfitting
            on our small noisy dataset. Logistic Regression generalized
            better *and* is fully interpretable via SHAP — critical for
            a hiring-adjacent application.
            """
        )

    with st.expander("5. Data leakage we caught", expanded=False):
        st.markdown(
            """
            Early models scored **100% accuracy on every metric** — a clear
            sign of data leakage. The cause: `length_ratio` (a length-similarity
            feature) correlated strongly with TF-IDF similarity, which was
            also our label source. Removing it dropped ROC-AUC from 0.73 to
            0.67, but made the model honest and its SHAP explanations trustworthy.
            """
        )

    with st.expander("6. Explainability (SHAP)", expanded=False):
        st.markdown(
            """
            Logistic Regression is explained with `shap.LinearExplainer`:

            - **Global**: which features matter most on average.
            - **Local**: why a single prediction came out the way it did.

            SHAP surfaced the leakage issue in step 5 — a feature with
            no real-world meaning (length ratio) dominating the model.
            """
        )


def _render_tfidf_vs_embeddings() -> None:
    st.markdown("### 🔍 TF-IDF vs. Sentence Embeddings")
    st.markdown(
        "<div class='muted mb-2'>"
        "Two text-similarity methods used in the match score. "
        "TF-IDF captures exact word overlap; embeddings capture meaning."
        "</div>",
        unsafe_allow_html=True,
    )

    example = {
        "Pair A (paraphrase)": (
            "Experienced coder who writes software and debugs programs",
            "Senior programmer building applications and fixing bugs",
        ),
        "Pair B (identical)": (
            "Python developer with Django",
            "Python developer with Django",
        ),
        "Pair C (unrelated)": (
            "Python developer",
            "Chef with culinary training",
        ),
    }

    try:
        from src.nlp.tfidf_matcher import compute_similarity as tfidf_sim
        from src.nlp.embeddings import compute_semantic_similarity as emb_sim

        rows = []
        for label, (a, b) in example.items():
            rows.append({
                "Pair": label,
                "TF-IDF": round(tfidf_sim(a, b), 3),
                "Embeddings": round(emb_sim(a, b), 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='card'><p class='muted' style='margin:0;'>"
            "Notice: for paraphrases that share few exact words, "
            "embeddings capture the semantic match while TF-IDF misses it. "
            "This is why we average the two in the final score."
            "</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not compute TF-IDF/embedding demo: {e}")


def main() -> None:
    inject_global_styles()
    render_sidebar()
    render_header(subtitle="How the ML model works and how well it performs")

    st.markdown(
        """
        <div class='card'>
            <div class='card-title'>📈 Model Analytics</div>
            <p class='muted mb-0'>
                This page shows the ML pipeline's actual performance on a
                held-out test set, our evaluation methodology, and the
                explainability approach. All numbers come from a real
                training run — no inflated metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Model comparison")
    _render_metrics_table()

    st.markdown("### 📉 Confusion matrices")
    _show_plot("confusion_matrices.png", "Confusion matrices on the test set")

    st.markdown("### 📈 ROC curves")
    _show_plot("roc_curves.png", "ROC curves for all three models")

    st.markdown("### 🏆 Metric comparison")
    _show_plot("model_comparison.png", "All metrics side-by-side")

    _render_tfidf_vs_embeddings()

    _render_methodology()

    render_footer()


main()