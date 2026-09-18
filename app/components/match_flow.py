"""
Orchestrates the full match pipeline for the UI.

Takes resume text + jd text + metadata, and returns a full results dict.
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from app.components.scoring import (
    compute_match_score_with_resume_education,
    detect_resume_education,
    detect_resume_years,
)


def run_match_pipeline(
    resume_text: str,
    jd_text: str,
    resume_category: Optional[str] = None,
    resume_skills: Optional[list] = None,
) -> Dict:
    """
    Run the full pipeline: JD parse -> skills -> gap -> tfidf -> embeddings -> ML -> SHAP -> recommendations.

    Never raises; on any failure returns an empty-ish result with an error string.
    """
    result: Dict = {"error": None}
    if not isinstance(resume_text, str) or not resume_text.strip():
        result["error"] = "Resume text is missing."
        return result
    if not isinstance(jd_text, str) or not jd_text.strip():
        result["error"] = "Job description is missing."
        return result

    # --- Parse JD ---
    try:
        from src.nlp.jd_parser import parse_jd
        jd_parsed = parse_jd(jd_text)
        result["jd_parsed"] = jd_parsed
    except Exception as e:
        result["error"] = f"Failed to parse job description: {e}"
        return result

    # --- Resume skills ---
    if resume_skills is None:
        try:
            from src.nlp.skill_extractor import extract_resume_skills
            resume_skills = extract_resume_skills(resume_text)
        except Exception:
            resume_skills = []
    result["resume_skills"] = resume_skills

    # --- Skill gap ---
    try:
        from src.matching.skill_gap import analyze_gap
        jd_skills_for_gap = jd_parsed.get("skills", {})
        gap = analyze_gap(resume_skills, jd_skills_for_gap)
        result["gap"] = gap
    except Exception as e:
        result["error"] = f"Skill gap analysis failed: {e}"
        return result

    # --- TF-IDF ---
    try:
        from src.nlp.tfidf_matcher import compute_similarity
        tfidf = compute_similarity(resume_text, jd_text)
    except Exception:
        tfidf = 0.0
    result["tfidf"] = tfidf

    # --- Embeddings ---
    try:
        from src.nlp.embeddings import compute_semantic_similarity
        embeddings = compute_semantic_similarity(resume_text, jd_text)
    except Exception:
        embeddings = 0.0
    result["embeddings"] = embeddings

    # --- Resume-side metadata ---
    resume_years = detect_resume_years(resume_text)
    resume_edu = detect_resume_education(resume_text)
    result["resume_years"] = resume_years
    result["resume_education"] = resume_edu

    # --- Weighted score ---
    try:
        score_result = compute_match_score_with_resume_education(
            resume_text=resume_text,
            jd_text=jd_text,
            resume_category=resume_category,
            jd_parsed=jd_parsed,
            gap=gap,
            tfidf_sim=tfidf,
            embeddings_sim=embeddings,
            resume_years=resume_years,
            resume_education=resume_edu,
        )
        result["score"] = score_result
    except Exception as e:
        result["error"] = f"Scoring failed: {e}"
        return result

    # --- ML model prediction ---
    try:
        ml = _run_ml_prediction(
            resume_skills=resume_skills,
            jd_parsed=jd_parsed,
            gap=gap,
            tfidf=tfidf,
            resume_years=resume_years,
            resume_edu=resume_edu,
            resume_category=resume_category,
        )
        result["ml"] = ml
    except Exception as e:
        result["ml"] = {"error": str(e), "probability": None}

    # --- Recommendations ---
    try:
        from src.recommendations.recommendation_engine import generate_recommendations
        recs = generate_recommendations(gap)
        result["recommendations"] = recs
    except Exception:
        result["recommendations"] = []

    return result


# --- ML prediction (feature vector -> model) ---------------------------------

def _compute_skill_overlap(resume_skills: list, jd_skills: list) -> float:
    r = {s.get("skill", "").lower() if isinstance(s, dict) else str(s).lower()
         for s in resume_skills if s}
    j = {s.get("skill", "").lower() if isinstance(s, dict) else str(s).lower()
         for s in jd_skills if s}
    return len(r & j) / len(j) if j else 0.0


def _run_ml_prediction(
    resume_skills, jd_parsed, gap, tfidf, resume_years, resume_edu, resume_category,
) -> Dict:
    import joblib
    import numpy as np
    from pathlib import Path

    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    model_path = models_dir / "logistic_regression.joblib"
    scaler_path = models_dir / "scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        return {"error": "Trained model not found.", "probability": None}

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Build the feature dict matching our 5-column FEATURE_COLS
    jd_req = [s["skill"] for s in jd_parsed.get("skills", {}).get("required", [])]
    jd_pref = [s["skill"] for s in jd_parsed.get("skills", {}).get("preferred", [])]
    jd_all = jd_req + jd_pref

    skill_match = _compute_skill_overlap(resume_skills, jd_all)
    skill_req = _compute_skill_overlap(resume_skills, jd_req)

    # Experience match
    jd_years = jd_parsed.get("experience_years")
    if jd_years and resume_years:
        exp_match = 1.0 if resume_years >= jd_years else resume_years / jd_years
    elif not jd_years:
        exp_match = 0.5
    else:
        exp_match = 0.3

    # Education match
    edu_rank = {"PhD": 6, "Master's": 5, "MBA": 5, "Bachelor's": 4,
                "Degree (unspecified)": 4, "Associate's": 3, "Diploma": 2, "High School": 1}
    jd_edu = jd_parsed.get("education")
    if jd_edu and resume_edu:
        j_rank = edu_rank.get(jd_edu, 0)
        r_rank = edu_rank.get(resume_edu, 0)
        edu_match = 1.0 if r_rank >= j_rank else (r_rank / j_rank if j_rank else 0.0)
    elif not jd_edu:
        edu_match = 0.5
    else:
        edu_match = 0.3

    # Category match
    from src.ml.feature_engineering import CATEGORY_KEYWORDS
    jd_title = jd_parsed.get("title") or ""
    kws = CATEGORY_KEYWORDS.get((resume_category or "").upper(), [])
    cat_match = 0.0
    if resume_category and jd_title:
        if kws:
            cat_match = 1.0 if any(kw in jd_title.lower() for kw in kws) else 0.5
        else:
            cat_match = 0.5

    feature_dict = {
        "skill_match_score": round(skill_match, 4),
        "skill_match_required": round(skill_req, 4),
        "experience_match": round(exp_match, 4),
        "education_match": round(edu_match, 4),
        "category_match": round(cat_match, 4),
    }

    # Build vector in the SAME ORDER as FEATURE_COLS
    cols = [
        "skill_match_score",
        "skill_match_required",
        "experience_match",
        "education_match",
        "category_match",
    ]
    vector = np.array([[feature_dict[c] for c in cols]])
    vector_scaled = scaler.transform(vector)
    proba = float(model.predict_proba(vector_scaled)[0][1])
    pred = int(model.predict(vector_scaled)[0])

    return {
        "probability": round(proba, 4),
        "prediction": pred,
        "features": feature_dict,
    }


def compute_shap_local(features_dict: Dict[str, float]) -> Dict:
    """Get SHAP local explanation for the given feature dict."""
    try:
        from src.ml.explain import explain_prediction
        return explain_prediction(features_dict)
    except Exception as e:
        return {"error": str(e), "contributions": []}


def persist_match_to_db(
    user_id: int,
    resume_filename: str,
    resume_text: str,
    jd_text: str,
    jd_parsed: dict,
    score_result: dict,
) -> Optional[int]:
    """
    Save resume, job, and match to MySQL. Best-effort — returns match_id or None.
    """
    if not user_id or user_id == 0:
        return None
    try:
        from src.database import queries as q
        from src.database.connection import is_available

        if not is_available():
            return None

        # Save resume
        resume_id = q.create_resume(user_id, resume_filename or "uploaded.pdf", resume_text)

        # Save job
        title = jd_parsed.get("title") or "Untitled role"
        company = "Unknown"
        job_id = q.create_job(title, company, jd_text)

        # Save match
        components = score_result.get("components", {})
        match_id = q.save_match_result(
            resume_id=resume_id,
            job_id=job_id,
            skill_score=components.get("skill_match", 0.0),
            semantic_score=components.get("semantic_similarity", 0.0),
            experience_score=components.get("experience_match", 0.0),
            education_score=components.get("education_match", 0.0),
            project_score=components.get("category_match", 0.0),
            overall_score=score_result.get("overall_score", 0.0),
        )
        return match_id
    except Exception:
        return None