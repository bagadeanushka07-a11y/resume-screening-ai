"""
Transparent match scoring.

Combines multiple signals into a single weighted score, and returns a
full breakdown that the UI can display.

WEIGHTS are project-defined (see master prompt Section 8) — they are NOT
an industry standard hiring formula.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# --- Weights (single source of truth) ----------------------------------------
WEIGHTS = {
    "skill_match": 0.40,          # required + preferred skill overlap
    "semantic_similarity": 0.25,  # TF-IDF + Sentence Transformers average
    "experience_match": 0.15,     # resume years vs JD years
    "education_match": 0.10,      # resume degree vs JD degree
    "category_match": 0.10,       # resume category vs JD title
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# --- Component computations --------------------------------------------------

def _safe(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _skill_score(gap: Dict) -> float:
    """
    Convert the skill gap output into a [0,1] score.
    The gap module already returns a weighted match_percentage in [0,100].
    """
    return _safe(gap.get("match_percentage", 0.0)) / 100.0


def _semantic_score(tfidf: float, embeddings: float) -> float:
    """Average of TF-IDF and embeddings similarity."""
    return round((_safe(tfidf) + _safe(embeddings)) / 2.0, 4)


def _experience_score(resume_years: Optional[int], jd_years: Optional[int]) -> float:
    if not jd_years:
        return 0.5
    if not resume_years:
        return 0.3
    if resume_years >= jd_years:
        return 1.0
    return round(resume_years / jd_years, 4)


_EDU_RANK = {
    "High School": 1, "Diploma": 2, "Associate's": 3,
    "Bachelor's": 4, "Degree (unspecified)": 4,
    "Master's": 5, "MBA": 5, "PhD": 6,
}


def _education_score(resume_edu: Optional[str], jd_edu: Optional[str]) -> float:
    if not jd_edu:
        return 0.5
    if not resume_edu:
        return 0.3
    r = _EDU_RANK.get(resume_edu, 0)
    j = _EDU_RANK.get(jd_edu, 0)
    if j == 0:
        return 0.5
    if r >= j:
        return 1.0
    return round(r / j, 4)


def _category_score(resume_category: Optional[str], jd_title: Optional[str]) -> float:
    """
    Reuse the exact logic from feature_engineering.CATEGORY_KEYWORDS
    so we don't have two definitions.
    """
    if not resume_category or not jd_title:
        return 0.0
    try:
        from src.ml.feature_engineering import CATEGORY_KEYWORDS
        kws = CATEGORY_KEYWORDS.get(resume_category.upper(), [])
        if not kws:
            return 0.5
        jd_lower = jd_title.lower()
        return 1.0 if any(kw in jd_lower for kw in kws) else 0.5
    except Exception:
        return 0.5


# --- Public API --------------------------------------------------------------

def compute_match_score(
    resume_text: str,
    jd_text: str,
    resume_category: Optional[str],
    jd_parsed: Dict,
    gap: Dict,
    tfidf_sim: float,
    embeddings_sim: float,
    resume_years: Optional[int],
) -> Dict:
    """
    Compute the final weighted match score with a full breakdown.

    Returns:
        {
            "overall_score": 0.0–1.0,
            "overall_percent": 0–100,
            "weights": WEIGHTS,
            "components": {
                "skill_match": 0.0–1.0,
                "semantic_similarity": 0.0–1.0,
                "experience_match": 0.0–1.0,
                "education_match": 0.0–1.0,
                "category_match": 0.0–1.0,
            },
            "weighted": {
                "skill_match": 0.0–1.0,   (component * weight)
                ...
            },
            "notes": str,
        }
    """
    components = {
        "skill_match": _skill_score(gap),
        "semantic_similarity": _semantic_score(tfidf_sim, embeddings_sim),
        "experience_match": _experience_score(resume_years, jd_parsed.get("experience_years")),
        "education_match": _education_score(None, jd_parsed.get("education")),  # resume edu filled by caller
        "category_match": _category_score(resume_category, jd_parsed.get("title")),
    }

    weighted = {k: round(components[k] * WEIGHTS[k], 4) for k in WEIGHTS}
    overall = round(sum(weighted.values()), 4)
    overall = max(0.0, min(1.0, overall))

    return {
        "overall_score": overall,
        "overall_percent": round(overall * 100, 1),
        "weights": dict(WEIGHTS),
        "components": components,
        "weighted": weighted,
        "notes": (
            "This is a project-defined scoring methodology that combines "
            "multiple signals. It is NOT an industry-standard hiring formula "
            "and does NOT represent actual hiring probability."
        ),
    }


def compute_match_score_with_resume_education(
    resume_text: str,
    jd_text: str,
    resume_category: Optional[str],
    jd_parsed: Dict,
    gap: Dict,
    tfidf_sim: float,
    embeddings_sim: float,
    resume_years: Optional[int],
    resume_education: Optional[str],
) -> Dict:
    """
    Full version that also takes the resume's detected education.
    """
    components = {
        "skill_match": _skill_score(gap),
        "semantic_similarity": _semantic_score(tfidf_sim, embeddings_sim),
        "experience_match": _experience_score(resume_years, jd_parsed.get("experience_years")),
        "education_match": _education_score(resume_education, jd_parsed.get("education")),
        "category_match": _category_score(resume_category, jd_parsed.get("title")),
    }

    weighted = {k: round(components[k] * WEIGHTS[k], 4) for k in WEIGHTS}
    overall = round(sum(weighted.values()), 4)
    overall = max(0.0, min(1.0, overall))

    return {
        "overall_score": overall,
        "overall_percent": round(overall * 100, 1),
        "weights": dict(WEIGHTS),
        "components": components,
        "weighted": weighted,
        "notes": (
            "This is a project-defined scoring methodology that combines "
            "multiple signals. It is NOT an industry-standard hiring formula "
            "and does NOT represent actual hiring probability."
        ),
    }


def detect_resume_education(resume_text: str) -> Optional[str]:
    """Find highest education level mentioned in the resume."""
    if not isinstance(resume_text, str):
        return None
    found = None
    text_lower = resume_text.lower()
    for edu in ["phd", "ph.d", "doctorate", "master", "mba", "bachelor", "b.tech", "b.e", "b.sc"]:
        if edu in text_lower:
            if edu in ("phd", "ph.d", "doctorate"):
                return "PhD"
            if edu in ("master", "mba"):
                found = found or "Master's"
            if edu in ("bachelor", "b.tech", "b.e", "b.sc"):
                found = found or "Bachelor's"
    return found


def detect_resume_years(resume_text: str) -> Optional[int]:
    """Find max years of experience mentioned in the resume."""
    import re
    if not isinstance(resume_text, str):
        return None
    matches = re.findall(r"(\d+)\s*\+?\s*years?", resume_text, re.IGNORECASE)
    if not matches:
        return None
    try:
        return max(int(y) for y in matches)
    except (ValueError, TypeError):
        return None