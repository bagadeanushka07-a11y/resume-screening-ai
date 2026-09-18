"""
Feature engineering for ML training.

Given a (resume_text, jd_text) pair, extract a fixed-length feature vector:
    - skill_match_score        : 0-1, overlap of required+preferred skills
    - skill_match_required     : 0-1, overlap of only required skills
    - tfidf_similarity         : 0-1, cosine similarity from Phase 10
    - experience_match         : 0-1, resume years vs JD required years
    - education_match          : 0-1, resume degree >= JD required degree
    - category_match           : 0-1, resume category vs JD title keywords
    

Speed note:
    extract_features accepts optional pre-computed skills to avoid the
    expensive ESCO extraction on every pair. Pass them in when you already
    have them — this gives a 50-100x speedup on large datasets.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.nlp.skill_extractor import extract_resume_skills
from src.nlp.jd_parser import parse_jd
from src.nlp.tfidf_matcher import compute_similarity


EDU_RANK = {
    "High School": 1,
    "Diploma": 2,
    "Associate's": 3,
    "Bachelor's": 4,
    "Degree (unspecified)": 4,
    "Master's": 5,
    "MBA": 5,
    "PhD": 6,
}


_RESUME_YEARS_RE = re.compile(r"(\d+)\s*\+?\s*years?", re.IGNORECASE)


def _extract_resume_years(resume_text: str) -> Optional[int]:
    if not isinstance(resume_text, str):
        return None
    matches = _RESUME_YEARS_RE.findall(resume_text)
    if not matches:
        return None
    try:
        return max(int(y) for y in matches)
    except (ValueError, TypeError):
        return None


CATEGORY_KEYWORDS = {
    "INFORMATION-TECHNOLOGY": [
        "engineer", "developer", "software", "it ", "tech",
        "python", "java", "javascript", "backend", "frontend",
        "full stack", "devops", "data", "machine learning",
        "ml", "ai ", "scientist", "analyst",
    ],
    "ENGINEERING": ["engineer", "civil", "mechanical", "electrical", "hardware"],
    "DESIGNER": ["designer", "design", "ux", "ui", "graphic", "creative"],
    "HR": ["hr", "human resources", "recruiter", "talent"],
    "TEACHER": ["teacher", "instructor", "educator", "professor", "tutor"],
    "ADVOCATE": ["advocate", "lawyer", "attorney", "legal", "counsel"],
    "BUSINESS-DEVELOPMENT": ["business", "development", "sales", "bd "],
    "SALES": ["sales", "account executive", "account manager"],
    "CONSULTANT": ["consultant", "consulting", "advisor"],
    "FINANCE": ["finance", "financial", "accountant", "accounting"],
    "HEALTHCARE": ["healthcare", "medical", "nurse", "doctor", "clinical"],
    "CHEF": ["chef", "cook", "culinary", "kitchen"],
    "ARTS": ["artist", "art ", "creative", "musician"],
    "AUTOMOBILE": ["automobile", "automotive", "mechanic", "vehicle"],
    "AVIATION": ["aviation", "pilot", "airline", "flight"],
    "CONSTRUCTION": ["construction", "site", "builder", "civil"],
    "PUBLIC-RELATIONS": ["pr ", "public relations", "communications", "media"],
    "MARKETING": ["marketing", "seo", "brand", "growth"],
    "DIGITAL-MEDIA": ["digital", "social media", "content"],
    "APPAREL": ["apparel", "fashion", "clothing", "textile"],
    "BANKING": ["bank", "banking", "loan", "credit"],
    "AGRICULTURE": ["agriculture", "farm", "crop"],
    "BPO": ["bpo", "call center", "customer service", "support"],
    "FITNESS": ["fitness", "trainer", "gym", "wellness"],
}


def _category_match(resume_category: Optional[str], jd_title: Optional[str]) -> float:
    if not resume_category or not jd_title:
        return 0.0
    keywords = CATEGORY_KEYWORDS.get(resume_category.upper(), [])
    if not keywords:
        return 0.5
    jd_lower = jd_title.lower()
    for kw in keywords:
        if kw in jd_lower:
            return 1.0
    return 0.5


def _skill_overlap(resume_skills: List[str], jd_skills: List[str]) -> float:
    r = {s.lower().strip() for s in resume_skills if s}
    j = {s.lower().strip() for s in jd_skills if s}
    if not j:
        return 0.0
    return len(r & j) / len(j)


def _length_ratio(a: str, b: str) -> float:
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    la, lb = len(a.split()), len(b.split())
    if la == 0 or lb == 0:
        return 0.0
    return 1.0 - abs(la - lb) / max(la, lb)


def extract_features(
    resume_text: str,
    jd_text: str,
    resume_category: Optional[str] = None,
    resume_skills: Optional[List[str]] = None,
    jd_skills: Optional[List[str]] = None,
    jd_parsed: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Extract a fixed-length feature vector from a (resume, JD) pair.

    Optional pre-computed arguments (BIG speedup):
        resume_skills : list of skill names already extracted from the resume
        jd_skills     : list of skill names already extracted from the JD
        jd_parsed     : output of parse_jd() for the JD (used for experience,
                        education, and title)

    If omitted, features that need them will fall back to slow extraction.
    """
    features: Dict[str, float] = {
        "skill_match_score": 0.0,
        "skill_match_required": 0.0,
        
        "experience_match": 0.0,
        "education_match": 0.0,
        "category_match": 0.0,
       
    }

    if not isinstance(resume_text, str) or not isinstance(jd_text, str):
        return features
    if not resume_text.strip() or not jd_text.strip():
        return features

    # --- Get JD parse (fast path or slow path) ---
    if jd_parsed is None:
        try:
            jd_parsed = parse_jd(jd_text)
        except Exception:
            jd_parsed = {"skills": {"required": [], "preferred": []},
                         "experience_years": None, "education": None, "title": None}

    # --- Skill match ---
    # If skills not provided, fall back to slow extraction
    if resume_skills is None:
        try:
            resume_skills = [s["skill"] for s in extract_resume_skills(resume_text)]
        except Exception:
            resume_skills = []

    required_names = [s["skill"] for s in jd_parsed.get("skills", {}).get("required", [])]
    preferred_names = [s["skill"] for s in jd_parsed.get("skills", {}).get("preferred", [])]
    all_jd_names = required_names + preferred_names

    if jd_skills is None:
        jd_skills = all_jd_names

    features["skill_match_score"] = round(
        _skill_overlap(resume_skills, all_jd_names), 4
    )
    features["skill_match_required"] = round(
        _skill_overlap(resume_skills, required_names), 4
    )

  

    # --- Experience match ---
    try:
        jd_years = jd_parsed.get("experience_years")
        resume_years = _extract_resume_years(resume_text)
        if jd_years and resume_years:
            if resume_years >= jd_years:
                features["experience_match"] = 1.0
            else:
                features["experience_match"] = round(resume_years / jd_years, 4)
        elif not jd_years:
            features["experience_match"] = 0.5
        else:
            features["experience_match"] = 0.3
    except Exception:
        pass

    # --- Education match ---
    try:
        jd_edu = jd_parsed.get("education")
        resume_edu = None
        for edu_name, _rank in EDU_RANK.items():
            if edu_name.lower() in resume_text.lower():
                if resume_edu is None or _rank > EDU_RANK.get(resume_edu, 0):
                    resume_edu = edu_name
        if jd_edu and resume_edu:
            jd_rank = EDU_RANK.get(jd_edu, 0)
            r_rank = EDU_RANK.get(resume_edu, 0)
            if r_rank >= jd_rank:
                features["education_match"] = 1.0
            else:
                features["education_match"] = round(r_rank / jd_rank, 4) if jd_rank else 0.0
        elif not jd_edu:
            features["education_match"] = 0.5
        else:
            features["education_match"] = 0.3
    except Exception:
        pass

    # --- Category match ---
    try:
        features["category_match"] = round(
            _category_match(resume_category, jd_parsed.get("title")), 4
        )
    except Exception:
        pass

    # --- Length ratio ---
   

    return features