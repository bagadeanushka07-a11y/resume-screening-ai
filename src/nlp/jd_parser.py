"""
Job Description analysis — extracts structured fields from a JD text.

Complements skill_extractor by pulling out:
    - Job title
    - Experience requirements (years + level)
    - Education requirements
    - Seniority level
    - Employment type hints

Public API:
    - parse_jd(text) -> dict with all fields
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.nlp.skill_extractor import extract_jd_skills


# --- Job title detection ------------------------------------------------------
TITLE_HEADER_RE = re.compile(
    r"^\s*(?:job\s*)?(?:title|position|role)\s*[:\-]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

# Common title keywords used as fallback when no explicit header exists
TITLE_HINT_RE = re.compile(
    r"\b(?:senior|junior|lead|staff|principal|associate|entry[- ]level)?\s*"
    r"(?:software|backend|frontend|full[- ]?stack|data|ml|machine learning|devops|cloud|"
    r"qa|test|mobile|android|ios|web|python|java|javascript|react|node|"
    r"engineer|developer|analyst|scientist|architect|manager|intern|consultant)\b",
    re.IGNORECASE,
)


# --- Experience extraction ---------------------------------------------------
# Matches "5+ years", "3-5 years", "at least 4 years", "minimum of 2 years"
YEARS_RE = re.compile(
    r"(?:at\s+least\s+|minimum\s+of\s+|minimum\s+|over\s+)?"
    r"(\d+)\s*(?:\+|\-|to)?\s*(?:\d+)?\s*\+?\s*years?",
    re.IGNORECASE,
)

EXPERIENCE_LEVELS = {
    "internship": ["intern", "internship", "trainee"],
    "entry": ["entry level", "entry-level", "junior", "graduate", "fresher"],
    "mid": ["mid level", "mid-level", "intermediate"],
    "senior": ["senior", "sr.", "sr "],
    "lead": ["lead", "team lead", "tech lead"],
    "staff": ["staff engineer", "staff"],
    "principal": ["principal"],
    "director": ["director"],
    "executive": ["vp ", "vice president", "executive", "cto", "ceo"],
}


# --- Education extraction ----------------------------------------------------
EDUCATION_PATTERNS = [
    (r"\bph\.?d\.?\b", "PhD"),
    (r"\bdoctorate\b", "PhD"),
    (r"\bmaster'?s?\b", "Master's"),
    (r"\bm\.?s\.?c?\.?\b", "Master's"),
    (r"\bm\.?tech\b", "Master's"),
    (r"\bmba\b", "MBA"),
    (r"\bbachelor'?s?\b", "Bachelor's"),
    (r"\bb\.?s\.?c?\.?\b", "Bachelor's"),
    (r"\bb\.?tech\b", "Bachelor's"),
    (r"\bb\.?e\.?\b", "Bachelor's"),
    (r"\bassociate'?s?\s+degree\b", "Associate's"),
    (r"\bdiploma\b", "Diploma"),
    (r"\bhigh\s+school\b", "High School"),
    (r"\bdegree\b", "Degree (unspecified)"),
]


# --- Employment type ---------------------------------------------------------
EMPLOYMENT_TYPES = {
    "full-time": ["full-time", "full time", "permanent"],
    "part-time": ["part-time", "part time"],
    "contract": ["contract", "contractor", "freelance"],
    "internship": ["internship", "intern "],
    "temporary": ["temporary", "temp "],
}


# --- Location / remote -------------------------------------------------------
REMOTE_PATTERNS = [
    (r"\bfully\s+remote\b", "Fully Remote"),
    (r"\bremote\b", "Remote"),
    (r"\bhybrid\b", "Hybrid"),
    (r"\bon[- ]?site\b", "On-site"),
]


# --- Public API --------------------------------------------------------------

def _detect_title(text: str) -> Optional[str]:
    """Try to find a job title. Returns None if not found."""
    # 1. Explicit header
    m = TITLE_HEADER_RE.search(text)
    if m:
        title = m.group(1).strip()
        # Clean: remove trailing punctuation and truncate
        title = re.sub(r"[.\-|•·]+$", "", title).strip()
        # Titles shouldn't be too long
        if 2 <= len(title) <= 120:
            return title

    # 2. First non-empty line that looks like a title
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 120:
            continue
        if TITLE_HINT_RE.search(line):
            return line[:120]
    return None


def _detect_experience_years(text: str) -> Optional[int]:
    """Return the minimum years of experience mentioned, or None."""
    matches = YEARS_RE.findall(text)
    if not matches:
        return None
    years = [int(y) for y in matches if y.isdigit()]
    return min(years) if years else None


def _detect_seniority(text: str) -> Optional[str]:
    text_lower = text.lower()
    for level, keywords in EXPERIENCE_LEVELS.items():
        for kw in keywords:
            if kw in text_lower:
                return level
    return None


def _detect_education(text: str) -> Optional[str]:
    """Return the highest education level mentioned, or None."""
    # Priority order: PhD > Master's > Bachelor's > Associate's > Diploma > High School
    priority = ["PhD", "Master's", "MBA", "Bachelor's", "Associate's", "Diploma", "High School", "Degree (unspecified)"]
    found = set()
    for pattern, label in EDUCATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.add(label)
    for p in priority:
        if p in found:
            return p
    return None


def _detect_employment_types(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for label, keywords in EMPLOYMENT_TYPES.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(label)
                break
    return found


def _detect_remote(text: str) -> Optional[str]:
    for pattern, label in REMOTE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return None


_KEYWORD_STOPWORDS = {
    "title", "this", "that", "these", "those", "required", "preferred",
    "strong", "familiarity", "experience", "knowledge", "must", "have",
    "plus", "nice", "bonus", "good", "great", "excellent", "we", "our",
    "you", "your", "the", "and", "for", "with", "from",
}


def _extract_keywords(text: str, max_keywords: int = 15) -> List[str]:
    """
    Extract keyword phrases: 1-3 consecutive capitalized words, filtering
    out generic English noise.
    """
    candidates = re.findall(
        r"\b(?:[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,2})\b",
        text,
    )
    seen = set()
    keywords = []
    for c in candidates:
        key = c.lower()
        if key in seen or key in _KEYWORD_STOPWORDS:
            continue
        # Filter single-word noise that is just a common first-word-of-sentence
        if " " not in c and len(c) < 4:
            continue
        seen.add(key)
        keywords.append(c)
        if len(keywords) >= max_keywords:
            break
    return keywords


def parse_jd(text: str) -> Dict:
    """
    Parse a job description into structured fields.

    Returns:
        {
            "title": Optional[str],
            "experience_years": Optional[int],
            "seniority": Optional[str],           # entry / mid / senior / ...
            "education": Optional[str],           # PhD / Master's / Bachelor's / ...
            "employment_types": List[str],        # full-time, contract, ...
            "remote": Optional[str],              # Fully Remote / Remote / Hybrid / On-site
            "keywords": List[str],
            "skills": {                           # from extract_jd_skills
                "required": [...],
                "preferred": [...],
                "unspecified": [...],
            },
        }
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "title": None,
            "experience_years": None,
            "seniority": None,
            "education": None,
            "employment_types": [],
            "remote": None,
            "keywords": [],
            "skills": {"required": [], "preferred": [], "unspecified": []},
        }

    return {
        "title": _detect_title(text),
        "experience_years": _detect_experience_years(text),
        "seniority": _detect_seniority(text),
        "education": _detect_education(text),
        "employment_types": _detect_employment_types(text),
        "remote": _detect_remote(text),
        "keywords": _extract_keywords(text),
        "skills": extract_jd_skills(text),
    }