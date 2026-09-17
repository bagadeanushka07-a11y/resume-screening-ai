"""
Skill extraction from resume / job description text.

Uses the Phase 7 skill knowledge base (ESCO + curated) and adds:
    - Section awareness (skills in a "Skills" section get higher confidence)
    - Frequency counting
    - Category grouping
    - Semantic deduplication (React vs React.js → one entry)
    - JD-specific required/preferred/unspecified classification

Public API:
    - extract_skills(text)            -> list of skill dicts with confidence
    - group_by_category(skills)       -> dict {category: [skills]}
    - extract_resume_skills(text)     -> resume-focused extraction
    - extract_jd_skills(text)         -> JD-focused extraction (required vs preferred)
"""

from __future__ import annotations

import re
from typing import Dict, List

from src.nlp.skill_kb import (
    search as kb_search,
    get_category,
    normalize,
)


# --- Section detection (resume) -----------------------------------------------
SKILL_SECTION_HEADERS = [
    r"\bskills?\b", r"\btechnical skills?\b", r"\btechnical proficienc(?:y|ies)\b",
    r"\btechnologies\b", r"\btech stack\b", r"\bcore competencies\b",
    r"\bkey skills?\b", r"\bcompetencies\b", r"\bexpertise\b",
    r"\btools?\b", r"\blanguages?\b", r"\bframeworks?\b",
]

_SKILL_SECTION_RE = re.compile(
    "|".join(SKILL_SECTION_HEADERS),
    re.IGNORECASE,
)


# --- JD section detection -----------------------------------------------------
_REQUIRED_HEADERS = [
    r"\brequired\b", r"\bmust have\b", r"\bmust-have\b",
    r"\bminimum qualifications?\b", r"\brequirements?\b",
    r"\bessential\b",
]

_PREFERRED_HEADERS = [
    r"\bpreferred\b", r"\bnice to have\b", r"\bnice-to-have\b",
    r"\bbonus\b", r"\bplus\b", r"\bdesirable\b", r"\bgood to have\b",
]

_REQUIRED_RE = re.compile("|".join(_REQUIRED_HEADERS), re.IGNORECASE)
_PREFERRED_RE = re.compile("|".join(_PREFERRED_HEADERS), re.IGNORECASE)


# --- Skill aliases for semantic deduplication --------------------------------
SKILL_ALIASES = {
    "react.js": "React",
    "reactjs": "React",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "expressjs": "Express.js",
    "vuejs": "Vue.js",
    "nextjs": "Next.js",
    "angularjs": "Angular",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "postgres": "PostgreSQL",
    "mongo": "MongoDB",
    "k8s": "Kubernetes",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
}


def _canonicalize(skill: str) -> str:
    """Map aliases to canonical form."""
    if not isinstance(skill, str):
        return ""
    return SKILL_ALIASES.get(skill.lower(), skill)


def _dedupe_by_canonical(hits: List[Dict]) -> List[Dict]:
    """
    Given raw hits (may contain both "React" and "React.js"), keep only
    one entry per canonical form. Prefers the higher-confidence entry.
    """
    by_canonical: Dict[str, Dict] = {}
    for h in hits:
        canonical = _canonicalize(h["skill"])
        key = canonical.lower()
        existing = by_canonical.get(key)
        if existing is None or h["confidence"] > existing["confidence"]:
            new_h = dict(h)
            new_h["skill"] = canonical
            by_canonical[key] = new_h
    return list(by_canonical.values())


# --- Section helpers ---------------------------------------------------------

def _find_skill_sections(text: str, window: int = 400) -> List[tuple]:
    """Ranges (start, end) where resume skill sections likely occur."""
    if not isinstance(text, str):
        return []
    ranges = []
    for m in _SKILL_SECTION_RE.finditer(text):
        start = m.start()
        end = min(len(text), start + window)
        ranges.append((start, end))
    return ranges


def _is_in_skill_section(pos: int, ranges: List[tuple]) -> bool:
    return any(s <= pos <= e for s, e in ranges)


def _find_section_ranges(text: str, regex: re.Pattern, max_window: int = 600) -> List[tuple]:
    """
    Find ranges for a JD section. A section starts at a header match and ends
    at the NEXT section header (required OR preferred), or at max_window chars,
    whichever comes first.
    """
    if not isinstance(text, str):
        return []

    # Find all section headers so we know where boundaries are
    all_headers = (
        list(_REQUIRED_RE.finditer(text))
        + list(_PREFERRED_RE.finditer(text))
    )
    header_positions = sorted(m.start() for m in all_headers)

    ranges = []
    for m in regex.finditer(text):
        start = m.start()
        end = min(len(text), start + max_window)
        # Shrink end to the next header after this one (with small offset)
        for pos in header_positions:
            if pos > start + 5:
                end = min(end, pos)
                break
        ranges.append((start, end))
    return ranges


def _section_of(pos: int, required_ranges: List[tuple], preferred_ranges: List[tuple]) -> str:
    """Classify a position as 'required', 'preferred', or 'unspecified'."""
    if any(s <= pos <= e for s, e in required_ranges):
        return "required"
    if any(s <= pos <= e for s, e in preferred_ranges):
        return "preferred"
    return "unspecified"


# --- Main extraction ---------------------------------------------------------

def extract_skills(text: str) -> List[Dict]:
    """
    Extract skills from text. Returns list of dicts:
        {
            "skill": canonical skill name,
            "category": Programming / Framework / ... / Other,
            "confidence": 0.0–1.0,
            "frequency": how many times it appeared,
            "in_skill_section": True if at least one occurrence is in a skills section,
        }
    """
    if not isinstance(text, str) or not text.strip():
        return []

    hits = kb_search(text)
    if not hits:
        return []

    text_lower = text.lower()
    sections = _find_skill_sections(text_lower)

    results = []
    for hit in hits:
        skill = hit["skill"]
        category = hit["category"]

        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(skill.lower())}(?![A-Za-z0-9])"
        )
        matches = list(pattern.finditer(text_lower))
        freq = len(matches)
        in_sec = any(_is_in_skill_section(m.start(), sections) for m in matches)

        # Confidence: base on frequency + boost for being in a skills section + known category
        freq_score = min(1.0, 0.3 + 0.15 * (freq - 1))
        section_boost = 0.25 if in_sec else 0.0
        category_boost = 0.15 if category != "Other" else 0.0
        confidence = round(min(1.0, freq_score + section_boost + category_boost), 3)

        results.append({
            "skill": skill,
            "category": category,
            "confidence": confidence,
            "frequency": freq,
            "in_skill_section": in_sec,
        })

    # Deduplicate semantically-identical skills (React.js == React)
    results = _dedupe_by_canonical(results)

    # Sort by confidence desc, then category, then skill name
    results.sort(key=lambda r: (-r["confidence"], r["category"], r["skill"]))
    return results


def group_by_category(skills: List[Dict]) -> Dict[str, List[str]]:
    """
    Return {"Programming": ["Python", "Java"], "Framework": ["Django"], ...}.
    Only skills with confidence >= 0.4 are included.
    """
    grouped: Dict[str, List[str]] = {}
    for s in skills:
        if s["confidence"] < 0.4:
            continue
        grouped.setdefault(s["category"], []).append(s["skill"])
    for cat in grouped:
        grouped[cat] = sorted(set(grouped[cat]))
    return grouped


# --- Resume / JD entry points ------------------------------------------------

def extract_resume_skills(text: str) -> List[Dict]:
    """Resume-focused extraction (same as extract_skills for now)."""
    return extract_skills(text)


def extract_jd_skills(text: str) -> Dict[str, List[Dict]]:
    """
    Extract JD skills, split into:
        {"required": [...], "preferred": [...], "unspecified": [...]}

    Each skill dict has the same shape as extract_skills output, plus a
    "role" field indicating required/preferred/unspecified.
    """
    empty = {"required": [], "preferred": [], "unspecified": []}
    if not isinstance(text, str) or not text.strip():
        return empty

    hits = kb_search(text)
    if not hits:
        return empty

    text_lower = text.lower()
    required_ranges = _find_section_ranges(text_lower, _REQUIRED_RE)
    preferred_ranges = _find_section_ranges(text_lower, _PREFERRED_RE)

    result: Dict[str, List[Dict]] = {"required": [], "preferred": [], "unspecified": []}

    for hit in hits:
        skill = hit["skill"]
        category = hit["category"]

        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(skill.lower())}(?![A-Za-z0-9])"
        )
        matches = list(pattern.finditer(text_lower))
        if not matches:
            continue
        freq = len(matches)

        roles = [_section_of(m.start(), required_ranges, preferred_ranges) for m in matches]
        if "required" in roles:
            role = "required"
        elif "preferred" in roles:
            role = "preferred"
        else:
            role = "unspecified"

        freq_score = min(1.0, 0.3 + 0.15 * (freq - 1))
        category_boost = 0.15 if category != "Other" else 0.0
        role_boost = 0.2 if role == "required" else (0.1 if role == "preferred" else 0.0)
        confidence = round(min(1.0, freq_score + category_boost + role_boost), 3)

        result[role].append({
            "skill": skill,
            "category": category,
            "confidence": confidence,
            "frequency": freq,
            "role": role,
        })

    # Deduplicate each role bucket and sort
    for role in result:
        result[role] = _dedupe_by_canonical(result[role])
        result[role].sort(key=lambda r: (-r["confidence"], r["category"], r["skill"]))

    return result