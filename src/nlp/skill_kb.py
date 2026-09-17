"""
Skill Knowledge Base — canonical source of skill names, variants, and categories.

Loads:
    - data/processed/esco_skills_clean.csv   (ESCO preferred labels + alt labels)
    - data/processed/esco_alias_map.json     (variant -> canonical mapping)
    - data/external/curated_skills.json      (our curated list, provides CATEGORIES)

Provides:
    - normalize(skill)              -> canonical ESCO name (or original if unknown)
    - get_category(skill)           -> Programming / Framework / ... / Soft Skills / Other
    - get_skills_by_category(cat)   -> list of canonical skill names
    - all_categories()              -> list of category names
    - all_canonical_skills()        -> set of ALL known canonical skills (lowercased)
    - search(text)                  -> list of dicts of skills found in text

Design notes:
    - ESCO gives us scale (13,960 skills + 99,624 aliases).
    - Our curated list gives us CATEGORIES, because ESCO doesn't have
      "Programming" / "Framework" / etc. — it has its own scheme
      (skill/competence vs knowledge).
    - The search() function uses word-boundary regex so "react" doesn't
      match inside "reaction", and "C" doesn't match inside "code".
    - The curated file is loaded with utf-8-sig so it tolerates BOMs
      (Windows tools like PowerShell's Out-File sometimes add them).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd


# --- Paths -------------------------------------------------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent.parent   # src/nlp/skill_kb.py -> project root
ESCO_SKILLS = PROJECT_ROOT / "data" / "processed" / "esco_skills_clean.csv"
ESCO_ALIAS = PROJECT_ROOT / "data" / "processed" / "esco_alias_map.json"
CURATED = PROJECT_ROOT / "data" / "external" / "curated_skills.json"


# --- Module-level caches (lazy loaded once) ----------------------------------
_ESCO_DF: Optional[pd.DataFrame] = None
_ALIAS_MAP: Optional[Dict[str, str]] = None
_CURATED: Optional[Dict[str, List[str]]] = None
_SKILL_TO_CATEGORY: Optional[Dict[str, str]] = None
_ALL_CANONICAL: Optional[Set[str]] = None


# --- Loaders -----------------------------------------------------------------

def _load_esco() -> pd.DataFrame:
    """Load ESCO cleaned skills CSV (cached)."""
    global _ESCO_DF
    if _ESCO_DF is None:
        if not ESCO_SKILLS.exists():
            raise FileNotFoundError(
                f"ESCO skills file not found: {ESCO_SKILLS}\n"
                "Run Phase 4 cleaning first to produce data/processed/esco_skills_clean.csv."
            )
        _ESCO_DF = pd.read_csv(ESCO_SKILLS)
    return _ESCO_DF


def _load_alias_map() -> Dict[str, str]:
    """Load ESCO alias map (cached)."""
    global _ALIAS_MAP
    if _ALIAS_MAP is None:
        if not ESCO_ALIAS.exists():
            raise FileNotFoundError(
                f"ESCO alias map not found: {ESCO_ALIAS}\n"
                "Run Phase 4 cleaning first to produce data/processed/esco_alias_map.json."
            )
        with open(ESCO_ALIAS, encoding="utf-8-sig") as f:
            _ALIAS_MAP = json.load(f)
    return _ALIAS_MAP


def _load_curated() -> Dict[str, List[str]]:
    """Load curated skills JSON and build the skill -> category reverse map (cached)."""
    global _CURATED, _SKILL_TO_CATEGORY
    if _CURATED is None:
        if not CURATED.exists():
            raise FileNotFoundError(
                f"Curated skills file not found: {CURATED}\n"
                "Create data/external/curated_skills.json first."
            )
        with open(CURATED, encoding="utf-8-sig") as f:
            _CURATED = json.load(f)

        # Build reverse map: lowercased skill -> category
        _SKILL_TO_CATEGORY = {}
        for category, skills in _CURATED.items():
            for skill in skills:
                _SKILL_TO_CATEGORY[skill.lower()] = category
    return _CURATED


# --- Public API --------------------------------------------------------------

def normalize(skill: str) -> str:
    """
    Map any variant to a canonical ESCO preferred label if possible.
    Falls back to the original text (stripped) if no match exists.
    Empty input returns "".
    """
    if not isinstance(skill, str) or not skill.strip():
        return ""
    key = skill.strip().lower()
    alias_map = _load_alias_map()
    return alias_map.get(key, skill.strip())


def get_category(skill: str) -> str:
    """
    Return the curated category for a skill, or 'Other' if unknown.
    Checks the raw form first, then the ESCO-normalized form.
    """
    _load_curated()
    if not isinstance(skill, str):
        return "Other"
    key = skill.strip().lower()
    if key in _SKILL_TO_CATEGORY:
        return _SKILL_TO_CATEGORY[key]

    # Try the ESCO-normalized form
    canonical = normalize(skill).lower()
    if canonical in _SKILL_TO_CATEGORY:
        return _SKILL_TO_CATEGORY[canonical]
    return "Other"


def get_skills_by_category(category: str) -> List[str]:
    """Return the curated skills for a given category (or [] if unknown)."""
    return list(_load_curated().get(category, []))


def all_categories() -> List[str]:
    """Return the list of category names, in file order."""
    return list(_load_curated().keys())


def all_canonical_skills() -> Set[str]:
    """
    Return the union of ESCO preferred labels and curated skills,
    all lowercased, as a set. Cached.
    """
    global _ALL_CANONICAL
    if _ALL_CANONICAL is None:
        skills: Set[str] = set()

        # ESCO preferred labels
        esco = _load_esco()
        for label in esco["preferredLabel"].dropna():
            skills.add(str(label).strip().lower())

        # Curated skills
        _load_curated()
        for cat_skills in _CURATED.values():
            for s in cat_skills:
                skills.add(s.strip().lower())

        _ALL_CANONICAL = skills
    return _ALL_CANONICAL


def search(text: str, min_length: int = 2) -> List[Dict[str, str]]:
    """
    Find all known skills mentioned in `text`.

    Returns a list of dicts, one per matched skill:
        {
            "skill":      the matched form (original case),
            "canonical":  canonical name (same as skill for curated),
            "category":   Programming / Framework / ... / Other,
            "matched_as": "curated" or "esco",
        }

    Matching strategy:
        1. Curated skills first (higher priority — we have categories for them).
        2. ESCO preferred labels fill in the rest.
    Uses word-boundary-style regex so "react" doesn't match inside "reaction"
    and "C" doesn't match inside "code".
    """
    if not isinstance(text, str) or not text.strip():
        return []

    _load_curated()
    text_lower = text.lower()
    found: Dict[str, Dict[str, str]] = {}

    # 1. Curated skills
    for category, skills in _CURATED.items():
        for skill in skills:
            if len(skill) < min_length:
                continue
            pattern = re.escape(skill.lower())
            # Not preceded/followed by an alphanumeric character
            if re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", text_lower):
                key = skill.lower()
                if key not in found:
                    found[key] = {
                        "skill": skill,
                        "canonical": skill,
                        "category": category,
                        "matched_as": "curated",
                    }

    # 2. ESCO preferred labels (only for skills not already found)
    esco = _load_esco()
    for label in esco["preferredLabel"].dropna():
        label_str = str(label).strip()
        if len(label_str) < min_length:
            continue
        pattern = re.escape(label_str.lower())
        if re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", text_lower):
            key = label_str.lower()
            if key not in found:
                found[key] = {
                    "skill": label_str,
                    "canonical": label_str,
                    "category": "Other",
                    "matched_as": "esco",
                }

    return list(found.values())