"""
Recommendation engine — generates personalized learning recommendations
based on missing skills identified by the skill gap analyzer.

Public API:
    - generate_recommendations(gap_result) -> list of recommendation dicts
    - summarize_recommendations(recs) -> short text summary

Each recommendation:
    {
        "skill":          name of the missing skill,
        "priority":       "high" | "medium" | "low",
        "category":       "course" | "project" | "practice" | "reading" | "other",
        "recommendation": the actual recommendation text,
        "role":           "required" | "preferred" (from the gap analysis),
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import re


_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent.parent
TEMPLATES_PATH = PROJECT_ROOT / "data" / "external" / "recommendations.json"


# --- Cached templates --------------------------------------------------------
_TEMPLATES: Optional[Dict] = None


def _load_templates() -> Dict:
    global _TEMPLATES
    if _TEMPLATES is None:
        if not TEMPLATES_PATH.exists():
            raise FileNotFoundError(
                f"Recommendation templates not found: {TEMPLATES_PATH}. "
                "Create data/external/recommendations.json first."
            )
        with open(TEMPLATES_PATH, encoding="utf-8-sig") as f:
            _TEMPLATES = json.load(f)
    return _TEMPLATES


# --- Skill normalization for template matching ------------------------------

def _normalize_skill_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace — for template key matching."""
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name.strip())


def _find_template(skill: str, templates: Dict) -> Optional[Dict]:
    """
    Look up a template for a skill name. Case-insensitive exact match.
    Falls back to None if no template found.
    """
    if not skill:
        return None
    key_norm = _normalize_skill_name(skill).lower()
    for tpl_skill, tpl in templates.items():
        if _normalize_skill_name(tpl_skill).lower() == key_norm:
            return tpl
    return None


# --- Fallback: no template available ----------------------------------------

def _generic_recommendation(skill: str, role: str) -> Dict:
    """
    When we don't have a specific template for a skill, generate a generic
    but honest recommendation based on the role (required vs preferred).
    """
    if role == "required":
        text = (
            f"'{skill}' is a required skill for this role. "
            f"Find an introductory tutorial or course and build a small project "
            f"that uses it — a portfolio piece is worth more than a certificate."
        )
        priority = "high"
    else:
        text = (
            f"'{skill}' is a nice-to-have. If you have time after covering "
            f"required skills, explore a beginner tutorial and try a small exercise."
        )
        priority = "medium"
    return {
        "recommendation": text,
        "priority": priority,
        "category": "other",
    }


# --- Main API ---------------------------------------------------------------

def generate_recommendations(gap_result: Dict) -> List[Dict]:
    """
    Given the output of analyze_gap() from src/matching/skill_gap.py,
    return an ordered list of recommendation dicts.

    Order:
        1. Missing required skills (high priority first)
        2. Missing preferred skills
    """
    if not isinstance(gap_result, dict):
        return []

    templates = _load_templates()

    recs: List[Dict] = []

    # --- Required first ---
    for skill in gap_result.get("missing_required", []) or []:
        tpl = _find_template(skill, templates)
        if tpl is None:
            tpl = _generic_recommendation(skill, "required")
        recs.append({
            "skill": skill,
            "priority": tpl.get("priority", "high"),
            "category": tpl.get("category", "other"),
            "recommendation": tpl.get("recommendation", ""),
            "role": "required",
        })

    # --- Then preferred ---
    for skill in gap_result.get("missing_preferred", []) or []:
        tpl = _find_template(skill, templates)
        if tpl is None:
            tpl = _generic_recommendation(skill, "preferred")
        recs.append({
            "skill": skill,
            "priority": tpl.get("priority", "medium"),
            "category": tpl.get("category", "other"),
            "recommendation": tpl.get("recommendation", ""),
            "role": "preferred",
        })

    # --- Sort: required first, then by priority (high > medium > low) ---
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    role_rank = {"required": 0, "preferred": 1}

    recs.sort(key=lambda r: (
        role_rank.get(r["role"], 9),
        priority_rank.get(r["priority"], 9),
        r["skill"].lower(),
    ))

    return recs


def summarize_recommendations(recs: List[Dict]) -> Dict[str, int]:
    """
    Return counts grouped by priority and role for the UI.
    """
    if not recs:
        return {"total": 0, "high": 0, "medium": 0, "low": 0,
                "required": 0, "preferred": 0}
    return {
        "total": len(recs),
        "high": sum(1 for r in recs if r["priority"] == "high"),
        "medium": sum(1 for r in recs if r["priority"] == "medium"),
        "low": sum(1 for r in recs if r["priority"] == "low"),
        "required": sum(1 for r in recs if r["role"] == "required"),
        "preferred": sum(1 for r in recs if r["role"] == "preferred"),
    }