"""
Skill gap analysis — compare a resume's skills against a JD's requirements.

Public API:
    - analyze_gap(resume_skills, jd_skills) -> dict with:
        {
            "matched":                [skill names],
            "missing_required":       [skill names],
            "missing_preferred":      [skill names],
            "extra":                  [skills the candidate has that the JD doesn't require],
            "match_percentage":       float in [0, 100],
            "required_coverage":      float in [0, 1],
            "preferred_coverage":     float in [0, 1],
            "importance_breakdown":   [{"skill": ..., "role": ..., "importance": ...}],
        }

Inputs `resume_skills` and `jd_skills` may be:
    - List[str]  (just skill names), OR
    - List[Dict] (extracted skills with "skill" and possibly "category" keys)

We accept both for flexibility with the extractor output.
"""

from __future__ import annotations

from typing import Dict, List, Union

SkillInput = Union[List[str], List[Dict], None]


# --- Normalization helpers ---------------------------------------------------

def _names(skills: SkillInput) -> List[str]:
    """Convert either list-of-str or list-of-dict to a normalized list of names."""
    if not skills:
        return []
    out = []
    for s in skills:
        if isinstance(s, str):
            name = s.strip()
        elif isinstance(s, dict):
            name = (s.get("skill") or s.get("canonical") or "").strip()
        else:
            continue
        if name:
            out.append(name)
    return out


def _key(name: str) -> str:
    """Normalize a skill name for case-insensitive comparison."""
    return name.strip().lower()


def _unique_preserving_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for x in items:
        k = _key(x)
        if k and k not in seen:
            seen.add(k)
            result.append(x)
    return result


# --- Importance weighting ----------------------------------------------------

# Required skills count 1.0; preferred count 0.4
# (chosen empirically to reflect "critical gap" vs "nice-to-have")
IMPORTANCE_REQUIRED = 1.0
IMPORTANCE_PREFERRED = 0.4


# --- Main entry point --------------------------------------------------------

def analyze_gap(resume_skills: SkillInput, jd_skills: SkillInput) -> Dict:
    """
    Compute the skill gap between a resume and a JD.

    `jd_skills` may be:
        - List[str]  → all treated as required
        - List[Dict] → each item may have {"skill": ..., "role": "required"|"preferred"}
        - Dict with keys {"required": [...], "preferred": [...]}  (extractor output)
    """
    # --- Normalize resume skills ---
    resume_names = _unique_preserving_order(_names(resume_skills))
    resume_keys = {_key(n) for n in resume_names}

    # --- Normalize JD skills (handle the various input shapes) ---
    required_names, preferred_names = _parse_jd_skills(jd_skills)

    required_keys = {_key(n) for n in required_names}
    preferred_keys = {_key(n) for n in preferred_names}

    # --- Compute matched / missing ---
    matched_keys = resume_keys & (required_keys | preferred_keys)
    matched = [n for n in resume_names if _key(n) in matched_keys]
    # Also include JD skills that matched but weren't in resume order (rare)
    for n in required_names + preferred_names:
        if _key(n) in matched_keys and _key(n) not in {_key(m) for m in matched}:
            matched.append(n)

    missing_required = [n for n in required_names if _key(n) not in resume_keys]
    missing_preferred = [n for n in preferred_names if _key(n) not in resume_keys]
    extra = [n for n in resume_names if _key(n) not in (required_keys | preferred_keys)]

    # --- Coverage ---
    required_coverage = (
        (len(required_keys) - len(missing_required)) / len(required_keys)
        if required_keys else 1.0
    )
    preferred_coverage = (
        (len(preferred_keys) - len(missing_preferred)) / len(preferred_keys)
        if preferred_keys else 1.0
    )

    # --- Weighted match percentage ---
    # Total weight = required * 1.0 + preferred * 0.4
    total_weight = (
        len(required_keys) * IMPORTANCE_REQUIRED
        + len(preferred_keys) * IMPORTANCE_PREFERRED
    )
    matched_weight = (
        (len(required_keys) - len(missing_required)) * IMPORTANCE_REQUIRED
        + (len(preferred_keys) - len(missing_preferred)) * IMPORTANCE_PREFERRED
    )
    match_percentage = (matched_weight / total_weight * 100) if total_weight > 0 else 0.0

    # --- Importance breakdown for UI display ---
    importance_breakdown = []
    for n in required_names:
        importance_breakdown.append({
            "skill": n,
            "role": "required",
            "importance": IMPORTANCE_REQUIRED,
            "matched": _key(n) in resume_keys,
        })
    for n in preferred_names:
        importance_breakdown.append({
            "skill": n,
            "role": "preferred",
            "importance": IMPORTANCE_PREFERRED,
            "matched": _key(n) in resume_keys,
        })
    importance_breakdown.sort(
        key=lambda x: (-x["importance"], not x["matched"], x["skill"])
    )

    return {
        "matched": matched,
        "missing_required": missing_required,
        "missing_preferred": missing_preferred,
        "extra": extra,
        "match_percentage": round(match_percentage, 2),
        "required_coverage": round(required_coverage, 4),
        "preferred_coverage": round(preferred_coverage, 4),
        "counts": {
            "required_total": len(required_names),
            "preferred_total": len(preferred_names),
            "matched_total": len(matched),
            "missing_required_total": len(missing_required),
            "missing_preferred_total": len(missing_preferred),
            "extra_total": len(extra),
        },
        "importance_breakdown": importance_breakdown,
    }


# --- Internal: parse JD skills in any shape ----------------------------------

def _parse_jd_skills(jd_skills: SkillInput) -> tuple:
    """
    Return (required_list, preferred_list) after normalizing any accepted shape.

    Accepted shapes:
        List[str]                                → (list, [])
        List[Dict] with "skill" and maybe "role" → split by role
        Dict with {"required": [...], "preferred": [...]} → use keys directly
    """
    required: List[str] = []
    preferred: List[str] = []

    if not jd_skills:
        return [], []

    # Case 1 — dict with required/preferred keys
    if isinstance(jd_skills, dict):
        req = jd_skills.get("required", [])
        pref = jd_skills.get("preferred", [])
        required = _unique_preserving_order(_names(req))
        preferred = _unique_preserving_order(_names(pref))
        return required, preferred

    # Case 2 — list
    if isinstance(jd_skills, list):
        for s in jd_skills:
            if isinstance(s, str):
                required.append(s.strip())
            elif isinstance(s, dict):
                name = (s.get("skill") or s.get("canonical") or "").strip()
                if not name:
                    continue
                role = (s.get("role") or "required").lower()
                if role == "preferred":
                    preferred.append(name)
                else:
                    required.append(name)
        return _unique_preserving_order(required), _unique_preserving_order(preferred)

    return [], []