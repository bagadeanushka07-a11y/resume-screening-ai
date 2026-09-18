"""
Tests for src/matching/skill_gap.py
Run with: python -m pytest tests/test_skill_gap.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.matching.skill_gap import analyze_gap  # noqa: E402


# --- Basic structure ---------------------------------------------------------

def test_returns_expected_keys():
    result = analyze_gap(["Python"], ["Python"])
    for key in ["matched", "missing_required", "missing_preferred",
                "extra", "match_percentage", "required_coverage",
                "preferred_coverage", "counts", "importance_breakdown"]:
        assert key in result


def test_empty_inputs():
    result = analyze_gap([], [])
    assert result["matched"] == []
    assert result["missing_required"] == []
    assert result["missing_preferred"] == []
    assert result["extra"] == []


def test_none_inputs():
    result = analyze_gap(None, None)
    assert result["matched"] == []
    assert result["match_percentage"] == 0.0


# --- Perfect match ----------------------------------------------------------

def test_perfect_match():
    skills = ["Python", "Django", "PostgreSQL"]
    result = analyze_gap(skills, skills)
    assert set(result["matched"]) == set(skills)
    assert result["missing_required"] == []
    assert result["match_percentage"] == 100.0
    assert result["required_coverage"] == 1.0


# --- Basic gap --------------------------------------------------------------

def test_missing_required_detected():
    result = analyze_gap(["Python"], ["Python", "Docker", "AWS"])
    assert "Python" in result["matched"]
    assert set(result["missing_required"]) == {"Docker", "AWS"}
    assert result["match_percentage"] < 100.0


def test_missing_preferred_separate_from_required():
    jd = {
        "required": [{"skill": "Python"}],
        "preferred": [{"skill": "Kubernetes"}, {"skill": "Redis"}],
    }
    result = analyze_gap(["Python"], jd)
    assert result["missing_required"] == []
    assert set(result["missing_preferred"]) == {"Kubernetes", "Redis"}


def test_extra_skills_identified():
    result = analyze_gap(["Python", "Django", "Cooking"], ["Python"])
    assert "Cooking" in result["extra"]
    assert "Django" in result["extra"]


# --- Match percentage math --------------------------------------------------

def test_match_percentage_weighted_correctly():
    """Required counts 1.0, preferred 0.4."""
    jd = {
        "required": [{"skill": "A"}, {"skill": "B"}],
        "preferred": [{"skill": "C"}, {"skill": "D"}],
    }
    # Matched: A, B, C, D (all)
    result = analyze_gap(["A", "B", "C", "D"], jd)
    assert result["match_percentage"] == 100.0

    # Matched: A, B, C (missing preferred D)
    result = analyze_gap(["A", "B", "C"], jd)
    # Expected: (2*1.0 + 1*0.4) / (2*1.0 + 2*0.4) = 2.4 / 2.8 = 85.71%
    assert abs(result["match_percentage"] - 85.71) < 0.1


def test_match_percentage_zero_when_nothing_matched():
    result = analyze_gap(["Ruby"], [{"skill": "Python", "role": "required"}])
    assert result["match_percentage"] == 0.0


def test_no_jd_skills_returns_zero():
    """No JD skills at all → 0% (nothing to match against)."""
    result = analyze_gap(["Python"], [])
    assert result["match_percentage"] == 0.0


# --- Case insensitivity -----------------------------------------------------

def test_case_insensitive_matching():
    result = analyze_gap(["python", "DJANGO"], ["Python", "Django"])
    assert len(result["matched"]) == 2
    assert result["match_percentage"] == 100.0


def test_whitespace_stripped():
    result = analyze_gap(["  Python  "], ["Python"])
    assert result["match_percentage"] == 100.0


# --- Different input shapes -------------------------------------------------

def test_accepts_list_of_strings():
    result = analyze_gap(["Python"], ["Python", "Docker"])
    assert "Python" in result["matched"]
    assert "Docker" in result["missing_required"]


def test_accepts_list_of_dicts():
    resume = [{"skill": "Python", "category": "Programming"}]
    jd = [{"skill": "Python", "role": "required"},
          {"skill": "Redis", "role": "preferred"}]
    result = analyze_gap(resume, jd)
    assert "Python" in result["matched"]
    assert "Redis" in result["missing_preferred"]


def test_accepts_dict_with_required_preferred():
    resume = ["Python"]
    jd = {"required": [{"skill": "Python"}], "preferred": [{"skill": "Redis"}]}
    result = analyze_gap(resume, jd)
    assert "Python" in result["matched"]
    assert "Redis" in result["missing_preferred"]


# --- Deduplication ----------------------------------------------------------

def test_duplicate_resume_skills_deduped():
    result = analyze_gap(["Python", "Python", "python"], ["Python"])
    assert len([s for s in result["matched"] if s.lower() == "python"]) == 1


def test_duplicate_jd_skills_deduped():
    jd = [{"skill": "Python"}, {"skill": "Python"}, {"skill": "python"}]
    result = analyze_gap([], jd)
    # Only one missing Python entry
    assert len(result["missing_required"]) == 1


# --- Coverage --------------------------------------------------------------

def test_required_coverage_calculation():
    jd = {"required": [{"skill": f"S{i}"} for i in range(10)]}
    resume = [{"skill": f"S{i}"} for i in range(7)]
    result = analyze_gap(resume, jd)
    assert result["required_coverage"] == 0.7


def test_preferred_coverage_calculation():
    jd = {"preferred": [{"skill": f"S{i}"} for i in range(4)]}
    resume = [{"skill": f"S{i}"} for i in range(2)]
    result = analyze_gap(resume, jd)
    assert result["preferred_coverage"] == 0.5


# --- Counts -----------------------------------------------------------------

def test_counts_dict_shape():
    jd = {"required": [{"skill": "Python"}, {"skill": "Docker"}],
          "preferred": [{"skill": "Redis"}]}
    result = analyze_gap(["Python"], jd)
    counts = result["counts"]
    assert counts["required_total"] == 2
    assert counts["preferred_total"] == 1
    assert counts["matched_total"] == 1
    assert counts["missing_required_total"] == 1
    assert counts["missing_preferred_total"] == 1


# --- Importance breakdown ---------------------------------------------------

def test_importance_breakdown_sorted_by_importance():
    jd = {"required": [{"skill": "A"}], "preferred": [{"skill": "B"}]}
    result = analyze_gap([], jd)
    breakdown = result["importance_breakdown"]
    # Required should come first (importance 1.0 > 0.4)
    assert breakdown[0]["role"] == "required"
    assert breakdown[0]["importance"] == 1.0
    assert breakdown[1]["role"] == "preferred"
    assert breakdown[1]["importance"] == 0.4


def test_importance_breakdown_has_matched_flag():
    jd = {"required": [{"skill": "Python"}, {"skill": "Docker"}]}
    result = analyze_gap(["Python"], jd)
    by_skill = {b["skill"]: b["matched"] for b in result["importance_breakdown"]}
    assert by_skill["Python"] is True
    assert by_skill["Docker"] is False