"""
Tests for src/recommendations/recommendation_engine.py
Run with: python -m pytest tests/test_recommendations.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommendations.recommendation_engine import (  # noqa: E402
    generate_recommendations,
    summarize_recommendations,
)


# --- Basic behavior ----------------------------------------------------------

def test_empty_gap_returns_empty_list():
    assert generate_recommendations({}) == []
    assert generate_recommendations(None) == []


def test_gap_with_no_missing_skills_returns_empty():
    gap = {"missing_required": [], "missing_preferred": []}
    assert generate_recommendations(gap) == []


def test_returns_expected_keys():
    gap = {"missing_required": ["Docker"], "missing_preferred": []}
    recs = generate_recommendations(gap)
    assert len(recs) == 1
    r = recs[0]
    for key in ["skill", "priority", "category", "recommendation", "role"]:
        assert key in r


# --- Required vs preferred ---------------------------------------------------

def test_required_skills_marked_required():
    gap = {"missing_required": ["Docker"], "missing_preferred": []}
    recs = generate_recommendations(gap)
    assert recs[0]["role"] == "required"


def test_preferred_skills_marked_preferred():
    gap = {"missing_required": [], "missing_preferred": ["Redis"]}
    recs = generate_recommendations(gap)
    assert recs[0]["role"] == "preferred"


def test_required_comes_before_preferred():
    gap = {"missing_required": ["Docker"], "missing_preferred": ["Redis"]}
    recs = generate_recommendations(gap)
    assert recs[0]["role"] == "required"
    assert recs[1]["role"] == "preferred"


# --- Priority ordering -------------------------------------------------------

def test_within_role_high_priority_comes_first():
    # Both Docker and AWS are "high" in our templates
    gap = {"missing_required": ["Docker", "AWS"], "missing_preferred": []}
    recs = generate_recommendations(gap)
    # Both should be high priority
    assert all(r["priority"] == "high" for r in recs)


def test_sorting_is_deterministic():
    """Same input should produce same order."""
    gap = {"missing_required": ["Docker", "AWS", "CI/CD"], "missing_preferred": []}
    recs1 = generate_recommendations(gap)
    recs2 = generate_recommendations(gap)
    assert [r["skill"] for r in recs1] == [r["skill"] for r in recs2]


# --- Template lookup --------------------------------------------------------

def test_known_skill_uses_template():
    gap = {"missing_required": ["Docker"], "missing_preferred": []}
    recs = generate_recommendations(gap)
    # Should mention "Dockerfile" or "containerize" from our template
    assert "docker" in recs[0]["recommendation"].lower()


def test_known_skill_case_insensitive():
    gap = {"missing_required": ["docker"], "missing_preferred": []}
    recs = generate_recommendations(gap)
    # Still finds the Docker template (case-insensitive)
    assert "docker" in recs[0]["recommendation"].lower()


def test_unknown_skill_uses_fallback():
    gap = {"missing_required": ["SomeRandomSkillXYZ"], "missing_preferred": []}
    recs = generate_recommendations(gap)
    assert len(recs) == 1
    assert recs[0]["skill"] == "SomeRandomSkillXYZ"
    assert recs[0]["priority"] == "high"  # required → high
    assert "SomeRandomSkillXYZ" in recs[0]["recommendation"]


def test_unknown_preferred_skill_gets_medium_priority():
    gap = {"missing_required": [], "missing_preferred": ["SomeRandomSkill"]}
    recs = generate_recommendations(gap)
    assert recs[0]["priority"] == "medium"


# --- Full scenario -----------------------------------------------------------

def test_full_scenario():
    gap = {
        "missing_required": ["Docker", "AWS", "REST API"],
        "missing_preferred": ["Kubernetes", "Redis"],
    }
    recs = generate_recommendations(gap)
    assert len(recs) == 5
    # Required first
    assert recs[0]["role"] == "required"
    assert recs[1]["role"] == "required"
    assert recs[2]["role"] == "required"
    assert recs[3]["role"] == "preferred"
    assert recs[4]["role"] == "preferred"


# --- Summarize ---------------------------------------------------------------

def test_summarize_empty():
    s = summarize_recommendations([])
    assert s["total"] == 0
    assert s["high"] == 0


def test_summarize_counts():
    gap = {
        "missing_required": ["Docker", "AWS"],   # both high
        "missing_preferred": ["Kubernetes"],     # medium
    }
    recs = generate_recommendations(gap)
    s = summarize_recommendations(recs)
    assert s["total"] == 3
    assert s["high"] == 2
    assert s["medium"] == 1
    assert s["low"] == 0
    assert s["required"] == 2
    assert s["preferred"] == 1


def test_summarize_handles_none():
    s = summarize_recommendations(None)
    assert s["total"] == 0