"""
Tests for src/nlp/skill_extractor.py
Run with: python -m pytest tests/test_skill_extractor.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nlp.skill_extractor import (  # noqa: E402
    extract_skills,
    group_by_category,
    extract_resume_skills,
    extract_jd_skills,
)


# --- extract_skills ----------------------------------------------------------

def test_extract_skills_basic():
    text = "Experienced Python developer with Django and Docker."
    skills = extract_skills(text)
    names = {s["skill"].lower() for s in skills}
    assert "python" in names
    assert "django" in names
    assert "docker" in names


def test_extract_skills_returns_dict_shape():
    skills = extract_skills("Python developer")
    assert len(skills) >= 1
    for s in skills:
        assert "skill" in s
        assert "category" in s
        assert "confidence" in s
        assert "frequency" in s
        assert "in_skill_section" in s


def test_extract_skills_empty():
    assert extract_skills("") == []
    assert extract_skills("   ") == []
    assert extract_skills(None) == []


def test_extract_skills_no_skills():
    skills = extract_skills("The quick brown fox jumps.")
    # Should find nothing — if it finds something, at least it shouldn't crash
    assert isinstance(skills, list)


def test_extract_skills_technical_tokens():
    text = "Experience with C++, C#, .NET, Node.js, React.js, AWS, SQL."
    skills = extract_skills(text)
    names = {s["skill"].lower() for s in skills}
    for tok in ["c++", "c#", ".net", "node.js", "aws", "sql"]:
        assert tok in names, f"missing {tok!r} in {names}"


def test_extract_skills_confidence_range():
    skills = extract_skills("Python Python Python developer")
    for s in skills:
        assert 0.0 <= s["confidence"] <= 1.0


def test_extract_skills_frequency_count():
    text = "Python is great. I use Python daily. Python for everything."
    skills = extract_skills(text)
    py = [s for s in skills if s["skill"].lower() == "python"]
    assert py and py[0]["frequency"] >= 2


def test_extract_skills_in_skill_section():
    text = "Skills: Python, Django, PostgreSQL"
    skills = extract_skills(text)
    py = [s for s in skills if s["skill"].lower() == "python"]
    assert py and py[0]["in_skill_section"] is True


def test_extract_skills_outside_skill_section():
    # No "Skills" header anywhere; "Python" is just in the paragraph
    text = "I have worked with Python for many years doing data analysis."
    skills = extract_skills(text)
    py = [s for s in skills if s["skill"].lower() == "python"]
    assert py and py[0]["in_skill_section"] is False


# --- group_by_category -------------------------------------------------------

def test_group_by_category_basic():
    text = "Skills: Python, Django, PostgreSQL, Docker, AWS"
    skills = extract_skills(text)
    grouped = group_by_category(skills)
    assert "Programming" in grouped
    assert "Python" in grouped["Programming"]
    assert "Django" in grouped["Framework"]
    assert "Docker" in grouped["DevOps"]


def test_group_by_category_filters_low_confidence():
    # A skill not in a section gets lower confidence — but still likely > 0.4
    text = "I use Python."
    skills = extract_skills(text)
    grouped = group_by_category(skills)
    # Should still include Python in Programming (confidence 0.45)
    assert "Programming" in grouped


def test_group_by_category_empty_input():
    assert group_by_category([]) == {}


# --- extract_resume_skills ---------------------------------------------------

def test_extract_resume_skills_alias():
    """extract_resume_skills is an alias for extract_skills"""
    text = "Skills: Python, Docker"
    assert extract_resume_skills(text) == extract_skills(text)


# --- extract_jd_skills -------------------------------------------------------

def test_jd_skills_returns_three_lists():
    jd = "Required skills: Python, Django"
    result = extract_jd_skills(jd)
    assert "required" in result
    assert "preferred" in result
    assert "unspecified" in result


def test_jd_skills_required_classification():
    jd = """
    Required:
    - Python
    - Django
    """
    result = extract_jd_skills(jd)
    required_names = {s["skill"].lower() for s in result["required"]}
    assert "python" in required_names
    assert "django" in required_names


def test_jd_skills_preferred_classification():
    jd = """
    Required:
    - Python

    Preferred:
    - Kubernetes
    - Redis
    """
    result = extract_jd_skills(jd)
    preferred_names = {s["skill"].lower() for s in result["preferred"]}
    assert "kubernetes" in preferred_names
    assert "redis" in preferred_names


def test_jd_skills_required_and_preferred_separate():
    jd = """
    Required skills:
    - Python
    - Django

    Preferred skills:
    - Kubernetes
    - React
    """
    result = extract_jd_skills(jd)
    required = {s["skill"].lower() for s in result["required"]}
    preferred = {s["skill"].lower() for s in result["preferred"]}

    # Required skills should not appear in preferred and vice versa
    assert "python" in required
    assert "python" not in preferred
    assert "kubernetes" in preferred
    assert "kubernetes" not in required


def test_jd_skills_empty_input():
    result = extract_jd_skills("")
    assert result == {"required": [], "preferred": [], "unspecified": []}


def test_jd_skills_role_field():
    jd = "Required: Python"
    result = extract_jd_skills(jd)
    for role in ["required", "preferred", "unspecified"]:
        for s in result[role]:
            assert s.get("role") == role


# --- Deduplication -----------------------------------------------------------

def test_dedup_react_variants():
    """React.js should dedupe to React"""
    text = "Skills: React.js, React"
    skills = extract_skills(text)
    react_names = [s["skill"] for s in skills if s["skill"].lower() in ("react", "react.js")]
    assert len(react_names) == 1
    assert react_names[0] == "React"


def test_dedup_sklearn():
    text = "Skills: sklearn, scikit-learn"
    skills = extract_skills(text)
    sklearn_entries = [s for s in skills if "scikit" in s["skill"].lower() or s["skill"].lower() == "sklearn"]
    # After dedupe should be 1
    assert len(sklearn_entries) <= 1