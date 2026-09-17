"""
Tests for src/nlp/jd_parser.py
Run with: python -m pytest tests/test_jd_parser.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nlp.jd_parser import parse_jd  # noqa: E402


SAMPLE_JD = """
Job Title: Senior Python Engineer

We are seeking a Senior Python Engineer with 5+ years of experience
to join our backend team. This is a fully remote full-time position.

Required skills:
- 5+ years Python experience
- Strong knowledge of Django, PostgreSQL
- Docker and AWS required
- Bachelor's degree in Computer Science or equivalent

Preferred skills:
- Kubernetes experience is a plus
- Familiarity with Redis and React.js nice-to-have
"""


# --- Title ------------------------------------------------------------------

def test_title_explicit_header():
    result = parse_jd(SAMPLE_JD)
    assert result["title"] == "Senior Python Engineer"


def test_title_missing():
    result = parse_jd("We are hiring. Contact us for details.")
    assert result["title"] is None


# --- Experience --------------------------------------------------------------

def test_experience_years_extracted():
    result = parse_jd(SAMPLE_JD)
    assert result["experience_years"] == 5


def test_experience_years_missing():
    result = parse_jd("A junior developer position in a cool team.")
    assert result["experience_years"] is None


def test_experience_years_range():
    # "3-5 years" should return 3 (the minimum)
    result = parse_jd("We need 3-5 years of experience.")
    assert result["experience_years"] == 3


def test_seniority_detection():
    result = parse_jd(SAMPLE_JD)
    assert result["seniority"] == "senior"


def test_seniority_entry():
    result = parse_jd("Entry level software engineer wanted.")
    assert result["seniority"] == "entry"


def test_seniority_missing():
    result = parse_jd("Software engineer wanted.")
    assert result["seniority"] is None


# --- Education --------------------------------------------------------------

def test_education_bachelors():
    result = parse_jd("Requires a Bachelor's degree in CS.")
    assert result["education"] == "Bachelor's"


def test_education_higher_priority():
    # When both Bachelor's and Master's are mentioned, Master's wins
    text = "Bachelor's degree required, Master's preferred."
    result = parse_jd(text)
    assert result["education"] == "Master's"


def test_education_phd_wins():
    text = "PhD preferred, Master's required."
    result = parse_jd(text)
    assert result["education"] == "PhD"


def test_education_missing():
    result = parse_jd("We just need a good coder.")
    assert result["education"] is None


# --- Employment type ---------------------------------------------------------

def test_full_time_detection():
    result = parse_jd("Full-time position available.")
    assert "full-time" in result["employment_types"]


def test_contract_detection():
    result = parse_jd("This is a contract role.")
    assert "contract" in result["employment_types"]


def test_employment_empty():
    result = parse_jd("Just a job.")
    assert result["employment_types"] == []


# --- Remote ------------------------------------------------------------------

def test_fully_remote():
    result = parse_jd("This is a fully remote position.")
    assert result["remote"] == "Fully Remote"


def test_remote_generic():
    result = parse_jd("Remote work available.")
    assert result["remote"] == "Remote"


def test_hybrid():
    result = parse_jd("Hybrid role, 3 days in office.")
    assert result["remote"] == "Hybrid"


# --- Keywords ----------------------------------------------------------------

def test_keywords_extracted():
    result = parse_jd(SAMPLE_JD)
    assert isinstance(result["keywords"], list)
    assert len(result["keywords"]) > 0


def test_keywords_filtered():
    """Common noise words should not appear as keywords."""
    result = parse_jd(SAMPLE_JD)
    keywords_lower = [k.lower() for k in result["keywords"]]
    assert "this" not in keywords_lower
    assert "title" not in keywords_lower
    assert "required" not in keywords_lower


# --- Skills ------------------------------------------------------------------

def test_skills_required_populated():
    result = parse_jd(SAMPLE_JD)
    required = {s["skill"].lower() for s in result["skills"]["required"]}
    assert "python" in required
    assert "django" in required


def test_skills_preferred_populated():
    result = parse_jd(SAMPLE_JD)
    preferred = {s["skill"].lower() for s in result["skills"]["preferred"]}
    assert "kubernetes" in preferred or "redis" in preferred


def test_skills_react_deduped():
    result = parse_jd(SAMPLE_JD)
    all_skills = (
        result["skills"]["required"]
        + result["skills"]["preferred"]
        + result["skills"]["unspecified"]
    )
    react_names = [s["skill"] for s in all_skills if s["skill"].lower() in ("react", "react.js")]
    assert len(react_names) == 1


# --- Empty input ------------------------------------------------------------

def test_empty_input():
    result = parse_jd("")
    assert result["title"] is None
    assert result["experience_years"] is None
    assert result["education"] is None
    assert result["employment_types"] == []
    assert result["remote"] is None
    assert result["keywords"] == []
    assert result["skills"] == {"required": [], "preferred": [], "unspecified": []}


def test_none_input():
    result = parse_jd(None)
    assert result["title"] is None
    assert result["skills"] == {"required": [], "preferred": [], "unspecified": []}