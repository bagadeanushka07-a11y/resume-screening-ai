"""
Tests for src/nlp/tfidf_matcher.py
Run with: python -m pytest tests/test_tfidf_matcher.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nlp.tfidf_matcher import (  # noqa: E402
    compute_similarity,
    compute_similarity_detailed,
)


RESUME = """
John Doe
Skills: Python, Django, PostgreSQL, Docker, AWS, Git, Machine Learning

Experience
3 years building web applications with Python and Django.
Deployed services to AWS using Docker.
Worked with PostgreSQL databases and REST APIs.
"""

GOOD_JD = """
We are hiring a Senior Python Engineer.

Required skills:
- 5+ years Python experience
- Strong knowledge of Django, PostgreSQL, Docker, AWS
- Experience building REST APIs

Preferred:
- Kubernetes, Redis
"""

BAD_JD = """
We need an experienced chef to run our kitchen.
Must have 5 years cooking Italian cuisine.
Knowledge of pasta, sauces, and plating required.
"""


# --- Basic behavior ----------------------------------------------------------

def test_similarity_returns_float_in_range():
    score = compute_similarity(RESUME, GOOD_JD)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_similarity_identical_texts():
    """Identical texts should have score 1.0 (or very close)."""
    score = compute_similarity(RESUME, RESUME)
    assert score > 0.95


def test_similarity_empty_inputs():
    assert compute_similarity("", GOOD_JD) == 0.0
    assert compute_similarity(RESUME, "") == 0.0
    assert compute_similarity("", "") == 0.0
    assert compute_similarity(None, GOOD_JD) == 0.0
    assert compute_similarity(RESUME, None) == 0.0


def test_similarity_good_match_higher_than_bad():
    """A relevant JD should score higher than an unrelated one."""
    good = compute_similarity(RESUME, GOOD_JD)
    bad = compute_similarity(RESUME, BAD_JD)
    assert good > bad, f"good={good}, bad={bad}"
    assert good > 0.05   # not just noise


def test_similarity_no_shared_vocabulary():
    score = compute_similarity("apple banana cherry", "xylophone yacht zebra")
    assert score == 0.0


def test_similarity_is_symmetric():
    """Cosine similarity is symmetric."""
    a = compute_similarity(RESUME, GOOD_JD)
    b = compute_similarity(GOOD_JD, RESUME)
    assert abs(a - b) < 1e-6


def test_similarity_case_insensitive():
    lower = compute_similarity("python django docker", "python django docker")
    upper = compute_similarity("PYTHON DJANGO DOCKER", "PYTHON DJANGO DOCKER")
    assert abs(lower - upper) < 1e-6


# --- Detailed breakdown ------------------------------------------------------

def test_detailed_returns_expected_keys():
    result = compute_similarity_detailed(RESUME, GOOD_JD)
    for key in ["score", "shared_terms", "resume_top_terms", "jd_top_terms"]:
        assert key in result


def test_detailed_score_matches_simple():
    """Score in detailed output should match compute_similarity."""
    simple = compute_similarity(RESUME, GOOD_JD)
    detailed = compute_similarity_detailed(RESUME, GOOD_JD)["score"]
    assert abs(simple - detailed) < 1e-3


def test_detailed_shared_terms_populated():
    result = compute_similarity_detailed(RESUME, GOOD_JD)
    shared = result["shared_terms"]
    assert len(shared) > 0
    shared_names = {term for term, _ in shared}
    # Should find at least python and docker
    assert "python" in shared_names
    assert "docker" in shared_names


def test_detailed_shared_terms_have_weights():
    result = compute_similarity_detailed(RESUME, GOOD_JD)
    for term, weight in result["shared_terms"]:
        assert isinstance(term, str)
        assert isinstance(weight, float)
        assert weight > 0


def test_detailed_resume_top_terms():
    result = compute_similarity_detailed(RESUME, GOOD_JD)
    assert len(result["resume_top_terms"]) > 0
    for term, weight in result["resume_top_terms"]:
        assert isinstance(term, str)
        assert weight > 0


def test_detailed_empty_input():
    result = compute_similarity_detailed("", "")
    assert result["score"] == 0.0
    assert result["shared_terms"] == []


def test_detailed_none_input():
    result = compute_similarity_detailed(None, None)
    assert result["score"] == 0.0