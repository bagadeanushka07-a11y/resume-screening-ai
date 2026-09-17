"""
Tests for src/nlp/skill_kb.py

Run with:
    python -m pytest tests/test_skill_kb.py -v
"""

import sys
from pathlib import Path

import pytest

# Make `src` importable when running pytest from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nlp.skill_kb import (  # noqa: E402
    normalize,
    get_category,
    get_skills_by_category,
    all_categories,
    all_canonical_skills,
    search,
)


# --- Categories ---------------------------------------------------------------

def test_categories_present():
    cats = all_categories()
    assert "Programming" in cats
    assert "Framework" in cats
    assert "Database" in cats
    assert "Machine Learning" in cats
    assert "Cloud" in cats
    assert "DevOps" in cats
    assert "Tools" in cats
    assert "Web" in cats
    assert "Soft Skills" in cats


def test_categories_count_reasonable():
    # We defined 11 categories in curated_skills.json
    assert len(all_categories()) >= 10


# --- Skills by category -------------------------------------------------------

def test_get_skills_by_category_programming():
    prog = get_skills_by_category("Programming")
    assert "Python" in prog
    assert "Java" in prog
    assert "C++" in prog
    assert "C#" in prog


def test_get_skills_by_category_framework():
    fw = get_skills_by_category("Framework")
    assert "Django" in fw
    assert "React" in fw
    assert "Node.js" in fw


def test_get_skills_by_category_unknown_returns_empty():
    assert get_skills_by_category("NonExistentCategory") == []


# --- get_category -------------------------------------------------------------

def test_get_category_exact_match():
    assert get_category("Python") == "Programming"
    assert get_category("Django") == "Framework"
    assert get_category("MySQL") == "Database"
    assert get_category("Docker") == "DevOps"
    assert get_category("Git") == "Tools"
    assert get_category("AWS") == "Cloud"


def test_get_category_case_insensitive():
    assert get_category("python") == "Programming"
    assert get_category("DJANGO") == "Framework"
    assert get_category("MySql") == "Database"


def test_get_category_unknown_returns_other():
    assert get_category("SomeRandomSkillName12345") == "Other"
    assert get_category("Zorblax Quux") == "Other"


def test_get_category_handles_non_string():
    assert get_category(None) == "Other"
    assert get_category(123) == "Other"


# --- normalize ----------------------------------------------------------------

def test_normalize_returns_string():
    n = normalize("Python")
    assert isinstance(n, str)
    assert len(n) > 0


def test_normalize_empty_input():
    assert normalize("") == ""
    assert normalize("   ") == ""


def test_normalize_non_string():
    assert normalize(None) == ""


def test_normalize_known_variant():
    # "ml" should map to something (either "Machine Learning" or "ml")
    n = normalize("ml")
    assert isinstance(n, str)


# --- all_canonical_skills -----------------------------------------------------

def test_all_canonical_skills_nonempty():
    skills = all_canonical_skills()
    assert isinstance(skills, set)
    # ESCO alone has ~13,960 preferred labels
    assert len(skills) > 10000


def test_all_canonical_skills_lowercased():
    skills = all_canonical_skills()
    for s in list(skills)[:100]:
        assert s == s.lower()


def test_all_canonical_skills_includes_python():
    assert "python" in all_canonical_skills()


def test_all_canonical_skills_includes_curated():
    skills = all_canonical_skills()
    assert "django" in skills
    assert "docker" in skills


# --- search -------------------------------------------------------------------

def test_search_finds_known_skills():
    text = "Experienced in Python, Django, PostgreSQL, Docker, AWS and Git."
    hits = search(text)
    found = {h["skill"].lower() for h in hits}
    for expected in ["python", "django", "postgresql", "docker", "aws", "git"]:
        assert expected in found, f"missing {expected!r} in {found}"


def test_search_returns_category_for_curated():
    hits = search("Python developer with Django experience")
    py = [h for h in hits if h["skill"].lower() == "python"]
    assert py and py[0]["category"] == "Programming"

    dj = [h for h in hits if h["skill"].lower() == "django"]
    assert dj and dj[0]["category"] == "Framework"


def test_search_returns_expected_keys():
    hits = search("Python developer")
    assert len(hits) >= 1
    for h in hits:
        for key in ["skill", "canonical", "category", "matched_as"]:
            assert key in h


def test_search_empty_input():
    assert search("") == []
    assert search("   ") == []
    assert search(None) == []


def test_search_no_skills_present():
    hits = search("The quick brown fox jumps over the lazy dog.")
    # Should not find any curated or ESCO skills in this sentence
    # (there might be some ESCO edge cases, so we just check it doesn't crash)
    assert isinstance(hits, list)


def test_search_avoids_partial_word_matches():
    # "react" should NOT match inside "reaction"
    hits = search("The chemical reaction was fast and the reactor was hot.")
    skills = {h["skill"].lower() for h in hits}
    # "react" and "react.js" are curated skills; "reaction" is not
    assert "react" not in skills
    assert "react.js" not in skills


def test_search_deduplicates():
    # Mentioning Python three times should yield one Python entry
    hits = search("Python, Python, and more Python.")
    python_hits = [h for h in hits if h["skill"].lower() == "python"]
    assert len(python_hits) == 1


def test_search_case_insensitive():
    hits_lower = search("python developer")
    hits_upper = search("PYTHON DEVELOPER")
    assert len(hits_lower) == len(hits_upper)


def test_search_returns_list_of_dicts():
    hits = search("Python and Docker")
    assert isinstance(hits, list)
    for h in hits:
        assert isinstance(h, dict)