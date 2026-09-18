"""
Tests for the MySQL integration.

These tests REQUIRE a running MySQL server and a valid .env file.
They will SKIP (not fail) if MySQL is unreachable — so CI or fresh
machines don't break.

Run with: python -m pytest tests/test_database.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.database.connection import is_available, init_schema, get_connection  # noqa: E402
from src.database import queries as q  # noqa: E402


# Skip entire module if MySQL is not reachable
pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="MySQL not reachable — check .env and MySQL service",
)


# --- Setup ------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def setup_schema():
    """Ensure the schema exists before any test runs."""
    init_schema()
    yield


@pytest.fixture(scope="module")
def test_user():
    """Create a test user; return its id and email."""
    email = "pytest_user@example.com"
    # Clean up any previous test user
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE email = %s", (email,))
        conn.commit()
    finally:
        conn.close()

    user_id = q.create_user("Pytest User", email, "pytest_password_123")
    return {"id": user_id, "email": email, "password": "pytest_password_123"}


# --- Password hashing -------------------------------------------------------

def test_hash_password_returns_string():
    h = q.hash_password("mypassword")
    assert isinstance(h, str)
    assert len(h) > 20


def test_hash_password_is_not_plaintext():
    h = q.hash_password("mypassword")
    assert h != "mypassword"


def test_hash_password_different_each_time():
    h1 = q.hash_password("mypassword")
    h2 = q.hash_password("mypassword")
    assert h1 != h2  # bcrypt uses random salt


def test_hash_password_empty_raises():
    with pytest.raises(ValueError):
        q.hash_password("")


def test_verify_password_correct():
    h = q.hash_password("secret123")
    assert q.verify_password("secret123", h) is True


def test_verify_password_wrong():
    h = q.hash_password("secret123")
    assert q.verify_password("wrongpass", h) is False


def test_verify_password_empty():
    assert q.verify_password("", "somehash") is False
    assert q.verify_password("somepass", "") is False


# --- Users ------------------------------------------------------------------

def test_create_and_fetch_user(test_user):
    user = q.get_user_by_email(test_user["email"])
    assert user is not None
    assert user["name"] == "Pytest User"
    assert user["email"] == test_user["email"]


def test_create_user_duplicate_email_raises(test_user):
    with pytest.raises(Exception):
        q.create_user("Another", test_user["email"], "anotherpass")


def test_create_user_empty_raises():
    with pytest.raises(ValueError):
        q.create_user("", "x@y.com", "pass")
    with pytest.raises(ValueError):
        q.create_user("Name", "", "pass")
    with pytest.raises(ValueError):
        q.create_user("Name", "x@y.com", "")


def test_get_user_by_email_case_insensitive(test_user):
    user = q.get_user_by_email(test_user["email"].upper())
    assert user is not None


def test_get_user_by_id(test_user):
    user = q.get_user_by_id(test_user["id"])
    assert user is not None
    assert user["id"] == test_user["id"]


def test_get_user_by_email_not_found():
    assert q.get_user_by_email("nonexistent_xyz@example.com") is None


# --- Resumes ----------------------------------------------------------------

def test_create_and_fetch_resume(test_user):
    rid = q.create_resume(test_user["id"], "my_resume.pdf", "Python Django Docker")
    assert isinstance(rid, int) and rid > 0

    r = q.get_resume(rid)
    assert r is not None
    assert r["filename"] == "my_resume.pdf"
    assert "Python" in r["extracted_text"]


def test_list_resumes_for_user(test_user):
    q.create_resume(test_user["id"], "second.pdf", "text")
    resumes = q.list_resumes_for_user(test_user["id"])
    assert isinstance(resumes, list)
    assert len(resumes) >= 1


def test_list_resumes_for_unknown_user():
    resumes = q.list_resumes_for_user(999999)
    assert resumes == []


# --- Jobs --------------------------------------------------------------------

def test_create_and_fetch_job():
    jid = q.create_job("ML Engineer", "Example Corp", "We need ML skills.")
    assert isinstance(jid, int) and jid > 0

    j = q.get_job(jid)
    assert j is not None
    assert j["title"] == "ML Engineer"
    assert j["company"] == "Example Corp"


def test_get_job_not_found():
    assert q.get_job(999999) is None


# --- Match results ----------------------------------------------------------

def test_save_and_fetch_match(test_user):
    rid = q.create_resume(test_user["id"], "match_test.pdf", "Python")
    jid = q.create_job("Test Job", "Test Co", "Test description")

    mid = q.save_match_result(rid, jid, 0.8, 0.6, 1.0, 0.9, 0.5, 0.78)
    assert isinstance(mid, int) and mid > 0


def test_match_history_returns_dicts(test_user):
    history = q.get_match_history(test_user["id"])
    assert isinstance(history, list)
    for h in history:
        assert "filename" in h
        assert "title" in h
        assert "overall_score" in h


def test_match_history_limit(test_user):
    history = q.get_match_history(test_user["id"], limit=1)
    assert len(history) <= 1


# --- Recommendations --------------------------------------------------------

def test_save_and_fetch_recommendation(test_user):
    rid = q.create_resume(test_user["id"], "rec_test.pdf", "text")
    jid = q.create_job("Rec Job", "Rec Co", "desc")
    mid = q.save_match_result(rid, jid, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    q.save_recommendation(mid, "Kubernetes", "medium", "Take a course")
    recs = q.get_recommendations_for_match(mid)
    assert len(recs) == 1
    assert recs[0]["skill_name"] == "Kubernetes"
    assert recs[0]["priority"] == "medium"


# --- Connection -------------------------------------------------------------

def test_is_available_true():
    assert is_available() is True


def test_init_schema_idempotent():
    """Running init_schema twice should not raise."""
    init_schema()
    init_schema()