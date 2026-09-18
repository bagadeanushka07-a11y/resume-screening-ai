"""
CRUD queries for the resume screening app.

All functions raise on DB error; caller is responsible for handling.
Every write returns the new/updated row's primary key.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import bcrypt

from src.database.connection import get_connection


# --- Password hashing (bcrypt) -----------------------------------------------

def hash_password(password: str) -> str:
    """Return bcrypt hash of the password as a UTF-8 string."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Check a plaintext password against a stored bcrypt hash."""
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# --- Users -------------------------------------------------------------------

def create_user(name: str, email: str, password: str) -> int:
    """Insert a new user. Returns user id. Raises on duplicate email."""
    if not name or not email or not password:
        raise ValueError("name, email, and password are required")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
            (name.strip(), email.strip().lower(), hash_password(password)),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[Dict]:
    """Return user row as dict, or None."""
    if not email:
        return None
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, name, email, password_hash, created_at "
            "FROM users WHERE email = %s",
            (email.strip().lower(),),
        )
        return cur.fetchone()
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict]:
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = %s",
            (user_id,),
        )
        return cur.fetchone()
    finally:
        conn.close()


# --- Resumes -----------------------------------------------------------------

def create_resume(user_id: int, filename: str, extracted_text: str) -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO resumes (user_id, filename, extracted_text) VALUES (%s, %s, %s)",
            (user_id, filename, extracted_text),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_resumes_for_user(user_id: int) -> List[Dict]:
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, filename, upload_date FROM resumes "
            "WHERE user_id = %s ORDER BY upload_date DESC",
            (user_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def get_resume(resume_id: int) -> Optional[Dict]:
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM resumes WHERE id = %s", (resume_id,))
        return cur.fetchone()
    finally:
        conn.close()


# --- Jobs --------------------------------------------------------------------

def create_job(title: str, company: str, description: str) -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO jobs (title, company, description) VALUES (%s, %s, %s)",
            (title, company, description),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_job(job_id: int) -> Optional[Dict]:
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        return cur.fetchone()
    finally:
        conn.close()


# --- Match results -----------------------------------------------------------

def save_match_result(
    resume_id: int,
    job_id: int,
    skill_score: float,
    semantic_score: float,
    experience_score: float,
    education_score: float,
    project_score: float,
    overall_score: float,
) -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO match_results
               (resume_id, job_id, skill_score, semantic_score, experience_score,
                education_score, project_score, overall_score)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (resume_id, job_id, skill_score, semantic_score, experience_score,
             education_score, project_score, overall_score),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_match_history(user_id: int, limit: int = 50) -> List[Dict]:
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """SELECT m.id, m.overall_score, m.created_at,
                      r.filename, j.title, j.company
               FROM match_results m
               JOIN resumes r ON r.id = m.resume_id
               JOIN jobs j    ON j.id = m.job_id
               WHERE r.user_id = %s
               ORDER BY m.created_at DESC
               LIMIT %s""",
            (user_id, limit),
        )
        return cur.fetchall()
    finally:
        conn.close()


# --- Recommendations ---------------------------------------------------------

def save_recommendation(match_id: int, skill_name: str, priority: str, recommendation: str) -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO recommendations (match_id, skill_name, priority, recommendation) "
            "VALUES (%s, %s, %s, %s)",
            (match_id, skill_name, priority, recommendation),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_recommendations_for_match(match_id: int) -> List[Dict]:
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT skill_name, priority, recommendation FROM recommendations "
            "WHERE match_id = %s ORDER BY FIELD(priority, 'high','medium','low')",
            (match_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()