"""
MySQL connection helper.

Reads credentials from .env via python-dotenv.
Uses mysql-connector-python.

Public API:
    - get_connection() -> mysql.connector.connection.MySQLConnection
    - is_available() -> bool (True if MySQL is reachable)
    - init_schema() -> creates all tables if they don't exist
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import mysql.connector
from dotenv import load_dotenv


# Load .env once at import time
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)


DB_CONFIG: Dict = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "resume_screening"),
    "autocommit": False,
}


def get_connection():
    """
    Return a new MySQL connection. Caller is responsible for closing it.
    Raises mysql.connector.Error if the server is unreachable.
    """
    return mysql.connector.connect(**DB_CONFIG)


def is_available() -> bool:
    """Return True if MySQL is reachable with the current config."""
    try:
        conn = get_connection()
        conn.close()
        return True
    except mysql.connector.Error:
        return False


# --- Schema ------------------------------------------------------------------

SCHEMA_STATEMENTS = [
    # --- users ---
    """
    CREATE TABLE IF NOT EXISTS users (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        name          VARCHAR(120) NOT NULL,
        email         VARCHAR(180) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_email (email)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # --- resumes ---
    """
    CREATE TABLE IF NOT EXISTS resumes (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        user_id         INT NOT NULL,
        filename        VARCHAR(255) NOT NULL,
        extracted_text  LONGTEXT NOT NULL,
        upload_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        INDEX idx_user_id (user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # --- skills (knowledge base) ---
    """
    CREATE TABLE IF NOT EXISTS skills (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        skill_name  VARCHAR(120) NOT NULL UNIQUE,
        category    VARCHAR(60) NOT NULL,
        INDEX idx_category (category)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # --- resume_skills (join) ---
    """
    CREATE TABLE IF NOT EXISTS resume_skills (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        resume_id   INT NOT NULL,
        skill_id    INT NOT NULL,
        confidence  FLOAT DEFAULT 0.0,
        FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE,
        FOREIGN KEY (skill_id)  REFERENCES skills(id)  ON DELETE CASCADE,
        UNIQUE KEY uniq_resume_skill (resume_id, skill_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # --- jobs ---
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        title       VARCHAR(180),
        company     VARCHAR(180),
        description LONGTEXT NOT NULL,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_title (title)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # --- job_skills (join) ---
    """
    CREATE TABLE IF NOT EXISTS job_skills (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        job_id      INT NOT NULL,
        skill_id    INT NOT NULL,
        required    BOOLEAN DEFAULT TRUE,
        FOREIGN KEY (job_id)   REFERENCES jobs(id)   ON DELETE CASCADE,
        FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE,
        UNIQUE KEY uniq_job_skill (job_id, skill_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # --- match_results ---
    """
    CREATE TABLE IF NOT EXISTS match_results (
        id                  INT AUTO_INCREMENT PRIMARY KEY,
        resume_id           INT NOT NULL,
        job_id              INT NOT NULL,
        skill_score         FLOAT DEFAULT 0.0,
        semantic_score      FLOAT DEFAULT 0.0,
        experience_score    FLOAT DEFAULT 0.0,
        education_score     FLOAT DEFAULT 0.0,
        project_score       FLOAT DEFAULT 0.0,
        overall_score       FLOAT DEFAULT 0.0,
        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE,
        FOREIGN KEY (job_id)    REFERENCES jobs(id)    ON DELETE CASCADE,
        INDEX idx_resume_job (resume_id, job_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,

    # --- recommendations ---
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        match_id        INT NOT NULL,
        skill_name      VARCHAR(120) NOT NULL,
        priority        VARCHAR(20) NOT NULL,
        recommendation  TEXT NOT NULL,
        FOREIGN KEY (match_id) REFERENCES match_results(id) ON DELETE CASCADE,
        INDEX idx_match_id (match_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """,
]


def init_schema() -> None:
    """
    Create all tables if they don't exist. Idempotent.
    Raises mysql.connector.Error if MySQL is unreachable or DDL fails.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        for stmt in SCHEMA_STATEMENTS:
            cursor.execute(stmt)
        conn.commit()
        cursor.close()
    finally:
        conn.close()