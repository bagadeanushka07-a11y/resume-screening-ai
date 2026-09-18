"""
Central configuration for the Streamlit app.

Everything that might change between environments lives here:
    - App metadata
    - Session keys
    - Paths
    - Feature flags
"""

from __future__ import annotations

import os
from pathlib import Path


# --- Paths -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# --- App metadata ------------------------------------------------------------
APP_TITLE = "AI Resume Screening"
APP_ICON = "📄"
APP_TAGLINE = "Match your resume to any job with explainable ML"
APP_VERSION = "1.0.0"


# --- Session keys (constant strings for st.session_state) --------------------
SESSION_USER_ID = "user_id"
SESSION_USER_NAME = "user_name"
SESSION_USER_EMAIL = "user_email"
SESSION_RESUME_TEXT = "resume_text"
SESSION_RESUME_FILENAME = "resume_filename"
SESSION_RESUME_SKILLS = "resume_skills"
SESSION_RESUME_PARSED = "resume_parsed"
SESSION_JD_TEXT = "jd_text"
SESSION_JD_PARSED = "jd_parsed"
SESSION_LAST_MATCH = "last_match"


# --- Feature flags -----------------------------------------------------------
ENABLE_REGISTRATION = True
ENABLE_DB_PERSISTENCE = True   # set False to run without MySQL


# --- Defaults ----------------------------------------------------------------
DEFAULT_MAX_UPLOAD_MB = 10
DEFAULT_SUPPORTED_EXTENSIONS = [".pdf", ".docx"]