"""
Authentication helpers: validation, login/logout, session management.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import streamlit as st

from app.config.config import (
    ENABLE_DB_PERSISTENCE,
    SESSION_USER_EMAIL,
    SESSION_USER_ID,
    SESSION_USER_NAME,
)


# --- Validation --------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")


def is_valid_email(email: str) -> bool:
    """Basic email format check — not perfect, but catches obvious typos."""
    if not isinstance(email, str):
        return False
    return bool(_EMAIL_RE.match(email.strip()))


def validate_registration(
    name: str,
    email: str,
    password: str,
    confirm: str,
) -> Tuple[bool, Optional[str]]:
    """
    Return (ok, error_message). If ok is True, error_message is None.
    """
    name = (name or "").strip()
    email = (email or "").strip()
    password = password or ""
    confirm = confirm or ""

    if not name:
        return False, "Please enter your name."
    if len(name) > 80:
        return False, "Name is too long (max 80 characters)."
    if not email:
        return False, "Please enter your email."
    if not is_valid_email(email):
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if password != confirm:
        return False, "Passwords do not match."
    return True, None


# --- Session helpers ---------------------------------------------------------

def is_logged_in() -> bool:
    return SESSION_USER_ID in st.session_state


def current_user() -> Optional[dict]:
    if not is_logged_in():
        return None
    return {
        "id": st.session_state.get(SESSION_USER_ID),
        "name": st.session_state.get(SESSION_USER_NAME),
        "email": st.session_state.get(SESSION_USER_EMAIL),
    }


def set_logged_in(user_id: int, name: str, email: str) -> None:
    st.session_state[SESSION_USER_ID] = user_id
    st.session_state[SESSION_USER_NAME] = name
    st.session_state[SESSION_USER_EMAIL] = email


def logout() -> None:
    for key in [SESSION_USER_ID, SESSION_USER_NAME, SESSION_USER_EMAIL]:
        st.session_state.pop(key, None)


def require_login(redirect_to: str = "pages/3_🔐_Login.py") -> None:
    """
    Redirect to the login page if the user isn't logged in.
    Call near the top of any protected page.
    """
    if not is_logged_in():
        st.warning("🔒 Please sign in to access this page.")
        if st.button("Go to Login"):
            st.switch_page(redirect_to)
        st.stop()


# --- DB-backed actions -------------------------------------------------------

def try_register(name: str, email: str, password: str) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Attempt to register a user in the DB.
    Returns (ok, error_message, user_id).

    If ENABLE_DB_PERSISTENCE is False or DB is unreachable, we fall back to
    an "in-memory" user with a synthetic id.
    """
    from app.config.config import ENABLE_DB_PERSISTENCE
    if not ENABLE_DB_PERSISTENCE:
        # In-memory demo mode — id 0 always "works"
        return True, None, 0

    try:
        from src.database import queries as q
        from src.database.connection import is_available

        if not is_available():
            return True, None, 0  # graceful fallback

        existing = q.get_user_by_email(email)
        if existing:
            return False, "An account with this email already exists.", None

        uid = q.create_user(name, email, password)
        return True, None, uid
    except Exception as e:
        return False, f"Database error: {e}", None


def try_login(email: str, password: str) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Attempt to log in. Returns (ok, error_message, user_dict).
    """
    from app.config.config import ENABLE_DB_PERSISTENCE
    if not ENABLE_DB_PERSISTENCE:
        # Demo mode — accept any non-empty credentials
        if email and password:
            return True, None, {"id": 0, "name": email.split("@")[0], "email": email}
        return False, "Please enter your email and password.", None

    try:
        from src.database import queries as q
        from src.database.connection import is_available

        if not is_available():
            # Fallback to demo mode
            if email and password:
                return True, None, {"id": 0, "name": email.split("@")[0], "email": email}
            return False, "Please enter your email and password.", None

        user = q.get_user_by_email(email)
        if not user:
            return False, "No account found with that email.", None
        if not q.verify_password(password, user["password_hash"]):
            return False, "Incorrect password.", None

        return True, None, {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
        }
    except Exception as e:
        return False, f"Database error: {e}", None