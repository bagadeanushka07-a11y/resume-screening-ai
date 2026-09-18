"""
File upload widget + resume processing helpers.
"""

from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import streamlit as st

from app.config.config import (
    DEFAULT_MAX_UPLOAD_MB,
    DEFAULT_SUPPORTED_EXTENSIONS,
)


def render_resume_uploader() -> Optional[object]:
    """
    Render the file uploader. Returns the uploaded file object or None.

    Streamlit's uploader accepts both drag-and-drop and click-to-browse,
    and works on mobile.
    """
    st.markdown(
        f"<div class='muted mb-1'>"
        f"Accepted: PDF, DOCX &nbsp;•&nbsp; Max size: {DEFAULT_MAX_UPLOAD_MB} MB"
        f"</div>",
        unsafe_allow_html=True,
    )
    return st.file_uploader(
        "Upload your resume",
        type=[ext.lstrip(".") for ext in DEFAULT_SUPPORTED_EXTENSIONS],
        accept_multiple_files=False,
        key="resume_uploader",
    )


def process_uploaded_resume(uploaded_file) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Parse the uploaded resume with the document parser.

    Returns:
        (ok, error_message, parsed)
        where parsed = {
            "filename": str,
            "size_bytes": int,
            "text": str,
            "num_chars": int,
            "num_words": int,
        }
    """
    if uploaded_file is None:
        return False, "No file uploaded.", None

    filename = uploaded_file.name
    size = uploaded_file.size

    # Size check
    max_bytes = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    if size > max_bytes:
        return False, f"File is too large ({size / 1024 / 1024:.1f} MB). Max is {DEFAULT_MAX_UPLOAD_MB} MB.", None

    # Parse
    try:
        from src.preprocessing.document_parser import parse_document

        # Wrap in BytesIO so parser reads from a file-like object
        uploaded_file.seek(0)
        buffer = io.BytesIO(uploaded_file.read())
        text = parse_document(buffer, filename=filename)
    except ValueError as e:
        return False, f"Could not read the file: {e}", None
    except Exception as e:
        return False, f"Unexpected error while reading the file: {e}", None

    if not text or not text.strip():
        return False, "The file has no readable text. If it's a scanned PDF, please upload a text-based version.", None

    parsed = {
        "filename": filename,
        "size_bytes": size,
        "text": text,
        "num_chars": len(text),
        "num_words": len(text.split()),
    }
    return True, None, parsed


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """
    Best-effort extraction of name, email, and phone from resume text.
    Not perfect — resumes are noisy. Returns None for fields we can't find.
    """
    import re

    info = {"name": None, "email": None, "phone": None}
    if not isinstance(text, str):
        return info

    # Email — standard pattern
    email_match = re.search(r"\b[\w\.\-\+]+@[\w\.\-]+\.\w{2,}\b", text)
    if email_match:
        info["email"] = email_match.group(0)

    # Phone — allows +country code, spaces, dashes, parens
    phone_match = re.search(
        r"(?:(?:\+|00)\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{3,5}[\s\-\.]?\d{4,6}",
        text,
    )
    if phone_match:
        candidate = phone_match.group(0).strip()
        # Only accept if it looks phone-like (at least 8 digits total)
        digits = re.sub(r"\D", "", candidate)
        if 8 <= len(digits) <= 15:
            info["phone"] = candidate

    # Name — first non-empty line of text, cleaned up
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        first_line = lines[0]
        # Skip if it looks like a header or contains special chars
        if len(first_line) <= 60 and not any(c.isdigit() for c in first_line):
            # Strip common prefixes
            cleaned = re.sub(r"^(resume|curriculum vitae|cv)\s*[:\-]?\s*", "", first_line, flags=re.IGNORECASE)
            if cleaned and len(cleaned.split()) <= 6:
                info["name"] = cleaned

    return info


def safe_rerun():
    """Wrapper for st.rerun that handles old Streamlit versions gracefully."""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()