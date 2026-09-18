"""
Job Description page — paste a JD, see parsed fields and match score.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import require_login  # noqa: E402
from app.components.navbar import render_footer, render_header  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.components.styles import inject_global_styles  # noqa: E402
from app.config.config import (  # noqa: E402
    SESSION_JD_PARSED,
    SESSION_JD_TEXT,
    SESSION_RESUME_PARSED,
    SESSION_RESUME_TEXT,
)


SAMPLE_JD = """Senior Python Engineer

We are hiring a Senior Python Engineer to join our backend team.

Required skills:
- 5+ years Python experience
- Strong knowledge of Django, PostgreSQL
- Docker and AWS required
- Experience building REST APIs

Preferred skills:
- Kubernetes experience is a plus
- Familiarity with Redis
- Bachelor's degree in Computer Science or equivalent
"""


def _parse_jd_safe(text: str) -> dict:
    try:
        from src.nlp.jd_parser import parse_jd
        return parse_jd(text)
    except Exception as e:
        st.error(f"Could not parse JD: {e}")
        return {}


def _render_parsed_jd(parsed: dict) -> None:
    """Show extracted JD fields."""
    st.markdown("### ✅ Job description analyzed")

    # --- Title / experience / education ---
    cols = st.columns(3)
    with cols[0]:
        title = parsed.get("title") or "—"
        st.markdown(
            f"<div class='card'><div class='card-title'>💼 Title</div>"
            f"<div class='muted'>{title}</div></div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        yrs = parsed.get("experience_years")
        st.markdown(
            f"<div class='card'><div class='card-title'>⏳ Experience</div>"
            f"<div class='muted'>{yrs}+ years" if yrs else "—" + "</div></div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        edu = parsed.get("education") or "—"
        st.markdown(
            f"<div class='card'><div class='card-title'>🎓 Education</div>"
            f"<div class='muted'>{edu}</div></div>",
            unsafe_allow_html=True,
        )

    # --- Skills ---
    skills = parsed.get("skills", {})
    required = skills.get("required", [])
    preferred = skills.get("preferred", [])

    st.markdown("### 🧩 JD skills")

    col1, col2 = st.columns(2)
    with col1:
        badge_html = "".join(
            f"<span class='badge badge-danger'>{s['skill']}</span>" for s in required
        )
        st.markdown(
            f"<div class='card'><div class='card-title'>🔴 Required ({len(required)})</div>"
            f"<div>{badge_html or '<span class=muted>None detected</span>'}</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        badge_html = "".join(
            f"<span class='badge badge-warning'>{s['skill']}</span>" for s in preferred
        )
        st.markdown(
            f"<div class='card'><div class='card-title'>🟡 Preferred ({len(preferred)})</div>"
            f"<div>{badge_html or '<span class=muted>None detected</span>'}</div></div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_global_styles()
    render_sidebar()
    render_header(subtitle="Paste the job description to compare")
    require_login()

    # Warn if resume not yet uploaded
    if not st.session_state.get(SESSION_RESUME_TEXT):
        st.warning("⚠️ Please upload your resume first.")
        if st.button("Go to Resume Upload", type="primary", use_container_width=True):
            st.switch_page("pages/4_📄_Resume.py")
        render_footer()
        return

    # --- Input area ---
    st.markdown(
        """
        <div class='card'>
            <div class='card-title'>💼 Job Description</div>
            <p class='muted mb-2'>
                Paste a job description below, or click "Load sample" to try.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📋 Load sample JD", use_container_width=True):
            st.session_state[SESSION_JD_TEXT] = SAMPLE_JD
            st.rerun()

    jd_text = st.text_area(
        "Job description text",
        value=st.session_state.get(SESSION_JD_TEXT, ""),
        height=280,
        placeholder="Paste the full job description here...",
        key="jd_input",
    )
    st.session_state[SESSION_JD_TEXT] = jd_text

    # --- Analyze button ---
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🔍 Analyze JD", type="primary", use_container_width=True):
            if not jd_text or len(jd_text.strip()) < 30:
                st.error("❌ Please paste a longer job description (at least 30 characters).")
            else:
                with st.spinner("Parsing job description..."):
                    parsed = _parse_jd_safe(jd_text)
                    if parsed:
                        st.session_state[SESSION_JD_PARSED] = parsed
                        st.rerun()
    with col_b:
        if st.button("🎯 Match now", use_container_width=True):
            if not jd_text or len(jd_text.strip()) < 30:
                st.error("❌ Please paste a longer job description.")
            else:
                with st.spinner("Parsing and matching..."):
                    parsed = _parse_jd_safe(jd_text)
                    if parsed:
                        st.session_state[SESSION_JD_PARSED] = parsed
                        st.switch_page("pages/6_🎯_Match.py")

    # --- Show parsed result if present ---
    parsed_now = st.session_state.get(SESSION_JD_PARSED)
    if parsed_now:
        st.markdown("---")
        _render_parsed_jd(parsed_now)

        st.markdown("---")
        if st.button("➡️ View full match analysis", type="primary", use_container_width=True):
            st.switch_page("pages/6_🎯_Match.py")

    render_footer()


main()