"""
Sidebar with user context and quick navigation.

On mobile, Streamlit collapses the sidebar into a hamburger menu, so
we keep it minimal and touch-friendly.
"""

from __future__ import annotations

import streamlit as st

from app.config.config import (
    SESSION_USER_ID,
    SESSION_USER_NAME,
    SESSION_USER_EMAIL,
)


def render_sidebar() -> None:
    """Render the app sidebar."""
    with st.sidebar:
        st.markdown("### 📄 AI Resume Screening")

        user_id = st.session_state.get(SESSION_USER_ID)
        if user_id:
            name = st.session_state.get(SESSION_USER_NAME, "User")
            email = st.session_state.get(SESSION_USER_EMAIL, "")
            st.markdown(
                f"<div class='card'>"
                f"<div class='card-title'>👤 {name}</div>"
                f"<div class='muted'>{email}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("🚪 Log out", use_container_width=True):
                for key in [
                    SESSION_USER_ID,
                    SESSION_USER_NAME,
                    SESSION_USER_EMAIL,
                ]:
                    st.session_state.pop(key, None)
                st.rerun()
        else:
            st.markdown(
                "<div class='muted mb-2'>Not signed in</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Quick tips**")
        st.markdown(
            "<div class='muted'>"
            "• Upload a resume in PDF or DOCX<br>"
            "• Paste or type a job description<br>"
            "• Get a score, skill gap, and tips"
            "</div>",
            unsafe_allow_html=True,
        )