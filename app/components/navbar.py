"""
Top navigation bar + helpers.

Renders a title bar at the top of every page and provides a small
helper for showing the logged-in user.
"""

from __future__ import annotations

import streamlit as st

from app.config.config import APP_ICON, APP_TITLE, SESSION_USER_NAME


def render_header(subtitle: str | None = None) -> None:
    """Render the app title header. Call near the top of every page."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        if subtitle:
            st.markdown(f"<div class='muted'>{subtitle}</div>", unsafe_allow_html=True)
    with col2:
        user_name = st.session_state.get(SESSION_USER_NAME)
        if user_name:
            st.markdown(
                f"<div style='text-align:right;padding-top:0.5rem'>"
                f"<span class='badge badge-info'>👤 {user_name}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_footer() -> None:
    """Render a small footer."""
    st.markdown("---")
    st.markdown(
        "<div class='center muted'>Built with Streamlit · "
        "ML: Logistic Regression · NLP: ESCO + Sentence Transformers</div>",
        unsafe_allow_html=True,
    )