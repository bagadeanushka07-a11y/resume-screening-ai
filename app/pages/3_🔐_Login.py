"""
Login page.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import (  # noqa: E402
    is_logged_in,
    set_logged_in,
    try_login,
)
from app.components.navbar import render_footer, render_header  # noqa: E402
from app.components.styles import inject_global_styles  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402

def main() -> None:
    inject_global_styles()
    render_sidebar()
    render_header(subtitle="Sign in to your account")

    if is_logged_in():
        st.info("You're already signed in.")
        if st.button("Go to Resume Upload", type="primary", use_container_width=True):
            st.switch_page("pages/4_📄_Resume.py")
        render_footer()
        return

    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            "<div class='card'><p class='muted mb-2'>"
            "Sign in to save analyses and view your history."
            "</p></div>",
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Sign in", type="primary", use_container_width=True
            )

        if submitted:
            if not email or not password:
                st.error("❌ Please enter both email and password.")
            else:
                success, err, user = try_login(email.strip(), password)
                if not success or not user:
                    st.error(f"❌ {err or 'Login failed.'}")
                else:
                    set_logged_in(user["id"], user["name"], user["email"])
                    st.success("✅ Signed in!")
                    st.switch_page("pages/4_📄_Resume.py")

        st.markdown(
            "<div class='center muted mt-3'>New here?</div>",
            unsafe_allow_html=True,
        )
        if st.button("Create an account", use_container_width=True):
            st.switch_page("pages/2_📝_Register.py")

    render_footer()


main()