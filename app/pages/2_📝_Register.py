"""
Register page — create a new account.
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
    try_register,
    validate_registration,
)
from app.components.navbar import render_footer, render_header  # noqa: E402
from app.components.styles import inject_global_styles  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402

def main() -> None:
    inject_global_styles()
    render_sidebar()
    render_header(subtitle="Create your account")

    if is_logged_in():
        st.info("You're already signed in.")
        if st.button("Go to Resume Upload", type="primary", use_container_width=True):
            st.switch_page("pages/4_📄_Resume.py")
        render_footer()
        return

    st.markdown(
        "<div class='card'><p class='muted mb-2'>"
        "Create an account to save your resume analyses and history."
        "</p></div>",
        unsafe_allow_html=True,
    )

    # Use a narrower column on wide screens for a cleaner form
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        with st.form("register_form", clear_on_submit=False):
            name = st.text_input("Name", placeholder="Your full name")
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password",
                                     help="At least 8 characters")
            confirm = st.text_input("Confirm password", type="password")

            submitted = st.form_submit_button(
                "Create account", type="primary", use_container_width=True
            )

        if submitted:
            ok, err = validate_registration(name, email, password, confirm)
            if not ok:
                st.error(f"❌ {err}")
            else:
                success, err, uid = try_register(name, email, password)
                if not success:
                    st.error(f"❌ {err}")
                else:
                    set_logged_in(uid or 0, name.strip(), email.strip().lower())
                    st.success("✅ Account created — welcome!")
                    st.switch_page("pages/4_📄_Resume.py")

        st.markdown(
            "<div class='center muted mt-3'>Already have an account?</div>",
            unsafe_allow_html=True,
        )
        if st.button("Sign in instead", use_container_width=True):
            st.switch_page("pages/3_🔐_Login.py")

    render_footer()


main()
