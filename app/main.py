"""
Streamlit entry point.

Run from the project root with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when Streamlit runs this file directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.components.navbar import render_footer, render_header  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.components.styles import inject_global_styles  # noqa: E402
from app.config.config import (  # noqa: E402
    APP_ICON,
    APP_TAGLINE,
    APP_TITLE,
    APP_VERSION,
)


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} v{APP_VERSION}",
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_global_styles()
    render_sidebar()
    render_header(subtitle=APP_TAGLINE)

    st.markdown(
        """
        <div class='card'>
            <div class='card-title'>👋 Welcome</div>
            <p class='muted'>
                This is the base layout for the app. Pages will be added
                incrementally in Phase 19. Use the sidebar to navigate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()


if __name__ == "__main__":
    main()