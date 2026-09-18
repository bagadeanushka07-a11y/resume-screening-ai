"""
Streamlit entry point.

Run from the project root:
    streamlit run app/main.py

This entry file immediately redirects to the Home page.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.styles import inject_global_styles  # noqa: E402
from app.config.config import APP_ICON, APP_TITLE, APP_VERSION  # noqa: E402


st.set_page_config(
    page_title=f"{APP_TITLE} v{APP_VERSION}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="auto",
)
inject_global_styles()

# Redirect to the Home page on first load
st.switch_page("pages/1_🏠_Home.py")