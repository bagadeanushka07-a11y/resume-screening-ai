"""Match page — placeholder. Real content in Phase 19f."""

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


def main() -> None:
    inject_global_styles()
    render_sidebar()
    render_header(subtitle="Match results")
    require_login()

    st.markdown(
        """
        <div class='card'>
            <div class='card-title'>🎯 Match Analysis</div>
            <p class='muted'>Coming soon — full breakdown, skill gap, and recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer()


main()