"""History page — full list of past analyses."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.auth import require_login  # noqa: E402
from app.components.navbar import render_footer, render_header  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.components.styles import inject_global_styles  # noqa: E402
from app.config.config import ENABLE_DB_PERSISTENCE  # noqa: E402


def _fetch_full_history(limit: int = 200) -> list:
    if not ENABLE_DB_PERSISTENCE:
        return []
    try:
        from src.database import queries as q
        from src.database.connection import is_available
        user_id = st.session_state.get("user_id")
        if not user_id or user_id == 0 or not is_available():
            return []
        return q.get_match_history(user_id, limit=limit) or []
    except Exception:
        return []


def main() -> None:
    inject_global_styles()
    render_sidebar()
    render_header(subtitle="All your past analyses")
    require_login()

    history = _fetch_full_history()

    if not history:
        st.info(
            "No history yet. Once you analyze resumes against jobs, "
            "they'll appear here — provided MySQL is connected and "
            "you're signed in with a real account."
        )
        if st.button("Go to Resume Upload", type="primary", use_container_width=True):
            st.switch_page("pages/4_📄_Resume.py")
        render_footer()
        return

    # --- DataFrame for a sortable table ---
    df = pd.DataFrame(history)
    # Compute % score
    df["score_percent"] = (df["overall_score"].fillna(0) * 100).round(1)
    # Reorder columns for display
    df_display = df[[
        "created_at", "filename", "title", "company", "score_percent"
    ]].rename(columns={
        "created_at": "Date",
        "filename": "Resume",
        "title": "Job Title",
        "company": "Company",
        "score_percent": "Score (%)",
    })
    df_display = df_display.sort_values("Date", ascending=False)

    st.markdown(f"### 📜 {len(df_display)} analyses")

    # Filter by score
    min_score = st.slider("Minimum score (%)", 0, 100, 0, step=5)
    filtered = df_display[df_display["Score (%)"] >= min_score]

    st.markdown(f"Showing **{len(filtered)}** analyses.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # --- CSV download ---
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV",
        data=csv,
        file_name="match_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    render_footer()


main()