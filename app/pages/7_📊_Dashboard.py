"""Dashboard page — quick stats and recent activity."""

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
    ENABLE_DB_PERSISTENCE,
    SESSION_LAST_MATCH,
    SESSION_USER_NAME,
)


def _stat_card(icon: str, title: str, value: str, subtitle: str = "") -> str:
    return f"""
        <div class='card' style='text-align:center;'>
            <div style='font-size:1.6rem;margin-bottom:0.25rem;'>{icon}</div>
            <div class='card-title'>{title}</div>
            <div class='card-value'>{value}</div>
            <div class='card-subtitle'>{subtitle}</div>
        </div>
    """


def _fetch_stats() -> dict:
    """Best-effort stats fetch. Falls back to zeros."""
    default = {
        "total_resumes": 0,
        "total_jobs_analyzed": 0,
        "total_matches": 0,
        "avg_match_score": 0.0,
        "source": "none",
    }
    if not ENABLE_DB_PERSISTENCE:
        return default
    try:
        from src.database import queries as q
        from src.database.connection import is_available
        user_id = st.session_state.get("user_id")
        if not user_id or user_id == 0 or not is_available():
            return default
        stats = q.get_user_stats(user_id)
        stats["source"] = "db"
        return stats
    except Exception:
        return default


def _fetch_recent_matches(limit: int = 10) -> list:
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
    render_header(subtitle="Your activity at a glance")
    require_login()

    name = st.session_state.get(SESSION_USER_NAME, "there")

    # --- Stats ---
    stats = _fetch_stats()

    st.markdown(f"### 👋 Welcome back, {name}!")
    st.markdown("")

    cols = st.columns(4)
    with cols[0]:
        st.markdown(_stat_card("📄", "Resumes analyzed", str(stats["total_resumes"])),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_stat_card("💼", "Jobs analyzed", str(stats["total_jobs_analyzed"])),
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(_stat_card("🎯", "Total matches", str(stats["total_matches"])),
                    unsafe_allow_html=True)
    with cols[3]:
        avg = stats["avg_match_score"]
        st.markdown(_stat_card("📊", "Avg. score", f"{avg:.1f}%"),
                    unsafe_allow_html=True)

    # --- Last match (from session) ---
    last = st.session_state.get(SESSION_LAST_MATCH)
    if last and not last.get("error"):
        st.markdown("### 🔥 Latest analysis")
        score = last.get("score", {}).get("overall_percent", 0)
        gap = last.get("gap", {})
        matched = len(gap.get("matched", []))
        missing = len(gap.get("missing_required", []))
        st.markdown(
            f"""
            <div class='card'>
                <div style='display:flex;justify-content:space-between;align-items:center;
                            flex-wrap:wrap;gap:1rem;'>
                    <div>
                        <div class='card-title'>Overall Match Score</div>
                        <div class='card-value'>{score}%</div>
                    </div>
                    <div style='display:flex;gap:0.5rem;flex-wrap:wrap;'>
                        <span class='badge badge-success'>✅ {matched} skills matched</span>
                        <span class='badge badge-danger'>🔴 {missing} missing</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View full analysis", type="primary", use_container_width=True):
            st.switch_page("pages/6_🎯_Match.py")

    # --- Recent matches ---
    st.markdown("### 📜 Recent analyses")
    recent = _fetch_recent_matches(limit=10)
    if not recent:
        st.info(
            "No saved history yet. Analyses are saved to MySQL when "
            "the database is connected and you're logged in."
        )
    else:
        for h in recent:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{h.get('filename', '—')}**")
                st.caption(f"{h.get('title', '—')} @ {h.get('company', '—')}")
            with col2:
                score = h.get("overall_score", 0)
                pct = round((score or 0) * 100, 1)
                st.markdown(f"<span class='badge badge-info'>{pct}%</span>",
                            unsafe_allow_html=True)
            with col3:
                st.caption(f"{h.get('created_at', '')}"[:16])

    # --- Quick actions ---
    st.markdown("### ⚡ Quick actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Upload new resume", use_container_width=True):
            st.switch_page("pages/4_📄_Resume.py")
    with col2:
        if st.button("💼 Match with a job", use_container_width=True):
            st.switch_page("pages/5_💼_Job.py")

    render_footer()


main()