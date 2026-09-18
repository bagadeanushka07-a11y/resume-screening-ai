"""
Resume upload + analysis page.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import require_login  # noqa: E402
from app.components.file_upload import (  # noqa: E402
    extract_contact_info,
    process_uploaded_resume,
    render_resume_uploader,
)
from app.components.navbar import render_footer, render_header  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.components.styles import inject_global_styles  # noqa: E402
from app.config.config import (  # noqa: E402
    ENABLE_DB_PERSISTENCE,
    SESSION_RESUME_FILENAME,
    SESSION_RESUME_PARSED,
    SESSION_RESUME_SKILLS,
    SESSION_RESUME_TEXT,
)


def _render_parsed_preview(parsed: dict) -> None:
    """Show the parsed resume with text preview, contact info, and skills."""
    st.markdown("### ✅ Resume analyzed")
    st.markdown(
        f"""
        <div class='card'>
            <div class='card-title'>📄 {parsed['filename']}</div>
            <div class='card-subtitle'>
                {parsed['num_words']:,} words · {parsed['num_chars']:,} characters
                · {parsed['size_bytes'] / 1024:.1f} KB
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Contact info ---
    info = extract_contact_info(parsed["text"])
    cols = st.columns(3)
    with cols[0]:
        name = info["name"] or "—"
        st.markdown(
            f"<div class='card'><div class='card-title'>👤 Name</div>"
            f"<div class='muted'>{name}</div></div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        email = info["email"] or "—"
        st.markdown(
            f"<div class='card'><div class='card-title'>✉️ Email</div>"
            f"<div class='muted'>{email}</div></div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        phone = info["phone"] or "—"
        st.markdown(
            f"<div class='card'><div class='card-title'>📞 Phone</div>"
            f"<div class='muted'>{phone}</div></div>",
            unsafe_allow_html=True,
        )

    # --- Skills ---
    st.markdown("### 🧩 Extracted skills")

    skills = st.session_state.get(SESSION_RESUME_SKILLS) or []
    if not skills:
        st.info(
            "No skills were detected. This can happen with non-technical resumes, "
            "or resumes with unusual formatting. You can still proceed to job matching."
        )
    else:
        # Group by category
        grouped: dict = {}
        for s in skills:
            grouped.setdefault(s["category"], []).append(s)

        # Sort categories: known ones first, then "Other"
        known = sorted([c for c in grouped if c != "Other"])
        cats = known + (["Other"] if "Other" in grouped else [])

        st.markdown(
            f"<div class='muted mb-2'>Detected "
            f"<b>{len(skills)}</b> skills across <b>{len(cats)}</b> categories.</div>",
            unsafe_allow_html=True,
        )

        # Display as category cards (2 columns on desktop, 1 on mobile)
        cols = st.columns(2)
        for i, cat in enumerate(cats):
            items = sorted(grouped[cat], key=lambda x: -x["confidence"])
            badge_html = "".join(
                f"<span class='badge badge-{'success' if cat != 'Other' else 'neutral'}'>"
                f"{s['skill']}</span>"
                for s in items
            )
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class='card'>
                        <div class='card-title'>{cat} ({len(items)})</div>
                        <div>{badge_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --- Raw text preview ---
    with st.expander("📖 View extracted text (first 3000 characters)"):
        preview = parsed["text"][:3000]
        st.text(preview)
        if len(parsed["text"]) > 3000:
            st.caption(f"...and {len(parsed['text']) - 3000:,} more characters")


def _extract_skills_safe(text: str) -> list:
    """Wrapper around extract_resume_skills that never throws."""
    try:
        from src.nlp.skill_extractor import extract_resume_skills
        return extract_resume_skills(text) or []
    except Exception:
        return []


def main() -> None:
    inject_global_styles()
    render_sidebar()
    render_header(subtitle="Upload your resume to begin")

    require_login()

    # --- Already-uploaded state ---
    parsed_existing = st.session_state.get(SESSION_RESUME_PARSED)

    # --- Uploader ---
    uploaded = render_resume_uploader()

    # --- Auto-process on upload ---
    if uploaded is not None:
        # Only reprocess if filename changed
        last_filename = parsed_existing["filename"] if parsed_existing else None
        if uploaded.name != last_filename:
            with st.spinner("Analyzing resume..."):
                ok, err, parsed = process_uploaded_resume(uploaded)
                if not ok:
                    st.error(f"❌ {err}")
                else:
                    # Extract skills
                    skills = _extract_skills_safe(parsed["text"])

                    # Save to session
                    st.session_state[SESSION_RESUME_TEXT] = parsed["text"]
                    st.session_state[SESSION_RESUME_FILENAME] = parsed["filename"]
                    st.session_state[SESSION_RESUME_PARSED] = parsed
                    st.session_state[SESSION_RESUME_SKILLS] = skills

                    # Persist to DB (best-effort)
                    if ENABLE_DB_PERSISTENCE:
                        try:
                            from src.database import queries as q
                            from src.database.connection import is_available
                            user_id = st.session_state.get("user_id")
                            if user_id and user_id != 0 and is_available():
                                q.create_resume(user_id, parsed["filename"], parsed["text"])
                        except Exception:
                            pass  # don't fail the UI if DB write fails

                    st.success(
                        f"✅ Analyzed **{parsed['filename']}** — "
                        f"{len(skills)} skills detected."
                    )
                    st.rerun()

    # --- Show parsed result if present ---
    parsed_now = st.session_state.get(SESSION_RESUME_PARSED)
    if parsed_now:
        _render_parsed_preview(parsed_now)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear resume", use_container_width=True):
                for key in [
                    SESSION_RESUME_TEXT,
                    SESSION_RESUME_FILENAME,
                    SESSION_RESUME_PARSED,
                    SESSION_RESUME_SKILLS,
                ]:
                    st.session_state.pop(key, None)
                st.rerun()
        with col2:
            if st.button("➡️ Continue to Job Description", type="primary", use_container_width=True):
                st.switch_page("pages/5_💼_Job.py")

    render_footer()


main()