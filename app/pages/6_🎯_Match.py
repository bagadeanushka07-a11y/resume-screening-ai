"""
Match results page — full breakdown of resume vs JD.

Shows:
    - Overall score (weighted)
    - Component breakdown (5 features)
    - Skill gap (matched / missing required / missing preferred)
    - ML prediction + SHAP explanation
    - Recommendations
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.auth import require_login  # noqa: E402
from app.components.match_flow import (  # noqa: E402
    compute_shap_local,
    run_match_pipeline,
)
from app.components.navbar import render_footer, render_header  # noqa: E402
from app.components.sidebar import render_sidebar  # noqa: E402
from app.components.styles import inject_global_styles  # noqa: E402
from app.config.config import (  # noqa: E402
    ENABLE_DB_PERSISTENCE,
    SESSION_JD_PARSED,
    SESSION_JD_TEXT,
    SESSION_LAST_MATCH,
    SESSION_RESUME_PARSED,
    SESSION_RESUME_SKILLS,
    SESSION_RESUME_TEXT,
)


# --- Rendering helpers -------------------------------------------------------

def _score_color(pct: float) -> str:
    if pct >= 75:
        return "var(--success)"
    if pct >= 50:
        return "var(--warning)"
    return "var(--danger)"


def _score_label(pct: float) -> str:
    if pct >= 75:
        return "Strong match"
    if pct >= 50:
        return "Moderate match"
    return "Weak match"


def _render_score_card(score_result: dict) -> None:
    pct = score_result["overall_percent"]
    color = _score_color(pct)
    label = _score_label(pct)

    st.markdown(
        f"""
        <div class='card' style='text-align:center;'>
            <div class='muted' style='margin-bottom:0.5rem;'>Overall Match Score</div>
            <div style='font-size:clamp(2.5rem,6vw,4rem);font-weight:800;color:{color};
                        line-height:1;margin:0.25rem 0;'>{pct}%</div>
            <div style='font-size:1rem;font-weight:600;color:{color};'>{label}</div>
            <div class='score-bar' style='max-width:520px;margin:1rem auto 0 auto;'>
                <div class='score-fill' style='width:{pct}%;'></div>
            </div>
            <p class='muted' style='margin-top:1rem;font-size:0.85rem;'>
                {score_result['notes']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_breakdown(score_result: dict) -> None:
    st.markdown("### 📊 Score breakdown")
    st.markdown(
        "<div class='muted mb-2'>Each component contributes to the overall score "
        "with the weight shown.</div>",
        unsafe_allow_html=True,
    )

    labels = {
        "skill_match": "🎯 Skill Match",
        "semantic_similarity": "📝 Semantic Similarity",
        "experience_match": "⏳ Experience Match",
        "education_match": "🎓 Education Match",
        "category_match": "🏷️ Category Match",
    }

    cols = st.columns(5)
    for i, (key, label) in enumerate(labels.items()):
        with cols[i]:
            val = score_result["components"][key]
            w = score_result["weights"][key]
            pct = round(val * 100)
            st.markdown(
                f"""
                <div class='card' style='text-align:center;'>
                    <div style='font-size:0.85rem;font-weight:600;'>{label}</div>
                    <div class='card-value' style='margin:0.3rem 0;'>{pct}%</div>
                    <div class='muted' style='font-size:0.75rem;'>weight {int(w*100)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_gap(gap: dict) -> None:
    st.markdown("### 🧩 Skill gap")

    cols = st.columns(3)
    with cols[0]:
        matched = gap.get("matched", [])
        badges = "".join(f"<span class='badge badge-success'>{s}</span>" for s in matched)
        empty_msg = '<span class="muted">None</span>'
        st.markdown(
            "<div class='card'><div class='card-title'>✅ Matched ("
            + str(len(matched)) + ")</div><div>"
            + (badges or empty_msg)
            + "</div></div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        missing_req = gap.get("missing_required", [])
        badges = "".join(f"<span class='badge badge-danger'>{s}</span>" for s in missing_req)
        empty_msg = '<span class="muted">None — you cover all required skills!</span>'
        st.markdown(
            "<div class='card'><div class='card-title'>🔴 Missing required ("
            + str(len(missing_req)) + ")</div><div>"
            + (badges or empty_msg)
            + "</div></div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        missing_pref = gap.get("missing_preferred", [])
        badges = "".join(f"<span class='badge badge-warning'>{s}</span>" for s in missing_pref)
        empty_msg = '<span class="muted">None</span>'
        st.markdown(
            "<div class='card'><div class='card-title'>🟡 Missing preferred ("
            + str(len(missing_pref)) + ")</div><div>"
            + (badges or empty_msg)
            + "</div></div>",
            unsafe_allow_html=True,
        )


def _render_ml(ml: dict) -> None:
    if not ml or ml.get("probability") is None:
        return
    proba = ml["probability"]
    pred = ml.get("prediction")
    verdict = "MATCH" if pred == 1 else "NO MATCH"
    st.markdown("### 🤖 ML model prediction")
    st.markdown(
        f"""
        <div class='card'>
            <div class='card-title'>Logistic Regression · {verdict}</div>
            <div class='card-value'>{round(proba*100,1)}%</div>
            <div class='muted'>Probability of match (weak-supervised labels)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_shap(features_dict: dict) -> None:
    st.markdown("### 🧠 Why this prediction? (SHAP explanation)")
    explanation = compute_shap_local(features_dict)
    contributions = explanation.get("contributions", [])
    if not contributions:
        st.info("SHAP explanation unavailable.")
        return

    feature_labels = {
        "skill_match_score": "Overall skill overlap",
        "skill_match_required": "Required skills only",
        "experience_match": "Experience",
        "education_match": "Education",
        "category_match": "Category alignment",
    }

    html = "<div class='card'>"
    for c in contributions:
        shap = c["shap"]
        name = feature_labels.get(c["feature"], c["feature"])
        sign = "+" if shap >= 0 else "−"
        bar_width = min(100, abs(shap) * 40)  # scale for display
        color = "var(--success)" if shap > 0 else "var(--danger)"
        html += (
            f"<div style='margin-bottom:0.75rem;'>"
            f"<div style='font-size:0.85rem;font-weight:600;'>{name}</div>"
            f"<div style='display:flex;align-items:center;gap:0.5rem;margin-top:0.25rem;'>"
            f"<div style='flex:0 0 auto;color:{color};font-weight:600;"
            f"font-variant-numeric:tabular-nums;width:70px;'>{sign}{abs(shap):.3f}</div>"
            f"<div style='flex:1;height:8px;background:var(--bg-subtle);"
            f"border-radius:999px;overflow:hidden;'>"
            f"<div style='width:{bar_width}%;height:100%;background:{color};'></div>"
            f"</div></div></div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption(
        "Positive values pushed the prediction toward MATCH; "
        "negative values pushed toward NO MATCH."
    )


def _render_recommendations(recs: list) -> None:
    st.markdown("### 💡 Recommended next steps")
    if not recs:
        st.info("No recommendations — you're well aligned with this role!")
        return

    for r in recs:
        badge_class = {"high": "danger", "medium": "warning", "low": "info"}.get(r["priority"], "neutral")
        role_class = "danger" if r["role"] == "required" else "warning"
        st.markdown(
            f"""
            <div class='card'>
                <div style='display:flex;justify-content:space-between;align-items:center;
                            gap:0.5rem;flex-wrap:wrap;'>
                    <div class='card-title' style='margin:0;'>{r['skill']}</div>
                    <div>
                        <span class='badge badge-{role_class}'>{r['role']}</span>
                        <span class='badge badge-{badge_class}'>priority: {r['priority']}</span>
                    </div>
                </div>
                <p class='muted' style='margin:0.5rem 0 0 0;'>{r['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Main --------------------------------------------------------------------

def main() -> None:
    inject_global_styles()
    render_sidebar()
    render_header(subtitle="Match analysis")
    require_login()

    resume_text = st.session_state.get(SESSION_RESUME_TEXT)
    jd_text = st.session_state.get(SESSION_JD_TEXT)

    if not resume_text:
        st.warning("⚠️ Please upload your resume first.")
        if st.button("Go to Resume Upload", type="primary", use_container_width=True):
            st.switch_page("pages/4_📄_Resume.py")
        render_footer()
        return

    if not jd_text:
        st.warning("⚠️ Please paste a job description first.")
        if st.button("Go to Job Description", type="primary", use_container_width=True):
            st.switch_page("pages/5_💼_Job.py")
        render_footer()
        return

    # --- Run pipeline (cached in session if same inputs) ---
    cache_key = f"match_cache_{hash(resume_text)}_{hash(jd_text)}"
    cached = st.session_state.get(cache_key)
    if cached is None:
        with st.spinner("Running full analysis..."):
            resume_skills = st.session_state.get(SESSION_RESUME_SKILLS)
            results = run_match_pipeline(
                resume_text=resume_text,
                jd_text=jd_text,
                resume_category=None,
                resume_skills=resume_skills,
            )
            st.session_state[cache_key] = results
            st.session_state[SESSION_LAST_MATCH] = results
            cached = results

    if cached.get("error"):
        st.error(f"❌ {cached['error']}")
        render_footer()
        return

    # --- Render ---
    _render_score_card(cached["score"])
    _render_breakdown(cached["score"])
    _render_gap(cached["gap"])
    _render_ml(cached.get("ml", {}))

    if cached.get("ml", {}).get("features"):
        _render_shap(cached["ml"]["features"])

    _render_recommendations(cached.get("recommendations", []))

    # --- Save to DB (best-effort) ---
    if ENABLE_DB_PERSISTENCE:
        try:
            user_id = st.session_state.get("user_id")
            if user_id and user_id != 0:
                from src.database import queries as q
                from src.database.connection import is_available
                if is_available():
                    score = cached["score"]["components"]
                    # We don't have resume_id/job_id here — skip DB save for now
                    # (Phase 18's queries create_job/create_resume would need ids)
                    pass
        except Exception:
            pass

    # --- Navigation ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Edit job description", use_container_width=True):
            st.switch_page("pages/5_💼_Job.py")
    with col2:
        if st.button("🔄 New resume", use_container_width=True):
            st.switch_page("pages/4_📄_Resume.py")

    render_footer()


main()