"""
Landing page — first thing a visitor sees.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app.components.navbar import render_footer  # noqa: E402
from app.components.styles import inject_global_styles  # noqa: E402
from app.config.config import (  # noqa: E402
    APP_ICON,
    APP_TAGLINE,
    APP_TITLE,
)


def _hero() -> None:
    st.markdown(
        f"""
        <div class='card' style='padding:clamp(1.2rem,4vw,2.5rem); text-align:center;'>
            <h1 style='margin-bottom:0.5rem;'>{APP_ICON} {APP_TITLE}</h1>
            <p style='font-size:clamp(1rem,2vw,1.15rem); color:var(--text-muted); margin-bottom:1.5rem;'>
                {APP_TAGLINE}
            </p>
            <p style='font-size:clamp(0.9rem,1.6vw,1rem); color:var(--text-primary); max-width:720px; margin:0 auto 1.5rem auto;'>
                Upload your resume, paste a job description, and get a transparent match score
                with explainable AI — plus a personalized skill-gap analysis and learning plan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _cta_row() -> None:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Analyze Your Resume", type="primary", use_container_width=True):
            st.switch_page("pages/4_📄_Resume.py")


def _feature_grid() -> None:
    st.markdown("### ✨ What you get")
    cols = st.columns(3)
    features = [
        ("🎯", "Transparent Match Score",
         "A weighted score across skill overlap, semantics, experience, education, and category alignment."),
        ("🧩", "Skill Gap Analysis",
         "See exactly which required and preferred skills you're missing for any role."),
        ("💡", "Personalized Recommendations",
         "Get specific next steps — not generic advice — based on your actual skill gaps."),
        ("🧠", "Explainable Predictions",
         "Every score is broken down feature-by-feature using SHAP, so you know why you matched."),
        ("📚", "Real Skill Taxonomy",
         "Powered by ESCO (14,000+ skills) and a curated catalog with proper categorization."),
        ("📱", "Works Everywhere",
         "Responsive design that runs cleanly on desktop, tablet, and mobile."),
    ]
    for i, (icon, title, body) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class='card' style='height:100%;'>
                    <div style='font-size:1.6rem;margin-bottom:0.35rem;'>{icon}</div>
                    <div class='card-title'>{title}</div>
                    <p class='muted' style='margin:0;'>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _how_it_works() -> None:
    st.markdown("### 🔄 How it works")
    steps = [
        ("1", "Upload your resume", "PDF or DOCX — we extract and analyze the text."),
        ("2", "Paste a job description", "We identify required and preferred skills."),
        ("3", "Get your score", "Weighted match score with a full explanation."),
        ("4", "Close the gaps", "Personalized recommendations for missing skills."),
    ]
    cols = st.columns(4)
    for col, (num, title, body) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class='card' style='text-align:center;'>
                    <div style='font-size:1.6rem;font-weight:700;color:var(--primary);
                                margin-bottom:0.3rem;'>{num}</div>
                    <div class='card-title'>{title}</div>
                    <p class='muted' style='margin:0;font-size:0.85rem;'>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _tech_stack() -> None:
    st.markdown("### 🛠 Powered by")
    badges = [
        ("Python", "info"), ("scikit-learn", "info"), ("XGBoost", "info"),
        ("spaCy", "info"), ("Sentence Transformers", "info"), ("SHAP", "info"),
        ("Streamlit", "neutral"), ("MySQL", "neutral"), ("ESCO", "neutral"),
    ]
    html = "".join(
        f"<span class='badge badge-{color}'>{name}</span>" for name, color in badges
    )
    st.markdown(f"<div class='card'>{html}</div>", unsafe_allow_html=True)


def main() -> None:
    inject_global_styles()
    _hero()
    _cta_row()
    st.markdown("")
    _feature_grid()
    st.markdown("")
    _how_it_works()
    st.markdown("")
    _tech_stack()
    render_footer()


main()