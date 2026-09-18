"""
Responsive CSS injected into every page.

Keeps the app looking like a modern SaaS dashboard on desktop / tablet / mobile.
Uses CSS clamp() and media queries so it works at any viewport width.
"""

from __future__ import annotations

import streamlit as st


CSS = """
<style>
/* ---------- Global ---------- */
:root {
    --primary: #4F46E5;
    --primary-dark: #3730A3;
    --accent: #06B6D4;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --bg-card: #FFFFFF;
    --bg-subtle: #F9FAFB;
    --text-primary: #111827;
    --text-muted: #6B7280;
    --border: #E5E7EB;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}

/* Main block padding — smaller on mobile */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: clamp(0.75rem, 3vw, 3rem);
    padding-right: clamp(0.75rem, 3vw, 3rem);
    max-width: 1300px;
}

/* Typography — clamp scales smoothly with viewport */
h1 { font-size: clamp(1.6rem, 4vw, 2.4rem) !important; line-height: 1.2 !important; }
h2 { font-size: clamp(1.3rem, 3vw, 1.8rem) !important; line-height: 1.25 !important; }
h3 { font-size: clamp(1.1rem, 2.5vw, 1.4rem) !important; }
p, li { font-size: clamp(0.9rem, 1.6vw, 1rem) !important; }

/* ---------- Cards ---------- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: clamp(0.9rem, 2vw, 1.4rem);
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

.card-title {
    font-weight: 600;
    font-size: clamp(0.95rem, 1.6vw, 1.05rem);
    color: var(--text-primary);
    margin-bottom: 0.35rem;
}

.card-value {
    font-weight: 700;
    font-size: clamp(1.4rem, 3vw, 2rem);
    color: var(--primary);
    line-height: 1.1;
}

.card-subtitle {
    color: var(--text-muted);
    font-size: clamp(0.78rem, 1.4vw, 0.9rem);
    margin-top: 0.2rem;
}

/* ---------- Badges ---------- */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem 0.2rem 0.15rem 0;
    white-space: nowrap;
}
.badge-success { background: #D1FAE5; color: #065F46; }
.badge-warning { background: #FEF3C7; color: #92400E; }
.badge-danger  { background: #FEE2E2; color: #991B1B; }
.badge-info    { background: #DBEAFE; color: #1E40AF; }
.badge-neutral { background: #F3F4F6; color: #374151; }

/* ---------- Progress bar ---------- */
.score-bar {
    height: 12px;
    border-radius: 999px;
    background: var(--bg-subtle);
    overflow: hidden;
    margin-top: 0.5rem;
}
.score-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    transition: width 0.4s ease;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    min-height: 42px;
    transition: transform 0.05s ease;
}
.stButton > button:active { transform: scale(0.98); }

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--primary-dark) !important;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border);
    border-radius: var(--radius);
    padding: 0.75rem;
    background: var(--bg-subtle);
}
[data-testid="stFileUploader"] label { font-weight: 600; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--bg-subtle);
    border-right: 1px solid var(--border);
}

/* ---------- Tables ---------- */
[data-testid="stDataFrame"] {
    border-radius: var(--radius);
    overflow-x: auto;
}

/* ---------- Metrics ---------- */
[data-testid="stMetricValue"] {
    font-size: clamp(1.1rem, 2.5vw, 1.6rem) !important;
}

/* ---------- Responsive tweaks ---------- */
@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 0.6rem;
        padding-right: 0.6rem;
    }
    .card { padding: 0.85rem; }
    .card-value { font-size: 1.4rem; }
    /* Buttons full-width on mobile */
    .stButton > button { width: 100%; }
}

@media (max-width: 480px) {
    h1 { font-size: 1.35rem !important; }
    h2 { font-size: 1.15rem !important; }
    .card-value { font-size: 1.25rem; }
    .badge { font-size: 0.7rem; padding: 0.15rem 0.5rem; }
}

/* ---------- Utility ---------- */
.muted { color: var(--text-muted); font-size: 0.88rem; }
.center { text-align: center; }
.mt-1 { margin-top: 0.5rem; }
.mt-2 { margin-top: 1rem; }
.mt-3 { margin-top: 1.5rem; }
.mb-1 { margin-bottom: 0.5rem; }
.mb-2 { margin-bottom: 1rem; }
.mb-3 { margin-bottom: 1.5rem; }
</style>
"""


def inject_global_styles() -> None:
    """Inject the shared stylesheet. Call once per page, right after set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)