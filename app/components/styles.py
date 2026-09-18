"""
Responsive + theme-aware CSS injected into every page.

Works in both light and dark mode by using Streamlit's theme variables
(defined in .streamlit/config.toml) via CSS `prefers-color-scheme` and
the `[data-theme="dark"]` selector Streamlit injects.
"""

from __future__ import annotations

import streamlit as st


CSS = """
<style>
/* ---------- Design tokens: light (default) ---------- */
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
    --badge-success-bg: #D1FAE5;
    --badge-success-fg: #065F46;
    --badge-warning-bg: #FEF3C7;
    --badge-warning-fg: #92400E;
    --badge-danger-bg:  #FEE2E2;
    --badge-danger-fg:  #991B1B;
    --badge-info-bg:    #DBEAFE;
    --badge-info-fg:    #1E40AF;
    --badge-neutral-bg: #F3F4F6;
    --badge-neutral-fg: #374151;
}

/* ---------- Design tokens: dark mode ---------- */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-card: #1F2937;
        --bg-subtle: #111827;
        --text-primary: #F9FAFB;
        --text-muted: #9CA3AF;
        --border: #374151;
        --shadow: 0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
        --badge-success-bg: #064E3B;
        --badge-success-fg: #6EE7B7;
        --badge-warning-bg: #78350F;
        --badge-warning-fg: #FCD34D;
        --badge-danger-bg:  #7F1D1D;
        --badge-danger-fg:  #FCA5A5;
        --badge-info-bg:    #1E3A8A;
        --badge-info-fg:    #93C5FD;
        --badge-neutral-bg: #374151;
        --badge-neutral-fg: #E5E7EB;
    }
}

/* Streamlit also injects a data-theme attribute; honor it too */
[data-theme="dark"], html[data-theme="dark"] {
    --bg-card: #1F2937;
    --bg-subtle: #111827;
    --text-primary: #F9FAFB;
    --text-muted: #9CA3AF;
    --border: #374151;
    --shadow: 0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
    --badge-success-bg: #064E3B;
    --badge-success-fg: #6EE7B7;
    --badge-warning-bg: #78350F;
    --badge-warning-fg: #FCD34D;
    --badge-danger-bg:  #7F1D1D;
    --badge-danger-fg:  #FCA5A5;
    --badge-info-bg:    #1E3A8A;
    --badge-info-fg:    #93C5FD;
    --badge-neutral-bg: #374151;
    --badge-neutral-fg: #E5E7EB;
}

/* ---------- Global ---------- */
* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    overflow-x: hidden;
    max-width: 100vw;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: clamp(0.75rem, 3vw, 3rem);
    padding-right: clamp(0.75rem, 3vw, 3rem);
    max-width: 1300px;
}

/* ---------- Typography ---------- */
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
    color: var(--text-primary);
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
.badge-success { background: var(--badge-success-bg); color: var(--badge-success-fg); }
.badge-warning { background: var(--badge-warning-bg); color: var(--badge-warning-fg); }
.badge-danger  { background: var(--badge-danger-bg);  color: var(--badge-danger-fg); }
.badge-info    { background: var(--badge-info-bg);    color: var(--badge-info-fg); }
.badge-neutral { background: var(--badge-neutral-bg); color: var(--badge-neutral-fg); }

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
}
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

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border);
}

/* Hide Streamlit's auto "main" header in the sidebar */
[data-testid="stSidebarHeader"] {
    display: none;
}

/* ---------- Tables scroll on mobile ---------- */
[data-testid="stDataFrame"] > div {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
}

/* ---------- Charts fill width ---------- */
[data-testid="stPlotlyChart"] { width: 100% !important; }

/* ---------- Code blocks wrap ---------- */
pre, code {
    white-space: pre-wrap !important;
    word-break: break-word;
    font-size: clamp(0.75rem, 1.6vw, 0.9rem) !important;
}

/* ---------- Metrics ---------- */
[data-testid="stMetricValue"] {
    font-size: clamp(1.1rem, 2.5vw, 1.6rem) !important;
}

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }
    .card { padding: 0.85rem; }
    .card-value { font-size: 1.4rem; }
    .stButton > button { width: 100%; }
    header[data-testid="stHeader"] { height: 2.5rem; }
    [data-testid="stMetric"] { padding: 0.5rem 0; }
}
@media (max-width: 480px) {
    h1 { font-size: 1.35rem !important; }
    h2 { font-size: 1.15rem !important; }
    .card-value { font-size: 1.25rem; }
    .badge { font-size: 0.7rem; padding: 0.15rem 0.5rem; }
    .block-container { padding-left: 0.5rem; padding-right: 0.5rem; }
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
    """Inject the shared stylesheet. Call once per page after set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)