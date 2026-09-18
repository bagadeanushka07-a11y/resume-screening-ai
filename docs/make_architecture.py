"""
Generate a clean architecture diagram for the README.

Run:  python docs/make_architecture.py
Output: docs/architecture.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUTPUT = Path(__file__).parent / "architecture.png"


# Colors
UI_COLOR = "#4F46E5"
APP_COLOR = "#06B6D4"
DATA_COLOR = "#10B981"
ML_COLOR = "#F59E0B"
DB_COLOR = "#8B5CF6"
TEXT_DARK = "#111827"
TEXT_LIGHT = "#FFFFFF"
BG_CARD = "#F9FAFB"


def box(ax, x, y, w, h, text, color, fontsize=10, text_color=TEXT_LIGHT, bold=False):
    """Draw a rounded box with centered text."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold" if bold else "normal",
        wrap=True,
    )


def group_label(ax, x, y, text, color):
    ax.text(
        x, y, text,
        ha="left", va="center",
        fontsize=11, fontweight="bold", color=color,
    )


def arrow(ax, x1, y1, x2, y2, color="#6B7280"):
    """Draw an arrow."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5,
        color=color,
        alpha=0.7,
    )
    ax.add_patch(arr)


def make_diagram():
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_facecolor("#FFFFFF")

    # Title
    ax.text(
        7, 10.6, "AI Resume Screening — Architecture",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=TEXT_DARK,
    )

    # ============================================================
    # Layer 1 — Frontend (Streamlit UI)
    # ============================================================
    group_label(ax, 0.3, 10.05, "① FRONTEND — Streamlit UI", UI_COLOR)
    y_ui = 9.2
    h_ui = 0.75
    w_ui = 1.95
    gap = 0.1
    x_start = 0.5

    ui_pages = [
        ("Home", "🏠"),
        ("Register", "📝"),
        ("Login", "🔐"),
        ("Resume", "📄"),
        ("Job", "💼"),
        ("Match", "🎯"),
    ]
    for i, (name, icon) in enumerate(ui_pages):
        x = x_start + i * (w_ui + gap)
        box(ax, x, y_ui, w_ui, h_ui, f"{icon}\n{name}", UI_COLOR, fontsize=9)

    ui_pages2 = [
        ("Dashboard", "📊"),
        ("History", "📜"),
        ("Analytics", "📈"),
    ]
    for i, (name, icon) in enumerate(ui_pages2):
        x = x_start + i * (w_ui + gap)
        box(ax, x, y_ui - 0.95, w_ui, h_ui, f"{icon}\n{name}", UI_COLOR, fontsize=9)

    # ============================================================
    # Layer 2 — Application Logic
    # ============================================================
    arrow(ax, 7, y_ui - 1.0, 7, 7.9)
    group_label(ax, 0.3, 7.9, "② APPLICATION LAYER — Core Logic", APP_COLOR)

    y_app = 7.05
    h_app = 0.7
    w_app = 2.28
    app_modules = [
        ("Document\nParser", "PDF/DOCX → text"),
        ("NLP\nPreprocessing", "clean · tokenize"),
        ("Skill\nExtractor", "ESCO + curated"),
        ("JD\nParser", "title · skills · exp"),
        ("Similarity\nMatcher", "TF-IDF + MiniLM"),
    ]
    for i, (title, sub) in enumerate(app_modules):
        x = x_start + i * (w_app + gap)
        box(ax, x, y_app, w_app, h_app, title, APP_COLOR, fontsize=9, bold=True)

    y_app2 = 6.05
    app_modules2 = [
        ("Skill Gap\nAnalyzer", "matched · missing"),
        ("Scoring\nEngine", "5-weighted score"),
        ("Recommendation\nEngine", "curated paths"),
        ("SHAP\nExplainer", "why this prediction"),
        ("MySQL\nQueries", "CRUD + history"),
    ]
    for i, (title, sub) in enumerate(app_modules2):
        x = x_start + i * (w_app + gap)
        box(ax, x, y_app2, w_app, h_app, title, APP_COLOR, fontsize=9, bold=True)

    # ============================================================
    # Layer 3 — Machine Learning
    # ============================================================
    arrow(ax, 7, y_app2 - 0.1, 7, 5.05)
    group_label(ax, 0.3, 5.05, "③ MACHINE LEARNING PIPELINE", ML_COLOR)

    y_ml = 4.2
    h_ml = 0.7
    w_ml = 2.28
    ml_steps = [
        ("Dataset\nBuild", "weak supervision"),
        ("Feature\nEngineering", "5 features"),
        ("SMOTE\nBalancing", "train fold only"),
        ("Model\nTraining", "LR · RF · XGBoost"),
        ("Evaluation", "ROC · F1 · CM"),
    ]
    for i, (title, sub) in enumerate(ml_steps):
        x = x_start + i * (w_ml + gap)
        box(ax, x, y_ml, w_ml, h_ml, title, ML_COLOR, fontsize=9, bold=True,
            text_color="#FFFFFF")

    # ============================================================
    # Layer 4 — Data & Artifacts
    # ============================================================
    arrow(ax, 7, y_ml - 0.1, 7, 3.15)
    group_label(ax, 0.3, 3.15, "④ DATA & MODELS", DATA_COLOR)

    y_data = 2.3
    h_data = 0.7
    w_data = 3.05
    data_items = [
        ("Resume Dataset", "2,484 resumes · 25 categories"),
        ("Job Postings", "5,000 LinkedIn JDs"),
        ("ESCO Skills KB", "13,960 skills · 99,624 aliases"),
        ("Trained Models", "LogReg · RF · XGBoost (.joblib)"),
    ]
    for i, (title, sub) in enumerate(data_items):
        x = x_start + i * (w_data + gap)
        box(ax, x, y_data, w_data, h_data, f"{title}\n{sub}",
            DATA_COLOR, fontsize=8, bold=True)

    # ============================================================
    # Layer 5 — Database
    # ============================================================
    arrow(ax, 7, y_data - 0.1, 7, 1.35)
    group_label(ax, 0.3, 1.35, "⑤ PERSISTENCE — MySQL 8 (local) / Stateless (cloud)",
                DB_COLOR)

    y_db = 0.5
    h_db = 0.7
    w_db = 2.28
    db_items = [
        "users", "resumes", "jobs", "match_results", "recommendations"
    ]
    for i, name in enumerate(db_items):
        x = x_start + i * (w_db + gap)
        box(ax, x, y_db, w_db, h_db, f"🗄️ {name}", DB_COLOR, fontsize=9, bold=True)

    # Save
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    make_diagram()