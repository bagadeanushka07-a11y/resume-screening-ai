"""
Build a labeled (resume, JD) pair dataset via weak supervision.

v4 — FINAL:
    - Fixes NaN crash on resume_text
    - Persists JD parse cache and resume skill cache to disk
    - Subsequent runs load caches instantly (~30 sec total)
    - Auto-calibrated TF-IDF thresholds (quantile-based)

Cache files:
    data/processed/jd_parse_cache.json
    data/processed/resume_skills_cache.json
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.nlp.jd_parser import parse_jd
from src.nlp.skill_extractor import extract_resume_skills
from src.nlp.tfidf_matcher import compute_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESUMES_PATH = PROJECT_ROOT / "data" / "processed" / "resumes_clean.csv"
POSTINGS_PATH = PROJECT_ROOT / "data" / "processed" / "postings_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
JD_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "jd_parse_cache.json"
RESUME_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "resume_skills_cache.json"


# --- Configuration ----------------------------------------------------------
RESUME_CAP = 800
JD_CAP = 800
N_PAIRS = 3000

# Quantile-based thresholds (auto-calibrated from data distribution)
TOP_QUANTILE = 0.70
BOTTOM_QUANTILE = 0.40


EDU_RANK = {
    "High School": 1, "Diploma": 2, "Associate's": 3,
    "Bachelor's": 4, "Degree (unspecified)": 4,
    "Master's": 5, "MBA": 5, "PhD": 6,
}


# --- Helpers -----------------------------------------------------------------

def _load_data(resume_cap: int, jd_cap: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not RESUMES_PATH.exists():
        raise FileNotFoundError(f"Resume file not found: {RESUMES_PATH}")
    if not POSTINGS_PATH.exists():
        raise FileNotFoundError(f"Postings file not found: {POSTINGS_PATH}")
    resumes = pd.read_csv(RESUMES_PATH).iloc[:resume_cap].reset_index(drop=True)
    postings = pd.read_csv(POSTINGS_PATH).dropna(subset=["description"])
    postings = postings.iloc[:jd_cap].reset_index(drop=True)
    return resumes, postings


def _sample_pairs(n_resumes: int, n_jds: int, n_pairs: int, seed: int = 42):
    random.seed(seed)
    return [
        {"resume_idx": random.randint(0, n_resumes - 1),
         "jd_idx": random.randint(0, n_jds - 1)}
        for _ in range(n_pairs)
    ]


def _safe_text(x) -> str:
    """Return a lowercase-safe string; NaN/None → ''."""
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def _skill_overlap(resume_skills: List[str], jd_skills: List[str]) -> float:
    r = {s.lower().strip() for s in resume_skills if s}
    j = {s.lower().strip() for s in jd_skills if s}
    return len(r & j) / len(j) if j else 0.0


def _extract_years(text: str) -> Optional[int]:
    text = _safe_text(text)
    if not text:
        return None
    matches = re.findall(r"(\d+)\s*\+?\s*years?", text, re.IGNORECASE)
    return max((int(y) for y in matches), default=None)


def _length_ratio(a: str, b: str) -> float:
    a, b = _safe_text(a), _safe_text(b)
    la, lb = len(a.split()), len(b.split())
    if la == 0 or lb == 0:
        return 0.0
    return 1.0 - abs(la - lb) / max(la, lb)


# --- Cache management --------------------------------------------------------

def _load_jd_cache() -> Dict[str, Dict]:
    if JD_CACHE_PATH.exists():
        try:
            with open(JD_CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            print(f"  [warn] Could not load JD cache: {e}")
    return {}


def _save_jd_cache(cache: Dict[int, Dict]) -> None:
    JD_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Ensure JSON-serializable
    serializable = {str(k): v for k, v in cache.items()}
    with open(JD_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


def _load_resume_cache() -> Dict[str, List[str]]:
    if RESUME_CACHE_PATH.exists():
        try:
            with open(RESUME_CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            print(f"  [warn] Could not load resume cache: {e}")
    return {}


def _save_resume_cache(cache: Dict[int, List[str]]) -> None:
    RESUME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in cache.items()}
    with open(RESUME_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


# --- Main pipeline -----------------------------------------------------------

def build_dataset(
    n_pairs: int = N_PAIRS,
    resume_cap: int = RESUME_CAP,
    jd_cap: int = JD_CAP,
    random_seed: int = 42,
) -> pd.DataFrame:
    print("=" * 60)
    print("PHASE 11 — DATASET BUILD (v4 — cached, NaN-safe)")
    print("=" * 60)

    # --- Load ---
    print("\nLoading data...")
    resumes, postings = _load_data(resume_cap, jd_cap)
    print(f"  Resumes (scoped): {len(resumes)}")
    print(f"  JDs (scoped):     {len(postings)}")

    # --- JD parse cache ---
    jd_cache = _load_jd_cache()
    missing_jds = [i for i in postings.index if i not in jd_cache]
    if missing_jds:
        print(f"\nParsing {len(missing_jds)} JDs (cache misses)...")
        for i, jd_idx in enumerate(missing_jds):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  JDs {i + 1}/{len(missing_jds)}...")
            try:
                jd_cache[jd_idx] = parse_jd(_safe_text(postings.loc[jd_idx, "description"]))
            except Exception:
                jd_cache[jd_idx] = {
                    "skills": {"required": [], "preferred": [], "unspecified": []},
                    "experience_years": None, "education": None, "title": None,
                }
        _save_jd_cache(jd_cache)
        print(f"  Saved JD cache to {JD_CACHE_PATH}")
    else:
        print(f"\n✅ Using cached JD parses ({len(jd_cache)} entries from disk)")

    # --- Resume skills cache ---
    resume_cache = _load_resume_cache()
    missing_resumes = [i for i in resumes.index if i not in resume_cache]
    if missing_resumes:
        print(f"\nExtracting skills for {len(missing_resumes)} resumes (cache misses)...")
        for i, r_idx in enumerate(missing_resumes):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Resumes {i + 1}/{len(missing_resumes)}...")
            try:
                resume_cache[r_idx] = [
                    s["skill"] for s in
                    extract_resume_skills(_safe_text(resumes.loc[r_idx, "Resume_str"]))
                ]
            except Exception:
                resume_cache[r_idx] = []
        _save_resume_cache(resume_cache)
        print(f"  Saved resume cache to {RESUME_CACHE_PATH}")
    else:
        print(f"✅ Using cached resume skills ({len(resume_cache)} entries from disk)")

    # --- Sample pairs and compute TF-IDF ---
    print(f"\nSampling {n_pairs} pairs and computing TF-IDF similarity...")
    pairs = _sample_pairs(len(resumes), len(postings), n_pairs, random_seed)

    raw_rows = []
    for i, pair in enumerate(pairs):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Pairs {i + 1}/{len(pairs)}...")
        r_idx = pair["resume_idx"]
        j_idx = pair["jd_idx"]
        resume_text = _safe_text(resumes.loc[r_idx, "Resume_str"])
        jd_text = _safe_text(postings.loc[j_idx, "description"])
        try:
            tfidf_sim = compute_similarity(resume_text, jd_text)
        except Exception:
            continue
        raw_rows.append({
            "resume_idx": r_idx,
            "jd_idx": j_idx,
            "tfidf_similarity": round(tfidf_sim, 4),
        })

    df_pairs = pd.DataFrame(raw_rows)
    if len(df_pairs) == 0:
        print("⚠️  No valid pairs.")
        return df_pairs

    # --- Auto-calibrate thresholds ---
    pos_t = float(np.quantile(df_pairs["tfidf_similarity"], TOP_QUANTILE))
    neg_t = float(np.quantile(df_pairs["tfidf_similarity"], BOTTOM_QUANTILE))
    print(f"\nAuto-calibrated thresholds:")
    print(f"  Positive (label=1): tfidf >= {pos_t:.4f}")
    print(f"  Negative (label=0): tfidf <= {neg_t:.4f}")

    # --- Compute features inline ---
    print("\nComputing features for retained pairs...")
    rows = []
    pos_count = 0
    neg_count = 0
    skipped = 0

    for i, row in df_pairs.iterrows():
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Rows {i + 1}/{len(df_pairs)}...")
        sim = row["tfidf_similarity"]
        if sim >= pos_t:
            label = 1
            pos_count += 1
        elif sim <= neg_t:
            label = 0
            neg_count += 1
        else:
            skipped += 1
            continue

        r_idx = int(row["resume_idx"])
        j_idx = int(row["jd_idx"])
        resume_text = _safe_text(resumes.loc[r_idx, "Resume_str"])
        jd_text = _safe_text(postings.loc[j_idx, "description"])
        resume_category = _safe_text(resumes.loc[r_idx, "Category"])

        r_skills = resume_cache.get(r_idx, [])
        jd_parsed = jd_cache.get(j_idx, {})

        jd_req = [s["skill"] for s in jd_parsed.get("skills", {}).get("required", [])]
        jd_pref = [s["skill"] for s in jd_parsed.get("skills", {}).get("preferred", [])]
        jd_all = jd_req + jd_pref

        skill_match_score = round(_skill_overlap(r_skills, jd_all), 4)
        skill_match_required = round(_skill_overlap(r_skills, jd_req), 4)

        jd_years = jd_parsed.get("experience_years")
        resume_years = _extract_years(resume_text)
        if jd_years and resume_years:
            experience_match = 1.0 if resume_years >= jd_years else round(resume_years / jd_years, 4)
        elif not jd_years:
            experience_match = 0.5
        else:
            experience_match = 0.3

        jd_edu = jd_parsed.get("education")
        resume_edu = None
        resume_lower = resume_text.lower()
        for edu_name, rank in EDU_RANK.items():
            if edu_name.lower() in resume_lower:
                if resume_edu is None or rank > EDU_RANK.get(resume_edu, 0):
                    resume_edu = edu_name
        if jd_edu and resume_edu:
            jd_r = EDU_RANK.get(jd_edu, 0)
            r_r = EDU_RANK.get(resume_edu, 0)
            education_match = 1.0 if r_r >= jd_r else (round(r_r / jd_r, 4) if jd_r else 0.0)
        elif not jd_edu:
            education_match = 0.5
        else:
            education_match = 0.3

        # Category match
        jd_title = _safe_text(jd_parsed.get("title"))
        cat_match = 0.0
        if resume_category and jd_title:
            from src.ml.feature_engineering import CATEGORY_KEYWORDS
            kws = CATEGORY_KEYWORDS.get(resume_category.upper(), [])
            if kws:
                cat_match = 1.0 if any(kw in jd_title.lower() for kw in kws) else 0.5
            else:
                cat_match = 0.5

        length_ratio = round(_length_ratio(resume_text, jd_text), 4)

        rows.append({
            "skill_match_score": skill_match_score,
            "skill_match_required": skill_match_required,
            "tfidf_similarity": sim,
            "experience_match": experience_match,
            "education_match": education_match,
            "category_match": cat_match,
            "length_ratio": length_ratio,
            "label": label,
            "resume_idx": r_idx,
            "jd_idx": j_idx,
        })

    print(f"\nRetained {len(rows)}; skipped {skipped} ambiguous.")
    print(f"  Positives: {pos_count}")
    print(f"  Negatives: {neg_count}")

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    if len(df) == 0:
        return
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nLabel proportions (%):")
    print((df["label"].value_counts(normalize=True) * 100).round(2).to_string())
    print("\nFeature summary:")
    cols = ["skill_match_score", "skill_match_required", "tfidf_similarity",
            "experience_match", "education_match", "category_match", "length_ratio"]
    existing = [c for c in cols if c in df.columns]
    print(df[existing].describe().round(3).to_string())


if __name__ == "__main__":
    df = build_dataset(n_pairs=N_PAIRS, resume_cap=RESUME_CAP, jd_cap=JD_CAP)
    summarize(df)