"""
TF-IDF text similarity between resume and job description.

This is one of the 5 components of the final match score. It captures
lexical/text overlap — how similar the actual words and phrases are.

Public API:
    - compute_similarity(resume_text, jd_text) -> float in [0, 1]
    - compute_similarity_detailed(resume_text, jd_text) -> dict with breakdown

Design notes:
    - Uses word-level TF-IDF with 1-2 grams to catch phrases like "machine learning".
    - Vectorizes resume and JD together so they share a vocabulary.
    - Cosine similarity is the classic measure for TF-IDF vectors.
    - We do NOT lowercase technical tokens' inner structure — our Phase 6
      clean_text already preserves C++, .NET, etc.
"""

from __future__ import annotations

import re
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaner import clean_text, normalize_for_matching


# --- Token pattern for TF-IDF -------------------------------------------------
# Keeps:
#   - multi-word phrases via n-grams (we'll set ngram_range=(1,2))
#   - technical tokens: C++, C#, .NET, Node.js, etc.
#   - alphanumerics with internal dots/+/#/-
_TOKEN_PATTERN = r"(?u)\b\w[\w\.\+\#\-/]*\b"

def _prepare_for_tfidf(text: str) -> str:
    """
    Light preparation: lowercase and collapse whitespace.
    We keep punctuation attached to technical tokens (C++, .NET, etc.)
    because our token_pattern preserves them.
    """
    if not isinstance(text, str):
        return ""
    t = clean_text(text).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_similarity(resume_text: str, jd_text: str) -> float:
    """
    Return a TF-IDF cosine similarity score in [0, 1].

    Returns 0.0 if either input is empty or if they share no vocabulary.
    """
    if not isinstance(resume_text, str) or not isinstance(jd_text, str):
        return 0.0
    r = _prepare_for_tfidf(resume_text)
    j = _prepare_for_tfidf(jd_text)
    if not r or not j:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            token_pattern=_TOKEN_PATTERN,
            ngram_range=(1, 2),     # unigrams + bigrams
            min_df=1,               # keep rare words (skills are rare)
            max_df=1.0,   # keep all terms; we have only 2 docs           # drop terms in >95% of docs (but only 2 docs here)
            sublinear_tf=True,      # 1+log(tf) — dampens frequent term effect
            lowercase=False,        # already lowercased
        )
        matrix = vectorizer.fit_transform([r, j])
    except ValueError:
        # Happens if vocabulary is empty (e.g. all stopwords) — return 0
        return 0.0

    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    # Numeric safety: clip to [0, 1]
    return float(max(0.0, min(1.0, sim)))


def compute_similarity_detailed(resume_text: str, jd_text: str) -> Dict:
    """
    Return a dict with:
        {
            "score": float in [0, 1],
            "shared_terms": list of top overlapping terms (by TF-IDF weight),
            "resume_top_terms": top terms in resume by TF-IDF weight,
            "jd_top_terms":     top terms in JD by TF-IDF weight,
        }
    Useful for explaining the score in the UI.
    """
    empty = {
        "score": 0.0,
        "shared_terms": [],
        "resume_top_terms": [],
        "jd_top_terms": [],
    }
    if not isinstance(resume_text, str) or not isinstance(jd_text, str):
        return empty
    r = _prepare_for_tfidf(resume_text)
    j = _prepare_for_tfidf(jd_text)
    if not r or not j:
        return empty

    vectorizer = TfidfVectorizer(
        token_pattern=_TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        lowercase=False,
    )
    try:
        matrix = vectorizer.fit_transform([r, j])
    except ValueError:
        return empty

    score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    score = max(0.0, min(1.0, score))

    feature_names = vectorizer.get_feature_names_out()
    resume_vec = matrix[0].toarray()[0]
    jd_vec = matrix[1].toarray()[0]

    def top_terms(vec, k=15):
        idx = vec.argsort()[::-1][:k]
        return [(feature_names[i], float(vec[i])) for i in idx if vec[i] > 0]

    resume_top = top_terms(resume_vec)
    jd_top = top_terms(jd_vec)

    # Shared terms = terms with non-zero weight in both, ranked by min weight
    resume_map = {feature_names[i]: resume_vec[i] for i in range(len(feature_names)) if resume_vec[i] > 0}
    jd_map = {feature_names[i]: jd_vec[i] for i in range(len(feature_names)) if jd_vec[i] > 0}
    shared = [
        (t, min(resume_map[t], jd_map[t]))
        for t in (resume_map.keys() & jd_map.keys())
    ]
    shared.sort(key=lambda x: -x[1])
    shared_top = [(t, float(w)) for t, w in shared[:15]]

    return {
        "score": round(score, 4),
        "shared_terms": shared_top,
        "resume_top_terms": resume_top,
        "jd_top_terms": jd_top,
    }