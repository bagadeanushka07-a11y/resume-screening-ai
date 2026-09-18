"""
Semantic text similarity using Sentence Transformers.

Complements the TF-IDF matcher by capturing MEANING rather than exact words.

Example:
    TF-IDF sees:    "Python developer" vs "Python engineer"      → partial overlap
    Embeddings see: "Python developer" vs "Python engineer"      → very similar
    Embeddings see: "Python developer" vs "basketball player"    → dissimilar

Model:
    all-MiniLM-L6-v2 — 80MB, fast (50ms per sentence on CPU), strong quality
    Chosen for deployment feasibility (works on Streamlit Cloud free tier).

Public API:
    - compute_semantic_similarity(text_a, text_b) -> float in [0, 1]
    - compare_tfidf_vs_embeddings(text_a, text_b) -> dict with both scores
    - get_embedding(text) -> np.ndarray
    - embed_texts(texts) -> np.ndarray (batched)
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re
from functools import lru_cache
from typing import Dict, List, Optional, Union

import numpy as np

# Import lazy-loaded to avoid heavy import at module load time
_MODEL = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence transformer model (downloads on first call)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


# --- Text preparation --------------------------------------------------------

def _prepare(text: str, max_chars: int = 2000) -> str:
    """Clean and truncate text for embedding."""
    if not isinstance(text, str):
        return ""
    # Normalize whitespace
    t = re.sub(r"\s+", " ", text).strip()
    # Truncate — embeddings become noisy on very long inputs, and MiniLM has
    # a 256-token limit anyway. 2000 chars ≈ 400 tokens → safe.
    return t[:max_chars]


# --- Public API --------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    """Return a single embedding vector (384-dim for MiniLM)."""
    model = _get_model()
    prepared = _prepare(text)
    if not prepared:
        return np.zeros(384, dtype=np.float32)
    return model.encode([prepared], convert_to_numpy=True, show_progress_bar=False)[0]


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return a matrix of embeddings for a list of texts (batched, faster)."""
    model = _get_model()
    prepared = [_prepare(t) for t in texts]
    if not any(prepared):
        return np.zeros((len(texts), 384), dtype=np.float32)
    return model.encode(prepared, convert_to_numpy=True, show_progress_bar=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, safe against zero vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Return semantic similarity in [0, 1] between two texts.

    Sentence-transformer cosine similarity is naturally in [-1, 1] but
    for our model and short texts it typically falls in [0, 1]. We clip
    negatives to 0 so downstream consumers see a clean [0, 1] score.
    """
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return 0.0
    pa, pb = _prepare(text_a), _prepare(text_b)
    if not pa or not pb:
        return 0.0
    model = _get_model()
    embeddings = model.encode([pa, pb], convert_to_numpy=True, show_progress_bar=False)
    sim = _cosine(embeddings[0], embeddings[1])
    return float(max(0.0, min(1.0, sim)))


def compare_tfidf_vs_embeddings(text_a: str, text_b: str) -> Dict[str, float]:
    """
    Convenience helper for Phase 14's UI comparison.

    Returns:
        {
            "tfidf": float in [0, 1],          from Phase 10
            "embeddings": float in [0, 1],     from this module
            "difference": float,               embeddings - tfidf
            "average": float,                  mean of the two
        }
    """
    from src.nlp.tfidf_matcher import compute_similarity as tfidf_sim
    tf = tfidf_sim(text_a, text_b) if isinstance(text_a, str) and isinstance(text_b, str) else 0.0
    em = compute_semantic_similarity(text_a, text_b)
    return {
        "tfidf": round(tf, 4),
        "embeddings": round(em, 4),
        "difference": round(em - tf, 4),
        "average": round((tf + em) / 2, 4),
    }