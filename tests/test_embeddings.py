"""
Tests for src/nlp/embeddings.py

NOTE: These tests require the sentence-transformers model to be cached
locally (downloaded once on first use). They run offline after that.

Run with: python -m pytest tests/test_embeddings.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nlp.embeddings import (  # noqa: E402
    compute_semantic_similarity,
    compare_tfidf_vs_embeddings,
    get_embedding,
    embed_texts,
)


# --- Basic behavior ----------------------------------------------------------

def test_similarity_returns_float_in_range():
    sim = compute_semantic_similarity(
        "Python developer with Django",
        "Python engineer skilled in Django",
    )
    assert isinstance(sim, float)
    assert 0.0 <= sim <= 1.0


def test_similarity_high_for_paraphrase():
    # Same meaning, different words
    sim = compute_semantic_similarity(
        "Python developer with Django experience",
        "Python engineer skilled in Django",
    )
    assert sim > 0.6, f"Expected high similarity, got {sim}"


def test_similarity_low_for_unrelated():
    sim = compute_semantic_similarity(
        "Python backend developer",
        "Chef with culinary training",
    )
    assert sim < 0.4, f"Expected low similarity, got {sim}"


def test_similarity_identical_texts():
    text = "Senior Python engineer building backend systems."
    sim = compute_semantic_similarity(text, text)
    # Identical texts should be ~1.0 (allowing for numeric precision)
    assert sim > 0.95


def test_similarity_symmetric():
    a = "Python developer"
    b = "Machine learning engineer"
    assert abs(compute_semantic_similarity(a, b) - compute_semantic_similarity(b, a)) < 1e-5


def test_similarity_empty_inputs():
    assert compute_semantic_similarity("", "") == 0.0
    assert compute_semantic_similarity("Python", "") == 0.0
    assert compute_semantic_similarity("", "Python") == 0.0


def test_similarity_none_inputs():
    assert compute_semantic_similarity(None, None) == 0.0
    assert compute_semantic_similarity("Python", None) == 0.0
    assert compute_semantic_similarity(None, "Python") == 0.0


def test_similarity_case_insensitive_behavior():
    # Lowercased vs uppercase should give same result (model is case-agnostic)
    lower = compute_semantic_similarity("python developer", "django engineer")
    upper = compute_semantic_similarity("PYTHON DEVELOPER", "DJANGO ENGINEER")
    assert abs(lower - upper) < 0.02


# --- Embedding API -----------------------------------------------------------

def test_get_embedding_returns_vector():
    emb = get_embedding("Python developer")
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) == 384   # MiniLM-L6-v2 dimension
    assert emb.dtype in (np.float32, np.float64)


def test_get_embedding_empty_returns_zero_vector():
    emb = get_embedding("")
    assert isinstance(emb, np.ndarray)
    assert len(emb) == 384
    assert np.allclose(emb, 0.0)


def test_embed_texts_batch():
    embs = embed_texts(["Python developer", "Chef", "Machine learning engineer"])
    assert embs.shape == (3, 384)


def test_embed_texts_empty_list():
    embs = embed_texts([])
    assert embs.shape == (0, 384)


# --- TF-IDF vs embeddings comparison ----------------------------------------

def test_compare_returns_expected_keys():
    result = compare_tfidf_vs_embeddings(
        "Python developer with Django",
        "Python engineer skilled in Django",
    )
    for key in ["tfidf", "embeddings", "difference", "average"]:
        assert key in result


def test_compare_embeddings_higher_than_tfidf_for_paraphrase():
    """
    The whole point of semantic matching: for paraphrases that share few
    exact words, embeddings should score notably higher than TF-IDF.
    """
    a = "Experienced coder who writes software and debugs programs"
    b = "Senior programmer building applications and fixing bugs"
    result = compare_tfidf_vs_embeddings(a, b)
    assert result["embeddings"] > result["tfidf"], (
        f"Expected embeddings > tfidf, got {result}"
    )


def test_compare_average_is_mean():
    result = compare_tfidf_vs_embeddings("Python", "Java")
    expected = round((result["tfidf"] + result["embeddings"]) / 2, 4)
    assert abs(result["average"] - expected) < 1e-4


def test_compare_difference_is_correct():
    result = compare_tfidf_vs_embeddings("Python", "Java")
    expected = round(result["embeddings"] - result["tfidf"], 4)
    assert abs(result["difference"] - expected) < 1e-4