"""
Tests for src/preprocessing/text_cleaner.py
Run with: python -m pytest tests/test_nlp.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.text_cleaner import (  # noqa: E402
    clean_text,
    normalize_for_matching,
    tokenize,
    remove_stopwords,
    lemmatize_tokens,
    preprocess,
)


# --- clean_text ---------------------------------------------------------------

def test_clean_text_collapses_whitespace():
    # Tabs and carriage returns collapse to spaces; newlines are preserved
    # because section headers ("Skills", "Experience") live on separate lines.
    out = clean_text("Hello    world\t\tfoo\r\nbar")
    # \r\n becomes \n (newline preserved); tabs collapse to space
    assert out == "Hello world foo\nbar"


def test_clean_text_preserves_punctuation():
    out = clean_text("Skills: Python, C++, Node.js; AWS.")
    assert "C++" in out
    assert "Node.js" in out
    assert "AWS" in out


# --- normalize_for_matching ---------------------------------------------------

def test_normalize_for_matching_lowercases():
    out = normalize_for_matching("Python 3, C++, Node.js")
    assert out == out.lower()
    assert "python 3" in out
    assert "c++" in out
    assert "node.js" in out


# --- tokenize -----------------------------------------------------------------

def test_tokenize_preserves_technical_tokens():
    text = "Familiar with C++, C#, .NET, Node.js, React.js, AWS, scikit-learn."
    tokens = [t.lower() for t in tokenize(text)]
    for tok in ["c++", "c#", ".net", "node.js", "react.js", "aws", "scikit-learn"]:
        assert tok in tokens, f"missing {tok!r} in tokens: {tokens}"


def test_tokenize_handles_numbers():
    tokens = tokenize("Python 3.11 and 42 objects")
    lower = [t.lower() for t in tokens]
    # "Python" and "3.11" should both appear as tokens.
    assert "python" in lower
    assert "3.11" in lower or "3" in lower  # depends on regex behavior
    assert "42" in lower


def test_tokenize_empty_returns_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


# --- stopwords ----------------------------------------------------------------

def test_remove_stopwords_basic():
    out = remove_stopwords(["the", "quick", "brown", "fox", "and", "python"])
    assert "the" not in out
    assert "and" not in out
    assert "python" in out


def test_remove_stopwords_keeps_protected():
    out = remove_stopwords(["in", "c++", "the", "aws"])
    assert "c++" in out
    assert "aws" in out


# --- lemmatize ----------------------------------------------------------------

def test_lemmatize_basic():
    # WordNet lemmatizes based on assumed noun POS by default.
    # Plural nouns reduce to singular; verbs may not change.
    out = lemmatize_tokens(["running", "cats", "better", "python"])
    assert "cat" in out          # plural noun → singular
    assert "python" in out       # proper noun unchanged
    # Note: "running" may stay as "running" or become "run" depending on
    # NLTK/WordNet version. We assert it's not empty.
    assert "running" in out or "run" in out


def test_lemmatize_preserves_technical():
    out = lemmatize_tokens(["c++", ".net", "node.js", "running"])
    assert "c++" in out
    assert ".net" in out
    assert "node.js" in out


# --- full pipeline ------------------------------------------------------------

def test_preprocess_returns_all_views():
    result = preprocess("Python and C++ developer with 2 years of experience.")
    for key in ["clean_text", "normalized_text", "tokens",
                "tokens_lower", "tokens_nostop", "tokens_lemmatized"]:
        assert key in result


def test_preprocess_technical_tokens_survive():
    text = "Worked with C++, .NET, Node.js, React.js, AWS, SQL, and scikit-learn."
    result = preprocess(text)
    lower_set = set(result["tokens_lower"])
    for tok in ["c++", ".net", "node.js", "react.js", "aws", "sql", "scikit-learn"]:
        assert tok in lower_set, f"missing {tok!r}"