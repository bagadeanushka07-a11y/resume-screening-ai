"""
NLP text preprocessing for resume / job description text.

Design philosophy:
    - Produce MULTIPLE VIEWS of the text rather than one aggressively-cleaned version.
    - Technical tokens (C++, C#, .NET, Node.js, React.js, AWS, SQL) must survive.
    - Aggressive steps (lowercasing, stopword removal, lemmatization) are OPTIONAL
      and applied only in the token view, never to the "clean text" view used for
      TF-IDF or skill extraction.

Public API:
    - clean_text(text)              -> whitespace-normalized, punctuation preserved
    - normalize_for_matching(text)  -> lowercase + collapse whitespace (for alias lookup)
    - tokenize(text)                -> list of tokens with technical terms protected
    - remove_stopwords(tokens)      -> tokens without English stopwords
    - lemmatize_tokens(tokens)      -> lemmatized tokens
    - preprocess(text)              -> dict with all views
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --- Technical tokens that must survive --------------------------------------
PROTECTED_TOKENS = [
    "c++", "c#", "f#", "objective-c", "objective c",
    ".net", "asp.net", "vb.net",
    "node.js", "react.js", "vue.js", "next.js", "express.js", "angular.js",
    "three.js", "d3.js", "nuxt.js", "backbone.js", "ember.js",
    "ci/cd", "rest api", "restful api",
    "scikit-learn", "sklearn",
    "aws", "gcp", "azure",
]

# Placeholder map for temporary protection during tokenization
_PROTECT_MAP = {tok: f"__PROTECTED_{i}__" for i, tok in enumerate(PROTECTED_TOKENS)}
_RESTORE_MAP = {v: k for k, v in _PROTECT_MAP.items()}


# --- NLTK setup ---------------------------------------------------------------

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

try:
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize("test")   # force load
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    LEMMATIZER = WordNetLemmatizer()


# --- Public API ---------------------------------------------------------------

def clean_text(text: str) -> str:
    """Whitespace-normalized text, punctuation preserved."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u202a-\u202e]", "", text)
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r" +\n", "\n", text)   # NEW: strip trailing spaces before newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Lowercase + collapse whitespace. Punctuation preserved."""
    return re.sub(r"\s+", " ", clean_text(text).lower()).strip()


def _protect(text: str) -> str:
    for tok, placeholder in _PROTECT_MAP.items():
        text = re.sub(re.escape(tok), placeholder, text, flags=re.IGNORECASE)
    return text


def _restore(text: str) -> str:
    for placeholder, tok in _RESTORE_MAP.items():
        text = text.replace(placeholder, tok)
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text; technical tokens protected from being split."""
    if not isinstance(text, str):
        return []
    protected = _protect(text)
    tokens = re.findall(
        r"__PROTECTED_\d+__|[A-Za-z_][A-Za-z0-9_']*|[0-9]+(?:\.[0-9]+)?",
        protected,
    )
    return [_restore(t) for t in tokens]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove English stopwords, keep protected technical tokens."""
    if not tokens:
        return []
    protected = set(PROTECTED_TOKENS)
    return [t for t in tokens if t.lower() not in STOPWORDS or t.lower() in protected]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatize with NLTK WordNet; keep protected technical tokens verbatim.

    Note: NLTK's default lemmatizer assumes noun POS unless told otherwise,
    so verb forms like "running" → "running" (not "run"). This is acceptable
    because our downstream tasks (skill matching, TF-IDF) rely on surface forms,
    not lemma reduction. POS-aware lemmatization is available via spaCy and
    can be swapped in later if needed.
    """
    if not tokens:
        return []
    protected = set(PROTECTED_TOKENS)
    out = []
    for t in tokens:
        if t.lower() in protected:
            out.append(t)
        else:
            out.append(LEMMATIZER.lemmatize(t.lower()))
    return out


def preprocess(text: str) -> Dict[str, object]:
    """Return all views of the text."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens_lower = [t.lower() for t in tokens]
    tokens_nostop = remove_stopwords(tokens)
    tokens_lemma = lemmatize_tokens(tokens_nostop)

    return {
        "clean_text": cleaned,
        "normalized_text": normalize_for_matching(cleaned),
        "tokens": tokens,
        "tokens_lower": tokens_lower,
        "tokens_nostop": tokens_nostop,
        "tokens_lemmatized": tokens_lemma,
    }