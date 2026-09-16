"""
Tests for src/preprocessing/document_parser.py
Run with:  python -m pytest tests/test_parser.py -v
"""

import sys
from pathlib import Path

import pytest

# Make `src` importable when running pytest from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.document_parser import (  # noqa: E402
    parse_pdf,
    parse_docx,
    parse_document,
)

FIXTURES = Path(__file__).parent / "fixtures"


# --- DOCX ---------------------------------------------------------------------

def test_parse_docx_returns_text():
    path = FIXTURES / "sample_resume.docx"
    if not path.exists():
        pytest.skip("sample_resume.docx not present")
    text = parse_docx(path)
    assert isinstance(text, str)
    assert len(text) > 50
    assert "Python" in text
    assert "Machine Learning" in text


def test_parse_docx_preserves_technical_tokens():
    path = FIXTURES / "sample_resume.docx"
    if not path.exists():
        pytest.skip("sample_resume.docx not present")
    text = parse_docx(path)
    assert "C++" in text
    assert ".NET" in text
    assert "Node.js" in text


def test_parse_docx_missing_file_raises():
    with pytest.raises(ValueError):
        parse_docx(FIXTURES / "does_not_exist.docx")


# --- PDF ----------------------------------------------------------------------

def test_parse_pdf_returns_text():
    path = FIXTURES / "sample_resume.pdf"
    if not path.exists():
        pytest.skip("sample_resume.pdf not present")
    text = parse_pdf(path)
    assert isinstance(text, str)
    assert len(text) > 20


def test_parse_pdf_missing_file_raises():
    with pytest.raises(ValueError):
        parse_pdf(FIXTURES / "does_not_exist.pdf")


# --- Unified entry point ------------------------------------------------------

def test_parse_document_detects_pdf():
    path = FIXTURES / "sample_resume.pdf"
    if not path.exists():
        pytest.skip("sample_resume.pdf not present")
    assert len(parse_document(path)) > 10


def test_parse_document_detects_docx():
    path = FIXTURES / "sample_resume.docx"
    if not path.exists():
        pytest.skip("sample_resume.docx not present")
    assert len(parse_document(path)) > 10


def test_parse_document_rejects_unsupported_ext():
    with pytest.raises(ValueError):
        parse_document("resume.txt")