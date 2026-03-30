"""Backward-compatible bridge for file parsers.

Canonical path: local_ai_core.indexing.parsers
"""

from .indexing.parsers import *  # noqa: F401,F403
from .indexing import parsers as _indexing_parsers

# Backward-compatibility re-exports for tests/legacy callers that monkeypatch
# parser internals from local_ai_core.parsers.
_extract_pdf_text_with_pypdf = _indexing_parsers._extract_pdf_text_with_pypdf
_extract_pdf_text_with_pdfium = _indexing_parsers._extract_pdf_text_with_pdfium
_ocr_with_pdfium_render = _indexing_parsers._ocr_with_pdfium_render
_ocr_with_rapidocr = _indexing_parsers._ocr_with_rapidocr
_ocr_with_pdf2image = _indexing_parsers._ocr_with_pdf2image
_resolve_poppler_path = _indexing_parsers._resolve_poppler_path
_resolve_tesseract_cmd = _indexing_parsers._resolve_tesseract_cmd
_resolve_ocr_language = _indexing_parsers._resolve_ocr_language

# These wrappers preserve old monkeypatch semantics against this bridge module.
TEXT_EXTENSIONS = _indexing_parsers.TEXT_EXTENSIONS
SUPPORTED_EXTENSIONS = _indexing_parsers.SUPPORTED_EXTENSIONS
ParseResult = _indexing_parsers.ParseResult
ParseError = _indexing_parsers.ParseError


def _read_text(path):
    return _indexing_parsers._read_text(path)


def _read_pdf(path):
    failures = []

    text = _extract_pdf_text_with_pypdf(path, failures)
    if text:
        return ParseResult(text=text, parser="pypdf", used_ocr=False)

    text = _extract_pdf_text_with_pdfium(path, failures)
    if text:
        return ParseResult(text=text, parser="pdfium-text", used_ocr=False)

    text = _ocr_pdf(path, failures)
    if text:
        return ParseResult(text=text, parser="ocr", used_ocr=True)

    details = "; ".join(failures) if failures else "no backend available"
    raise ParseError(f"PDF parsing failed: {details}")


def _ocr_pdf(path, failures):
    text = _ocr_with_pdfium_render(path, failures)
    if text:
        return text
    text = _ocr_with_rapidocr(path, failures)
    if text:
        return text
    text = _ocr_with_pdf2image(path, failures)
    if text:
        return text
    return ""


def parse_file(path):
    if not path.exists() or not path.is_file():
        raise ParseError("File does not exist")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ParseError(f"Unsupported extension: {suffix}")

    if suffix in TEXT_EXTENSIONS:
        return _read_text(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ParseError(f"No parser for extension: {suffix}")
