from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass(slots=True)
class ParseResult:
    text: str
    parser: str
    used_ocr: bool = False


class ParseError(RuntimeError):
    pass


def _read_text(path: Path) -> ParseResult:
    return ParseResult(text=path.read_text(encoding="utf-8"), parser="plain-text")


def _read_pdf(path: Path) -> ParseResult:
    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if text:
        return ParseResult(text=text, parser="pypdf", used_ocr=False)

    ocr_text = _ocr_pdf(path)
    if not ocr_text.strip():
        raise ParseError("PDF parsing failed: no text found after OCR fallback")
    return ParseResult(text=ocr_text, parser="ocr", used_ocr=True)


def _ocr_pdf(path: Path) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as exc:  # pragma: no cover - dependency availability varies
        raise ParseError(f"OCR dependency unavailable: {exc}") from exc

    images = convert_from_path(str(path), dpi=180)
    texts = [pytesseract.image_to_string(img, lang="kor+eng") for img in images]
    return "\n".join(texts).strip()


def parse_file(path: Path) -> ParseResult:
    if not path.exists() or not path.is_file():
        raise ParseError("File does not exist")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ParseError(f"Unsupported extension: {suffix}")

    if suffix in {".txt", ".md"}:
        return _read_text(path)
    if suffix == ".pdf":
        return _read_pdf(path)

    raise ParseError(f"No parser for extension: {suffix}")
