from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

try:
    from pypdf import PdfReader as _PdfReader
except Exception:  # pragma: no cover - optional runtime dependency
    _PdfReader = None

if _PdfReader is None:  # pragma: no cover - compatibility fallback
    try:
        from PyPDF2 import PdfReader as _PdfReader
    except Exception:
        _PdfReader = None

try:
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover - optional runtime dependency
    pdfium = None

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".swift", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".kt", ".kts", ".go", ".rs", ".c", ".cpp", ".cc", ".h", ".hpp",
    ".m", ".mm", ".rb", ".php", ".sh", ".zsh", ".bash",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".xml", ".csv", ".ini", ".cfg",
    ".sql", ".proto",
}
SUPPORTED_EXTENSIONS = {*TEXT_EXTENSIONS, ".pdf"}


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
    failures: list[str] = []

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


def _extract_pdf_text_with_pypdf(path: Path, failures: list[str]) -> str:
    if _PdfReader is None:
        failures.append("pypdf backend unavailable")
        return ""

    try:
        reader = _PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                failures.append("pypdf encrypted pdf")
                return ""
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if text:
            return text
        failures.append("pypdf extracted empty text")
        return ""
    except Exception as exc:
        failures.append(f"pypdf failed: {exc}")
        return ""


def _extract_pdf_text_with_pdfium(path: Path, failures: list[str]) -> str:
    if pdfium is None:
        failures.append("pdfium backend unavailable")
        return ""

    try:
        doc = pdfium.PdfDocument(str(path))
        pages = len(doc)
        chunks: list[str] = []
        for page_index in range(pages):
            page = doc.get_page(page_index)
            textpage = page.get_textpage()
            page_text = textpage.get_text_range() or ""
            page.close()
            if page_text.strip():
                chunks.append(page_text.strip())
        text = "\n".join(chunks).strip()
        if text:
            return text
        failures.append("pdfium extracted empty text")
        return ""
    except Exception as exc:
        failures.append(f"pdfium failed: {exc}")
        return ""


def _ocr_pdf(path: Path, failures: list[str]) -> str:
    # Prefer pdfium-based OCR first to avoid hard dependency on poppler binaries.
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


def _ocr_with_pdf2image(path: Path, failures: list[str]) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as exc:  # pragma: no cover - dependency availability varies
        failures.append(f"ocr(pdf2image) dependency unavailable: {exc}")
        return ""

    tesseract_cmd = _resolve_tesseract_cmd()
    if not tesseract_cmd:
        failures.append("ocr(pdf2image) failed: tesseract executable not found")
        return ""
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    poppler_path = _resolve_poppler_path()
    lang = _resolve_ocr_language(pytesseract)
    try:
        if poppler_path:
            images = convert_from_path(str(path), dpi=180, poppler_path=poppler_path)
        else:
            images = convert_from_path(str(path), dpi=180)
        texts = [pytesseract.image_to_string(img, lang=lang) for img in images]
        text = "\n".join(texts).strip()
        if text:
            return text
        failures.append("ocr(pdf2image) extracted empty text")
        return ""
    except Exception as exc:
        failures.append(f"ocr(pdf2image) failed: {exc}")
        return ""


def _ocr_with_pdfium_render(path: Path, failures: list[str]) -> str:
    if pdfium is None:
        failures.append("ocr(pdfium) backend unavailable")
        return ""

    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover - dependency availability varies
        failures.append(f"ocr(pytesseract) dependency unavailable: {exc}")
        return ""

    tesseract_cmd = _resolve_tesseract_cmd()
    if not tesseract_cmd:
        failures.append("ocr(pdfium) failed: tesseract executable not found")
        return ""
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    lang = _resolve_ocr_language(pytesseract)

    try:
        doc = pdfium.PdfDocument(str(path))
        pages = len(doc)
        texts: list[str] = []
        for page_index in range(pages):
            page = doc.get_page(page_index)
            bitmap = page.render(scale=2.0)
            image = bitmap.to_pil()
            page.close()
            texts.append(pytesseract.image_to_string(image, lang=lang))
        text = "\n".join(texts).strip()
        if text:
            return text
        failures.append("ocr(pdfium) extracted empty text")
        return ""
    except Exception as exc:
        failures.append(f"ocr(pdfium) failed: {exc}")
        return ""


def _ocr_with_rapidocr(path: Path, failures: list[str]) -> str:
    if pdfium is None:
        failures.append("ocr(rapidocr) backend unavailable: pdfium missing")
        return ""

    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:  # pragma: no cover - dependency availability varies
        failures.append(f"ocr(rapidocr) dependency unavailable: {exc}")
        return ""

    try:
        doc = pdfium.PdfDocument(str(path))
        engine = RapidOCR()
        texts: list[str] = []
        for page_index in range(len(doc)):
            page = doc.get_page(page_index)
            bitmap = page.render(scale=2.0)
            image = bitmap.to_pil().convert("RGB")
            page.close()

            ocr_result, _ = engine(np.array(image))
            if not ocr_result:
                continue

            lines: list[str] = []
            for item in ocr_result:
                if not item or len(item) < 2:
                    continue
                text = str(item[1]).strip()
                if text:
                    lines.append(text)
            if lines:
                texts.append("\n".join(lines))

        text = "\n".join(texts).strip()
        if text:
            return text
        failures.append("ocr(rapidocr) extracted empty text")
        return ""
    except Exception as exc:
        failures.append(f"ocr(rapidocr) failed: {exc}")
        return ""


def parse_file(path: Path) -> ParseResult:
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


def _resolve_poppler_path() -> str | None:
    env_path = os.environ.get("LOCAL_AI_POPPLER_PATH", "").strip()
    # Trust explicit env override even when sandbox blocks filesystem probing.
    if env_path:
        return env_path

    for command in ("pdftoppm", "pdfinfo"):
        found = shutil.which(command)
        if found:
            return str(Path(found).parent)

    candidates = ["/opt/homebrew/bin", "/usr/local/bin"]
    for directory in candidates:
        pdftoppm = Path(directory) / "pdftoppm"
        pdfinfo = Path(directory) / "pdfinfo"
        if pdftoppm.exists() and pdfinfo.exists():
            return directory
    return None


def _resolve_tesseract_cmd() -> str | None:
    env_cmd = os.environ.get("LOCAL_AI_TESSERACT_CMD", "").strip()
    # Trust explicit env override even when sandbox blocks filesystem probing.
    if env_cmd:
        return env_cmd

    found = shutil.which("tesseract")
    if found:
        return found

    for candidate in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
        if Path(candidate).exists():
            return candidate
    return None


def _resolve_ocr_language(pytesseract_module) -> str:
    forced = os.environ.get("LOCAL_AI_OCR_LANG", "").strip()
    if forced:
        return forced

    try:
        languages = {lang.strip() for lang in pytesseract_module.get_languages(config="") if lang.strip()}
    except Exception:
        return "eng"

    if "kor" in languages and "eng" in languages:
        return "kor+eng"
    if "eng" in languages:
        return "eng"
    if languages:
        # Pick a deterministic fallback when English/Korean packs are unavailable.
        return sorted(languages)[0]
    return "eng"
