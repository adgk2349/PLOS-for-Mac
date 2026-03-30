from pathlib import Path

import pytest

import local_ai_core.parsers as parsers
from local_ai_core.parsers import ParseError, parse_file


def test_parse_txt_file(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("hello local ai", encoding="utf-8")
    result = parse_file(path)
    assert "hello" in result.text
    assert result.parser == "plain-text"


def test_parse_code_file_as_text(tmp_path: Path):
    path = tmp_path / "main.py"
    path.write_text("def hello():\n    return 'ok'\n", encoding="utf-8")
    result = parse_file(path)
    assert "hello" in result.text
    assert result.parser == "plain-text"


def test_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "data.exe"
    path.write_text("binary-like", encoding="utf-8")
    with pytest.raises(ParseError):
        parse_file(path)


def test_pdf_falls_back_to_pdfium_text_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4\n%stub\n")

    monkeypatch.setattr(parsers, "_extract_pdf_text_with_pypdf", lambda *_: "")
    monkeypatch.setattr(parsers, "_extract_pdf_text_with_pdfium", lambda *_: "pdfium extracted text")
    monkeypatch.setattr(parsers, "_ocr_pdf", lambda *_: "")

    result = parse_file(path)
    assert result.text == "pdfium extracted text"
    assert result.parser == "pdfium-text"
    assert result.used_ocr is False


def test_pdf_uses_ocr_when_text_backends_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4\n%scan\n")

    monkeypatch.setattr(parsers, "_extract_pdf_text_with_pypdf", lambda *_: "")
    monkeypatch.setattr(parsers, "_extract_pdf_text_with_pdfium", lambda *_: "")
    monkeypatch.setattr(parsers, "_ocr_pdf", lambda *_: "ocr text")

    result = parse_file(path)
    assert result.text == "ocr text"
    assert result.parser == "ocr"
    assert result.used_ocr is True


def test_ocr_pipeline_prefers_pdfium_then_rapidocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4\n%scan\n")

    calls: list[str] = []

    def fake_pdfium(*_):
        calls.append("pdfium")
        return ""

    def fake_rapidocr(*_):
        calls.append("rapidocr")
        return "rapidocr text"

    def fake_pdf2image(*_):
        calls.append("pdf2image")
        return ""

    monkeypatch.setattr(parsers, "_ocr_with_pdfium_render", fake_pdfium)
    monkeypatch.setattr(parsers, "_ocr_with_rapidocr", fake_rapidocr)
    monkeypatch.setattr(parsers, "_ocr_with_pdf2image", fake_pdf2image)

    text = parsers._ocr_pdf(path, [])
    assert text == "rapidocr text"
    assert calls == ["pdfium", "rapidocr"]


def test_explicit_env_paths_are_trusted_for_ocr_commands(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_AI_POPPLER_PATH", "/custom/poppler/bin")
    monkeypatch.setenv("LOCAL_AI_TESSERACT_CMD", "/custom/bin/tesseract")

    assert parsers._resolve_poppler_path() == "/custom/poppler/bin"
    assert parsers._resolve_tesseract_cmd() == "/custom/bin/tesseract"
