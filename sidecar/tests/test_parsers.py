from pathlib import Path

import pytest

from local_ai_core.parsers import ParseError, parse_file


def test_parse_txt_file(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("hello local ai", encoding="utf-8")
    result = parse_file(path)
    assert "hello" in result.text
    assert result.parser == "plain-text"


def test_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(ParseError):
        parse_file(path)
