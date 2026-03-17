from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    order: int
    text: str


def chunk_text(text: str, *, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks
