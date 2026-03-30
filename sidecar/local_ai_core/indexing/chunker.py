import re
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    order: int
    text: str


def chunk_text(text: str, *, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Fixed-size chunking fallback."""
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


def semantic_chunk_text(
    text: str,
    embedding_service: Any,
    *,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1500,
) -> list[str]:
    """
    Splits text at natural topic boundaries using sentence similarity.
    Falls back to fixed-size chunking if embedding_service is not provided.
    """
    if not text.strip():
        return []
    
    # 1. Split into sentences (simple regex-based sentence splitter)
    # Handles common endings followed by whitespace
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
    
    if len(sentences) <= 1 or not embedding_service:
        return chunk_text(text, chunk_size=max_chunk_size)

    # 2. Group small sentences to form semantically meaningful units
    # This reduces the number of embedding calls.
    units: list[str] = []
    current_unit = ""
    for s in sentences:
        if len(current_unit) + len(s) < min_chunk_size:
            current_unit += (" " if current_unit else "") + s
        else:
            if current_unit:
                units.append(current_unit)
            current_unit = s
    if current_unit:
        units.append(current_unit)

    if len(units) <= 1:
        return units if units else [text[:max_chunk_size]]

    # 3. Embed units and calculate similarity between adjacent ones
    try:
        vectors = embedding_service.embed_documents(units)
    except Exception:
        # Fallback if batch embedding fails
        return chunk_text(text, chunk_size=max_chunk_size)

    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    # 4. Find breakpoints where similarity is significantly low
    breakpoints = []
    for i in range(len(vectors) - 1):
        sim = cosine_similarity(vectors[i], vectors[i+1])
        # Threshold 0.75 is a good heuristic for topic change
        # Also ensure we don't create tiny chunks unless necessary
        if sim < 0.75:
            breakpoints.append(i + 1)

    # 5. Assemble chunks based on breakpoints
    chunks: list[str] = []
    start_idx = 0
    current_chunk = ""
    
    for i, unit in enumerate(units):
        # Break if i is a breakpoint OR if adding next unit exceeds max_chunk_size
        should_break = (i in breakpoints) or (len(current_chunk) + len(unit) > max_chunk_size)
        
        if should_break and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = unit
        else:
            current_chunk += (" " if current_chunk else "") + unit
            
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
