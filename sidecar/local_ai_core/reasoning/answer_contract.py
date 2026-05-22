from __future__ import annotations
import json
import re
from typing import Any, Optional, Dict, List, Tuple

def extract_contract_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the first valid JSON object from the given text.
    Prioritizes text within ```json blocks.
    """
    if not text:
        return None
        
    # 1. Try to find json code blocks
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_blocks:
        for block in json_blocks:
            try:
                # Clean up any potential leading/trailing junk
                content = block.strip()
                return json.loads(content)
            except json.JSONDecodeError:
                continue
                
    # 2. Try to find any curly brace blocks
    potential_json = re.findall(r"\{.*\}", text, re.DOTALL)
    if potential_json:
        # Sort by length descending to find the largest potential object
        for candidate in sorted(potential_json, key=len, reverse=True):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
                
    return None

def validate_contract_response(
    answer: str,
    raw_text: str,
    expected_language: str,
    answer_type_hint: str,
    declared_answer_type: Optional[str] = None,
    declared_language: Optional[str] = None,
) -> List[str]:
    """
    Validates the parsed contract response according to Sidecar V2 rules.
    Returns a list of failure reasons (empty list if valid).
    """
    reasons = []
    
    if not answer:
        reasons.append("empty_answer")
        return reasons

    # Type matching (simplified)
    if declared_answer_type and answer_type_hint != "freeform":
        if declared_answer_type != answer_type_hint:
            reasons.append(f"type_mismatch:expected={answer_type_hint}:got={declared_answer_type}")

    # Language matching (simplified)
    if declared_language and expected_language:
        if declared_language.lower() != expected_language.lower():
            reasons.append(f"language_mismatch:expected={expected_language}:got={declared_language}")

    return reasons

def infer_answer_type_hint(query: str) -> str:
    """
    Determines the expected answer length based on query keywords.
    """
    lowered = str(query or "").lower()
    if any(k in lowered for k in ["짧게", "간단히", "한줄", "brief", "short"]):
        return "short"
    if any(k in lowered for k in ["자세히", "상세히", "길게", "deep", "detail", "long"]):
        return "long"
    return "medium"

def coerce_answer_type_hint(hint: Any) -> str:
    """
    Coerces various hint types into a canonical string.
    """
    s = str(hint or "").lower().strip()
    if s in {"short", "medium", "long", "freeform"}:
        return s
    return "medium"
