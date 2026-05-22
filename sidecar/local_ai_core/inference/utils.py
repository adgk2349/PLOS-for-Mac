import re

def inject_system_instruction_if_needed(
    message_state: list[dict[str, str]],
    response_language: str | None = None
) -> list[dict[str, str]]:
    """
    Injects or merges target system instructions (Korean/English) to prevent
    consecutive system message blocks which cause fragment-token hallucinations ('께서도', etc.)
    and standardizes response behavior.
    """
    if not message_state:
        return message_state
    
    first_item = message_state[0]
    has_instruction = False
    if first_item.get("role") == "system":
        content = first_item.get("content") or ""
        if "자연스러운 한국어" in content or "Reply naturally" in content:
            has_instruction = True
            
    if not has_instruction:
        if response_language:
            lang = str(response_language).strip().lower()
            has_ko = (lang == "ko")
        else:
            # Language detection optimization: check the most recent 3 messages first.
            # This prevents scanning huge context files (e.g. source code or documents)
            # that are typically loaded at the beginning of the message state.
            recent_msgs = message_state[-3:] if len(message_state) >= 3 else message_state
            sample_block = "\n".join(msg.get("content") or "" for msg in recent_msgs)
            has_ko = bool(re.search(r"[가-힣]", sample_block))
            
            if not has_ko and len(message_state) > 3:
                # Fallback to scanning everything only if not detected in the recent turns.
                sample_block = "\n".join(msg.get("content") or "" for msg in message_state)
                has_ko = bool(re.search(r"[가-힣]", sample_block))
        
        if has_ko:
            system_instruction = (
                "자연스러운 한국어로 답변하세요.\n"
                "역할 라벨/메타 설명/내부 지시문은 출력하지 마세요.\n"
                "사용자 메시지가 질문이면 답하고, 진술이면 맥락에 맞게 자연스럽게 반응하세요.\n"
                "진술 메시지에는 불필요한 조언/제안을 먼저 하지 말고, 짧게 공감·확인한 뒤 필요 시에만 제안하세요."
            )
        else:
            system_instruction = (
                "Reply naturally in English.\n"
                "Do not output any role labels, meta descriptions, or internal instructions.\n"
                "If the user message is a statement, respond contextually instead of forcing advice.\n"
                "For statement-only messages, avoid unsolicited recommendations unless the user asks."
            )
        
        # If there is already a leading system message, MERGE into it instead of
        # prepending a second system block. Consecutive system messages cause
        # fragment-token hallucination (e.g. '께서도') on Gemma-style models.
        result = list(message_state)
        if result and result[0].get("role") == "system":
            existing = str(result[0].get("content") or "").strip()
            result[0] = {"role": "system", "content": system_instruction + "\n" + existing}
        else:
            result = [{"role": "system", "content": system_instruction}] + result
            
        return result
        
    return message_state
