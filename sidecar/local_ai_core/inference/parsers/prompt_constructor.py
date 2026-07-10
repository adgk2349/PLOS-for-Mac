from __future__ import annotations
import re
from typing import TYPE_CHECKING
from ..base import BaseDelegate
from ...models import WorkMode, Citation
from ...language_utils import insufficient_evidence_message, response_language_instruction

if TYPE_CHECKING:
    from ...local_inference import LocalInferenceEngine

# Component: prompt_constructor.py
class PromptConstructor(BaseDelegate):
    def _last_resort_direct_reply_prompt(self, *, query: str, response_language: str) -> str:
        if response_language == "ko":
            return (
                "사용자 마지막 메시지에 바로 답하세요. "
                "한 문장으로 자연스러운 한국어 존댓말만 사용하세요. "
                "규칙/메타/역할 라벨은 출력하지 마세요.\n"
                f"사용자 메시지: {query}\n"
                "답변:"
            )
        return (
            "Answer the user's last message directly in one natural sentence. "
            "Do not output rules, meta text, or role labels.\n"
            f"User message: {query}\n"
            "Answer:"
        )

    def _conversation_repair_prompt(self, prompt: str, *, response_language: str) -> str:
        if response_language == "ko":
            repair = (
                "보정 지시: 사용자 마지막 메시지에 바로 답하세요. "
                "자연스러운 한국어 존댓말만 출력하고, 메타 문장/규칙 문장/역할 라벨은 금지합니다."
            )
        else:
            repair = (
                "Repair instruction: answer the user's latest message directly in natural language. "
                "Do not output meta commentary, rules, or role labels."
            )
        return f"{prompt}\n{repair}"

    def _grounded_repair_prompt(self, prompt: str, *, response_language: str) -> str:
        if response_language == "ko":
            repair = (
                "최종 답변 규칙: 근거를 자연스럽게 재서술해 답하세요. "
                "근거 문장을 길게 그대로 복붙하지 말고, 같은 내용은 하나로 합치세요. "
                "반복 문구가 있으면 핵심만 한 번만 정리하세요. "
                "Evidence/Explanation/Question/Mode/Continuation 같은 메타 문구를 출력하지 마세요."
            )
        else:
            repair = (
                "Final response rule: Paraphrase the evidence naturally. "
                "Do not copy long evidence phrases verbatim, and merge duplicates into one point. "
                "Output only the final grounded answer without meta labels such as Evidence/Explanation/Question/Mode/Continuation."
            )
        return f"{prompt}\n{repair}"

    def _prompt(self, query: str, mode: WorkMode, citations: list[Citation], response_language: str) -> str:
        evidence_lines = self._prepare_evidence_lines(
            citations=citations[:8],
            max_items=5,
            response_language=response_language,
        )
        snippets = "\n".join(evidence_lines)
        strict_rule = ""
        strict_msg = insufficient_evidence_message(response_language)
        if mode == WorkMode.STRICT_SEARCH:
            strict_rule = (
                "STRICT RULE: If evidence is insufficient, output exactly "
                f"'{strict_msg}' "
                "Do not speculate.\n"
            )
        ko_tone = ""
        if response_language == "ko":
            ko_tone = (
                "Korean style rule: Use natural polite Korean with concise, direct phrasing. "
                "Avoid repetitive or copy-paste wording.\n"
            )
        output_guard = (
            "Output guard: Return only the final answer text. "
            "Never print labels like Evidence:, Explanation:, Question:, Mode:, Continuation:, User:, Assistant:.\n"
        )
        synthesis_rule = (
            "Synthesis rule: Paraphrase evidence naturally instead of copying long phrases verbatim. "
            "Merge duplicate evidence into one concise point.\n"
        )
        return (
            "You are a local-first assistant. Answer only from citation evidence.\n"
            f"{response_language_instruction(response_language)}\n"
            f"{ko_tone}"
            f"{strict_rule}"
            f"{output_guard}"
            f"{synthesis_rule}"
            f"Mode: {mode.value}\n"
            f"Question: {query}\n"
            f"Evidence:\n{snippets}"
        )

    def _conversational_prompt(
        self,
        *,
        query: str,
        mode: WorkMode,
        response_language: str,
        session_summary: str | None = None,
    ) -> str:
        if response_language == "ko":
            context_line = ""
            if session_summary:
                context_line = f"참고: {session_summary}\n"
            return (
                "자연스러운 한국어로 답변하세요.\n"
                "현재 사용자 메시지에 바로 답하세요.\n"
                "메타 설명이나 역할 라벨 없이, 본문부터 시작하세요.\n"
                "정보가 아주 부족한 경우가 아니면 되묻지 말고 답을 먼저 주세요.\n"
                f"{context_line}"
                f"{query}\n"
            )
        context_line = ""
        if session_summary:
            context_line = f"Context: {session_summary}\n"
        return (
            "Reply naturally in English.\n"
            "Answer the current user message directly.\n"
            "Start with the answer itself, without meta commentary or role labels.\n"
            "Unless critical information is missing, answer first instead of asking follow-up questions.\n"
            f"{context_line}"
            f"{query}\n"
        )

    def _conversation_rewrite_prompt(self, *, query: str, draft_answer: str, response_language: str) -> str:
        source = (draft_answer or "").strip()
        is_code_heavy = bool(
            "```" in source
            or re.search(r"(?m)^\s*(?:def|class|for|while|if|return|import|from)\b", source)
        )
        draft = (source[:900] if is_code_heavy else re.sub(r"\s+", " ", source)[:260])
        if response_language == "ko":
            return (
                "초안 답변을 자연스러운 한국어 존댓말로 고쳐 쓰세요.\n"
                "사용자 마지막 메시지에 직접 답하고, 메타 문장 없이 답변 본문부터 바로 쓰세요.\n"
                "예고문만 쓰지 말고, 끊긴 문장이 있으면 끝까지 완성하세요.\n"
                f"사용자 질문: {query}\n"
                f"초안 답변: {draft}\n"
                "최종 답변:"
            )
        return (
            "Rewrite the draft answer naturally.\n"
            "Answer the user's latest message directly and start with the answer itself.\n"
            "Do not output meta commentary or an unfinished sentence.\n"
            f"User message: {query}\n"
            f"Draft answer: {draft}\n"
            "Final answer:"
        )

    def _korean_rewrite_prompt(self, *, query: str, draft_answer: str) -> str:
        source = (draft_answer or "").strip()
        is_code_heavy = bool(
            "```" in source
            or re.search(r"(?m)^\s*(?:def|class|for|while|if|return|import|from)\b", source)
        )
        draft = (source[:900] if is_code_heavy else re.sub(r"\s+", " ", source)[:240])
        return (
            "초안 답변을 자연스러운 한국어 존댓말로 고쳐 쓰세요.\n"
            "사용자 마지막 메시지에 직접 답하고, 메타 문장 없이 답변 본문부터 바로 쓰세요.\n"
            "예고문만 쓰지 말고, 끊긴 문장이 있으면 끝까지 완성하세요.\n"
            f"사용자 질문: {query}\n"
            f"초안 답변: {draft}\n"
            "최종 답변:"
        )

    def agentic_system_prompt(self, response_language: str = "ko") -> str:
        if response_language == "ko":
            return (
                "당신은 도구를 사용하는 자율 AI 에이전트입니다.\n"
                "사용자의 요청을 분석하고, 필요한 경우 도구를 호출하세요.\n"
                "사고 과정은 <thought> 태그 안에, 도구 호출은 <action> 태그 안에 JSON 형태로 작성하세요.\n"
                "모든 도구 사용이 끝나거나 도구가 필요 없는 경우 <final_answer> 태그 안에 최종 답변을 작성하세요."
            )
        return (
            "You are an autonomous AI agent with tool-use capabilities.\n"
            "Analyze the user's request and call tools if necessary.\n"
            "Write your reasoning inside <thought> tags, and tool calls inside <action> tags as JSON.\n"
            "When done or if no tools are needed, write your final response inside <final_answer> tags."
        )
