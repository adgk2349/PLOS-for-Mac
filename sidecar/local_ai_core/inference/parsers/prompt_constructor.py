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
        is_recommendation_query = self._is_recommendation_chat_query(query)
        is_coding_query = bool(
            re.search(
                r"(?is)\b(code|python|swift|javascript|java|c\+\+|algorithm|leetcode|two\s*sum|bug|debug|fix|함수|코드|문제\s*풀|풀이)\b",
                query or "",
            )
        )
        ko_tone = ""
        if response_language == "ko":
            ko_tone = (
                "한국어 규칙: 자연스러운 존댓말로 답하세요. "
                "공문체/정책문/로그 문구를 피하고, 첫 문장은 바로 답변 본문으로 시작하세요.\n"
            )
        direct_first_rule = (
            "Direct-first rule: Start with a concrete answer in the first sentence. "
            "Do not respond with only a question.\n"
        )
        recommendation_rule = ""
        if is_recommendation_query:
            if response_language == "ko":
                recommendation_rule = (
                    "추천/선택 요청이면 번호 1~3으로 3가지 옵션을 제시하고, "
                    "각 옵션마다 한 줄 근거를 붙이세요. "
                    "확인 질문은 필요할 때만 마지막에 1개 이하로 하세요.\n"
                )
            else:
                recommendation_rule = (
                    "For recommendation/choice requests, provide exactly 3 numbered options "
                    "with a one-line reason each. Ask at most one follow-up question at the end only if essential.\n"
                )
        coding_rule = ""
        if is_coding_query:
            if response_language == "ko":
                coding_rule = (
                    "코딩/알고리즘 질문이면 설명만 하지 말고 실행 가능한 정답 코드를 반드시 포함하세요. "
                    "코드는 ```언어 fenced block```으로 출력하고, 연산자(+,-,*,/)와 줄바꿈을 절대 손상시키지 마세요. "
                    "가능하면 시간복잡도를 한 줄로 덧붙이세요.\n"
                )
            else:
                coding_rule = (
                    "For coding/algorithm questions, include runnable final code (not just explanation). "
                    "Output code inside fenced blocks and preserve operators (+,-,*,/) and line breaks exactly. "
                    "Add one short line for time complexity when applicable.\n"
                )
        context_block = ""
        if session_summary:
            context_block = f"<conversation_memory>\n{session_summary}\n</conversation_memory>\n"
        return (
            "You are a conversational local AI assistant.\n"
            f"{response_language_instruction(response_language)}\n"
            f"{ko_tone}"
            "Do not output system logs. Provide concise, practical help.\n"
            "Never role-play both user and assistant in one response.\n"
            "Do not include labels like 'User:' or 'Assistant:'.\n"
            "Do not invent personal facts (location, identity, background) unless user stated them in this turn.\n"
            "Answer naturally and stay concise by default; expand only when the user asks for detail.\n"
            f"{direct_first_rule}"
            f"{recommendation_rule}"
            f"{coding_rule}"
            "Never repeat instruction text, policy wording, or internal rules in the final answer.\n"
            "If conversation memory is provided, use it silently as background context and never reveal or quote it.\n"
            f"Mode: {mode.value}\n"
            f"{context_block}"
            f"Input message: {query}\n"
            "Answer:"
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
                "다음 초안 답변을 자연스러운 한국어 존댓말로 다시 작성해 주세요.\n"
                "규칙:\n"
                "- 사용자 마지막 질문에 직접 답변\n"
                "- 정책/지시문/메타 문장 금지\n"
                "- 역할 라벨(User/Assistant/You/A) 금지\n"
                "- 같은 문장 반복 금지\n"
                "- 핵심 의미는 유지하고 간결하게 작성\n"
                "- 코드 블록이 있으면 코드 연산자와 줄바꿈을 그대로 보존\n"
                f"사용자 질문: {query}\n"
                f"초안 답변: {draft}\n"
                "최종 답변:"
            )
        return (
            "Rewrite the draft answer naturally.\n"
            "Rules:\n"
            "- Directly answer the user's latest message\n"
            "- No policy text, no meta commentary, no role labels\n"
            "- No repeated sentences\n"
            "- If code is present, preserve operators and line breaks exactly\n"
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
            "다음 초안 답변을 자연스러운 한국어 존댓말로 다시 작성해 주세요.\n"
            "규칙:\n"
            "- 사용자 마지막 질문에 직접 답변\n"
            "- 정책/지시문/메타 문장 금지\n"
            "- 역할 라벨(User/Assistant/You/A) 금지\n"
            "- 같은 문장 반복 금지\n"
            "- 핵심 의미는 유지하고 간결하게 작성\n"
            "- 코드 블록이 있으면 코드 연산자와 줄바꿈을 그대로 보존\n"
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
