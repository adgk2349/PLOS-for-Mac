from __future__ import annotations

from ... import utils


class GeneralChatRecallGateHelpers:
    @staticmethod
    def has_memory_recall_cue(query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        # Do not route concise follow-up requests ("한 줄 요약/상기") to memory recall.
        # They should stay in conversational flow and summarize the previous assistant turn.
        concise_terms = ("한 줄", "한줄", "한 문장", "한문장", "짧게", "요약", "상기")
        refer_terms = ("방금", "아까", "직전", "그거", "그 답변", "that", "previous")
        if any(t in text for t in concise_terms) and any(t in text for t in refer_terms):
            return False
        return bool(utils._is_memory_recall_query(query))
