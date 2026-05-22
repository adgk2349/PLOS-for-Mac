from __future__ import annotations

from pathlib import Path
from ....models import Citation, LocalChatRequestV2, WorkMode


class WorkspaceRagRetriever:
    _CODE_EXTENSIONS = {
        ".swift", ".m", ".mm", ".h", ".hpp", ".c", ".cc", ".cpp",
        ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
        ".rb", ".php", ".cs", ".scala", ".sql", ".sh", ".zsh", ".bash",
        ".yaml", ".yml", ".json", ".toml", ".ini", ".xml",
    }
    _DEVELOPMENT_BATCH_TRIGGER_FILES = 12
    _DEVELOPMENT_BATCH_TRIGGER_CITATIONS = 36

    def __init__(
        self,
        *,
        file_limit: int,
        chunks_per_file: int,
        char_limit: int,
        max_files: int,
    ) -> None:
        self._file_limit = int(file_limit)
        self._chunks_per_file = int(chunks_per_file)
        self._char_limit = int(char_limit)
        self._max_files = int(max_files)

    @classmethod
    def is_code_file(cls, file_path: str) -> bool:
        return Path(str(file_path or "")).suffix.lower() in cls._CODE_EXTENSIONS

    @classmethod
    def is_large_development_review_request(
        cls,
        *,
        req: LocalChatRequestV2,
        citations: list[Citation],
        file_doc_ids: list[str],
    ) -> bool:
        if req.mode != WorkMode.DEVELOPMENT:
            return False
        if len(file_doc_ids) >= cls._DEVELOPMENT_BATCH_TRIGGER_FILES:
            return True
        if len(citations) >= cls._DEVELOPMENT_BATCH_TRIGGER_CITATIONS:
            return True
        lowered = str(req.query or "").strip().lower()
        trigger_tokens = (
            "전체",
            "프로젝트 전체",
            "전부",
            "full repo",
            "entire project",
            "code review all",
        )
        return any(token in lowered for token in trigger_tokens)

    def build_development_batches(self, citations: list[Citation]) -> list[list[Citation]]:
        grouped: dict[str, list[Citation]] = {}
        for item in citations:
            grouped.setdefault(str(item.doc_id), []).append(item)
        ranked_docs = sorted(
            grouped.items(),
            key=lambda row: max((c.score for c in row[1]), default=0.0),
            reverse=True,
        )[: self._max_files]
        batches: list[list[Citation]] = []
        current: list[Citation] = []
        current_chars = 0
        current_files = 0
        for _, doc_citations in ranked_docs:
            picked = sorted(doc_citations, key=lambda c: c.score, reverse=True)[: self._chunks_per_file]
            picked_chars = sum(len(str(c.snippet or "")) for c in picked)
            next_files = current_files + 1
            if (
                current
                and (
                    next_files > self._file_limit
                    or current_chars + picked_chars > self._char_limit
                )
            ):
                batches.append(current)
                current = []
                current_chars = 0
                current_files = 0
            current.extend(picked)
            current_chars += picked_chars
            current_files += 1
        if current:
            batches.append(current)
        return batches[:10]

