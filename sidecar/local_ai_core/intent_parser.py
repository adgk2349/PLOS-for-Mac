from __future__ import annotations

import re

from .models import (
    ParsedEntities,
    ParsedIntent,
    ParsedTimeFilters,
    ParsedWorkspaceFilters,
    ReasoningIntent,
    WorkMode,
    WorkspaceResponse,
)

_FILE_PATTERN = re.compile(r"([A-Za-z0-9_\-\.]+\.(?:txt|md|markdown|pdf|docx|py|swift|json|yaml|yml))", re.IGNORECASE)
_TAG_PATTERN = re.compile(r"#([A-Za-z가-힣0-9_+\-]{2,24})")
_YEAR_PATTERN = re.compile(r"(19|20)\d{2}")
_YEAR_RANGE_PATTERN = re.compile(r"(19|20)\d{2}\s*[-~]\s*(19|20)\d{2}")
_PROJECT_PATTERN = re.compile(r"(?:project|프로젝트)\s*[:\-]?\s*([A-Za-z가-힣0-9 _\-]{2,40})", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[A-Za-z가-힣0-9_+\-]{2,24}")

_STOPWORDS = {
    "please",
    "find",
    "file",
    "files",
    "document",
    "documents",
    "summary",
    "summarize",
    "compare",
    "explain",
    "draft",
    "classify",
    "질문",
    "요약",
    "비교",
    "설명",
    "파일",
    "문서",
    "찾아",
    "정리",
    "해줘",
    "해주세요",
    "지금",
    "있지",
    "있어",
    "있는",
    "것",
    "관련",
}


class IntentParser:
    def parse(
        self,
        *,
        query: str,
        mode: WorkMode,
        workspace: WorkspaceResponse,
    ) -> ParsedIntent:
        text = (query or "").strip()
        lowered = text.lower()

        intent = self._detect_intent(lowered, mode)
        entities = self._extract_entities(text)
        time_filters = self._extract_time_filters(text, lowered)
        workspace_filters = ParsedWorkspaceFilters(
            included_paths=list(workspace.included_paths),
            excluded_paths=list(workspace.excluded_paths),
        )
        confidence = self._confidence(intent=intent, entities=entities, query=text)

        return ParsedIntent(
            intent=intent,
            entities=entities,
            time_filters=time_filters,
            workspace_filters=workspace_filters,
            confidence=confidence,
        )

    @staticmethod
    def _detect_intent(lowered_query: str, mode: WorkMode) -> ReasoningIntent:
        if mode == WorkMode.SUMMARY:
            return ReasoningIntent.SUMMARIZE_FILE
        if mode == WorkMode.WRITING:
            return ReasoningIntent.DRAFT_EDIT
        if mode == WorkMode.PLANNING:
            return ReasoningIntent.EXPLAIN_CONTENT

        if any(token in lowered_query for token in ("그거 말고", "다른 거", "다음 거", "other one", "next one")):
            return ReasoningIntent.NEXT_CANDIDATE

        if any(
            token in lowered_query
            for token in (
                "요약만",
                "열어줘",
                "열기",
                "비교해봐",
                "비교만",
                "후속 질문",
                "재질문",
                "질문 예시",
                "질문 3개",
                "제안해줘",
                "just summarize",
                "open it",
                "follow-up questions",
            )
        ):
            return ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST

        if any(token in lowered_query for token in ("1주차", "2주차", "3주차", "4주차", "5주차", "6주차", "7주차", "8주차", "최근 것만", "pdf로")):
            return ReasoningIntent.FOLLOWUP_REFINE

        if any(token in lowered_query for token in ("그거?", "그 파일?", "이거지", "right?", "that one?")):
            return ReasoningIntent.SOFT_CONFIRM

        if any(token in lowered_query for token in ("이어서", "그럼 다음은", "더 보여줘", "continue", "show more")):
            return ReasoningIntent.CONTINUE_PREVIOUS_RESULT

        followup_clues = ("그거", "이거", "방금", "앞에", "that", "those", "it", "previous", "above")
        if any(clue in lowered_query for clue in followup_clues) and len(lowered_query.split()) <= 14:
            return ReasoningIntent.FOLLOWUP_QUESTION

        broad_list_patterns = (
            "뭐뭐",
            "무슨",
            "무엇",
            "어떤",
            "있지",
            "있어",
            "뭐있어",
            "몇주차",
            "몇 주차",
            "몇개",
            "몇 개",
            "목록",
            "리스트",
            "what do i have",
            "what is there",
            "list",
        )
        domain_nouns = (
            "강의",
            "노트",
            "문서",
            "파일",
            "자료",
            "폴더",
            "디렉토리",
            "주차",
            "lecture",
            "note",
            "file",
            "document",
            "folder",
            "directory",
            "week",
        )
        if any(p in lowered_query for p in broad_list_patterns) and any(noun in lowered_query for noun in domain_nouns):
            return ReasoningIntent.FIND_FILE

        classify_clues = ("classify", "분류", "태그", "category", "카테고리")
        if any(clue in lowered_query for clue in classify_clues):
            return ReasoningIntent.CLASSIFY

        draft_clues = ("draft", "rewrite", "edit", "작성", "다듬", "초안", "고쳐")
        if any(clue in lowered_query for clue in draft_clues):
            return ReasoningIntent.DRAFT_EDIT

        compare_clues = ("compare", "차이", "비교", "diff", "공통점")
        if any(clue in lowered_query for clue in compare_clues):
            return ReasoningIntent.COMPARE_FILES

        summarize_clues = (
            "summary",
            "summarize",
            "요약",
            "핵심",
            "중요",
            "important",
            "시험",
            "정리",
            "뭐였지",
            "key point",
            "what mattered",
        )
        if any(clue in lowered_query for clue in summarize_clues):
            return ReasoningIntent.SUMMARIZE_FILE

        file_find_clues = ("find", "locate", "where", "찾아", "어디", "경로", "파일", "폴더", "디렉토리", "folder", "directory")
        if any(clue in lowered_query for clue in file_find_clues):
            return ReasoningIntent.FIND_FILE

        if IntentParser._is_general_chat(lowered_query):
            return ReasoningIntent.GENERAL_CHAT

        return ReasoningIntent.EXPLAIN_CONTENT

    @staticmethod
    def _is_general_chat(lowered_query: str) -> bool:
        if not lowered_query:
            return False
        social_cues = (
            "안녕",
            "반가워",
            "고마워",
            "감사",
            "오늘 어때",
            "잘 지내",
            "what can you do",
            "who are you",
            "hello",
            "hi",
            "hey",
            "thanks",
            "thank you",
            "how are you",
        )
        conversational_cues = (
            "배고파",
            "뭐 먹",
            "추천",
            "어때",
            "심심",
            "피곤",
            "졸려",
            "기분",
            "대화",
            "잡담",
            "농담",
            "위로",
            "hungry",
            "what should i eat",
            "recommend",
            "bored",
            "chat",
        )
        task_cues = (
            "요약",
            "정리",
            "비교",
            "분석",
            "파일",
            "폴더",
            "문서",
            "검색",
            "찾아",
            "주차",
            "연도",
            "태그",
            "요청",
            "summarize",
            "summary",
            "compare",
            "analysis",
            "file",
            "folder",
            "document",
            "find",
            "search",
            "tag",
            "year",
            "week",
        )
        if any(cue in lowered_query for cue in task_cues):
            return False
        if any(cue in lowered_query for cue in social_cues):
            return True
        if any(cue in lowered_query for cue in conversational_cues):
            return True
        token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered_query))
        if token_count <= 3 and lowered_query.endswith(("?", "!", "요", "냐", "니", "파", "데", "까")):
            return True
        anchor_cues = (
            "파일",
            "문서",
            "폴더",
            "디렉토리",
            "주차",
            "연도",
            "태그",
            "검색",
            "project",
            "file",
            "document",
            "folder",
            "directory",
            "week",
            "year",
            "tag",
            "search",
        )
        if token_count <= 7 and not any(cue in lowered_query for cue in anchor_cues):
            return True
        return False

    @staticmethod
    def _extract_entities(query: str) -> ParsedEntities:
        file_names = [match.group(1) for match in _FILE_PATTERN.finditer(query)]
        tags = [match.group(1).strip() for match in _TAG_PATTERN.finditer(query)]

        projects: list[str] = []
        project_match = _PROJECT_PATTERN.search(query)
        if project_match:
            projects.append(project_match.group(1).strip())

        topics: list[str] = []
        for token in _TOKEN_PATTERN.findall(query):
            key = token.lower()
            if key in _STOPWORDS:
                continue
            if token in file_names:
                continue
            topics.append(token)
            if len(topics) >= 6:
                break

        return ParsedEntities(
            file_names=list(dict.fromkeys(file_names)),
            tags=list(dict.fromkeys(tags)),
            topics=list(dict.fromkeys(topics)),
            projects=list(dict.fromkeys(projects)),
        )

    @staticmethod
    def _extract_time_filters(query: str, lowered_query: str) -> ParsedTimeFilters:
        year = None
        year_from = None
        year_to = None
        relative_days = None

        range_match = _YEAR_RANGE_PATTERN.search(query)
        if range_match:
            year_from = int(range_match.group(0)[:4])
            year_to = int(range_match.group(0)[-4:])
        else:
            year_match = _YEAR_PATTERN.search(query)
            if year_match:
                year = int(year_match.group(0))

        if "지난주" in lowered_query or "last week" in lowered_query:
            relative_days = 7
        elif "지난달" in lowered_query or "last month" in lowered_query:
            relative_days = 30
        elif "올해" in lowered_query or "this year" in lowered_query:
            # This still routes through year filter using explicit year when available in metadata.
            relative_days = 365

        return ParsedTimeFilters(
            year=year,
            year_from=year_from,
            year_to=year_to,
            relative_days=relative_days,
        )

    @staticmethod
    def _confidence(intent: ReasoningIntent, entities: ParsedEntities, query: str) -> float:
        score = 0.45
        if intent in {
            ReasoningIntent.FIND_FILE,
            ReasoningIntent.SUMMARIZE_FILE,
            ReasoningIntent.COMPARE_FILES,
            ReasoningIntent.DRAFT_EDIT,
            ReasoningIntent.CLASSIFY,
            ReasoningIntent.FOLLOWUP_REFINE,
            ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
            ReasoningIntent.SOFT_CONFIRM,
            ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
            ReasoningIntent.NEXT_CANDIDATE,
            ReasoningIntent.REDUCE_SCOPE,
            ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
            ReasoningIntent.OPEN_FILE,
        }:
            score += 0.15
        if entities.file_names:
            score += 0.15
        if entities.tags or entities.projects:
            score += 0.1
        if len(query.split()) >= 12:
            score += 0.08
        return max(0.2, min(score, 0.98))
