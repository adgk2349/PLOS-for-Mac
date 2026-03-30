from __future__ import annotations

import re

from ..models import (
    ParsedEntities,
    ParsedIntent,
    ParsedTimeFilters,
    ParsedWorkspaceFilters,
    ReasoningIntent,
    WorkMode,
    WorkspaceResponse,
)

_FILE_PATTERN = re.compile(
    r"([A-Za-z가-힣0-9_+\-().\[\]]+\.(?:txt|md|markdown|pdf|docx|py|swift|json|yaml|yml))",
    re.IGNORECASE,
)
_TAG_PATTERN = re.compile(r"#([A-Za-z가-힣0-9_+\-]{2,24})")
_YEAR_PATTERN = re.compile(r"(19|20)\d{2}")
_YEAR_RANGE_PATTERN = re.compile(r"(19|20)\d{2}\s*[-~]\s*(19|20)\d{2}")
_PROJECT_PATTERN = re.compile(r"(?:project|프로젝트)\s*[:\-]?\s*([A-Za-z가-힣0-9 _\-]{2,40})", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[A-Za-z가-힣0-9_+\-]{2,24}")
_TARGET_WITH_DOMAIN_PATTERN = re.compile(
    r"([A-Za-z가-힣0-9_+\-]{2,24})\s*(?:파일|문서|자료|폴더|디렉토리|file|document|folder|directory)",
    re.IGNORECASE,
)

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
_SCOPE_ALL_TOKENS = ("전체", "전부", "모두", "모든", "all", "every", "entire")
_SCOPE_TOPN_TOKENS = ("상위", "top", "목록", "리스트", "list", "몇개", "몇 개")
_OPERATION_OPEN_TOKENS = ("열어", "열기", "open", "show file", "파일 열어")
_OPERATION_SUMMARIZE_TOKENS = ("요약", "정리", "핵심", "summary", "summarize")
_OPERATION_FIND_TOKENS = (
    "찾아",
    "검색",
    "어디",
    "목록",
    "리스트",
    "파일",
    "문서",
    "폴더",
    "디렉토리",
    "find",
    "search",
    "list",
    "file",
    "document",
    "folder",
    "directory",
)
_FIND_ACTION_TOKENS = (
    "찾아",
    "검색",
    "보여",
    "목록",
    "리스트",
    "어디",
    "열어",
    "뭐뭐",
    "무슨",
    "무엇",
    "뭐있어",
    "있지",
    "있어",
    "몇주차",
    "몇 주차",
    "몇개",
    "몇 개",
    "what do i have",
    "what is there",
    "list",
    "find",
    "search",
    "show",
    "where",
    "open",
)
_FILE_TARGET_TOKENS = (
    "파일",
    "문서",
    "자료",
    "폴더",
    "디렉토리",
    "주차",
    "강의",
    "노트",
    "file",
    "document",
    "folder",
    "directory",
    "week",
    "lecture",
    "note",
)
_SUMMARY_ACTION_TOKENS = (
    "요약",
    "정리",
    "핵심",
    "요점",
    "summary",
    "summarize",
    "key point",
)
_TARGET_NOISE = {
    *_STOPWORDS,
    "전체",
    "전부",
    "모두",
    "모든",
    "상위",
    "top",
    "all",
    "every",
    "핵심",
    "요약해줘",
    "보여줘",
    "찾아줘",
    "해줘",
    "해주세요",
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
        operation = self._infer_operation(lowered_query=lowered, intent=intent)
        scope = self._extract_scope(query=text, lowered_query=lowered, operation=operation)
        target = self._extract_target(query=text, entities=entities, operation=operation)
        ambiguity = self._infer_ambiguity(
            operation=operation,
            scope=scope,
            target=target,
            entities=entities,
        )
        confidence = self._confidence(
            intent=intent,
            entities=entities,
            query=text,
            operation=operation,
            scope=scope,
            ambiguity=ambiguity,
        )

        return ParsedIntent(
            intent=intent,
            entities=entities,
            time_filters=time_filters,
            workspace_filters=workspace_filters,
            confidence=confidence,
            operation=operation,
            target=target,
            scope=scope,
            ambiguity=ambiguity,
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

        if (
            re.search(r"([1-9]|1[0-9]|2[0-4])\s*주차", lowered_query) is not None
            or any(token in lowered_query for token in ("최근 것만", "pdf로"))
        ):
            return ReasoningIntent.FOLLOWUP_REFINE

        if any(token in lowered_query for token in ("그거?", "그 파일?", "이거지", "right?", "that one?")):
            return ReasoningIntent.SOFT_CONFIRM

        if any(token in lowered_query for token in ("이어서", "그럼 다음은", "더 보여줘", "continue", "show more")):
            return ReasoningIntent.CONTINUE_PREVIOUS_RESULT

        followup_clues = ("그거", "이거", "방금", "앞에", "that", "those", "it", "previous", "above")
        if any(clue in lowered_query for clue in followup_clues) and len(lowered_query.split()) <= 14:
            return ReasoningIntent.FOLLOWUP_QUESTION

        if IntentParser._has_explicit_find_request(lowered_query):
            system_cues = ("내 컴퓨터", "전체 검색", "spotlight", "스포트라이트", "전체에서", "컴퓨터에서", "모든 파일")
            if any(token in lowered_query for token in system_cues):
                return ReasoningIntent.SYSTEM_ACTION
            return ReasoningIntent.FIND_FILE

        classify_clues = ("classify", "분류", "태그", "category", "카테고리")
        if any(clue in lowered_query for clue in classify_clues):
            return ReasoningIntent.CLASSIFY

        draft_clues = ("draft", "rewrite", "edit", "작성", "다듬", "초안", "고쳐")
        if any(clue in lowered_query for clue in draft_clues):
            return ReasoningIntent.DRAFT_EDIT

        compare_clues = ("compare", "차이", "비교", "diff", "공통점")
        if any(clue in lowered_query for clue in compare_clues) and IntentParser._has_file_target_token(lowered_query):
            return ReasoningIntent.COMPARE_FILES

        if IntentParser._has_explicit_summary_request(lowered_query):
            return ReasoningIntent.SUMMARIZE_FILE

        file_find_clues = ("find", "locate", "where", "찾아", "어디", "경로")
        if any(clue in lowered_query for clue in file_find_clues) and IntentParser._has_file_target_token(lowered_query):
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
            "잠",
            "자야",
            "몇 시",
            "몇시",
            "새벽",
            "아침에",
            "루틴",
            "습관",
            "고민",
            "괜찮아",
            "괜찮을까",
            "운동하고",
            "목이",
            "아파",
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
            "최신",
            "latest",
            "버전",
            "version",
            "업데이트",
            "update",
            "아이폰",
            "아이패드",
            "맥북",
            "iphone",
            "ipad",
            "macbook",
            "nvidia",
            "엔비디아",
            "삼성",
            "갤럭시",
            "samsung",
            "galaxy",
            "날씨",
            "뉴스",
            "weather",
            "news",
            "알아",
            "알려줘",
            "알고있어",
            "알고있니",
        )
        if IntentParser._has_explicit_find_request(lowered_query) or IntentParser._has_explicit_summary_request(lowered_query):
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
            "project",
            "file",
            "document",
            "folder",
            "directory",
            "week",
            "year",
            "tag",
        )
        if (
            any(token in lowered_query for token in ("검색", "search", "찾아", "find"))
            and not IntentParser._has_file_target_token(lowered_query)
        ):
            return True
        if token_count <= 12 and not any(cue in lowered_query for cue in anchor_cues):
            return True
        return False

    @staticmethod
    def _has_scope_all_token(lowered_query: str) -> bool:
        return any(token in lowered_query for token in _SCOPE_ALL_TOKENS)

    @staticmethod
    def _has_file_target_token(lowered_query: str) -> bool:
        return any(token in lowered_query for token in _FILE_TARGET_TOKENS)

    @staticmethod
    def _has_explicit_find_request(lowered_query: str) -> bool:
        has_target = IntentParser._has_file_target_token(lowered_query)
        has_action = any(token in lowered_query for token in _FIND_ACTION_TOKENS)
        if has_target and has_action:
            return True
        if has_target and IntentParser._has_scope_all_token(lowered_query):
            return True
        return False

    @staticmethod
    def _has_explicit_summary_request(lowered_query: str) -> bool:
        has_summary_action = any(token in lowered_query for token in _SUMMARY_ACTION_TOKENS)
        if has_summary_action and (IntentParser._has_file_target_token(lowered_query) or len(lowered_query.split()) >= 2):
            return True
        important_clues = ("중요", "important", "시험", "what mattered")
        if any(token in lowered_query for token in important_clues) and IntentParser._has_file_target_token(lowered_query):
            return True
        return False

    @staticmethod
    def _infer_operation(*, lowered_query: str, intent: ReasoningIntent) -> str:
        if any(token in lowered_query for token in _OPERATION_OPEN_TOKENS):
            return "open"
        if intent == ReasoningIntent.OPEN_FILE:
            return "open"
        if IntentParser._has_explicit_summary_request(lowered_query):
            return "summarize"
        if intent in {
            ReasoningIntent.SUMMARIZE_FILE,
            ReasoningIntent.COMPARE_FILES,
            ReasoningIntent.EXPLAIN_CONTENT,
            ReasoningIntent.DRAFT_EDIT,
            ReasoningIntent.CLASSIFY,
            ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
        }:
            return "summarize"
        if IntentParser._has_explicit_find_request(lowered_query):
            return "find"
        if intent in {
            ReasoningIntent.FIND_FILE,
            ReasoningIntent.FOLLOWUP_QUESTION,
            ReasoningIntent.FOLLOWUP_REFINE,
            ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
            ReasoningIntent.SOFT_CONFIRM,
            ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
            ReasoningIntent.NEXT_CANDIDATE,
            ReasoningIntent.REDUCE_SCOPE,
        }:
            return "find"
        return "chat"

    @staticmethod
    def _extract_scope(*, query: str, lowered_query: str, operation: str) -> str:
        if any(token in lowered_query for token in _SCOPE_ALL_TOKENS):
            return "all"
        explicit_top_n = re.search(r"\btop\s*\d+\b", lowered_query) is not None
        numeric_count = re.search(r"([2-9]|[1-9]\d)\s*(개|개만|files?|docs?)", lowered_query) is not None
        if explicit_top_n or numeric_count:
            return "top_n"
        if operation in {"find", "summarize"} and any(token in lowered_query for token in _SCOPE_TOPN_TOKENS):
            return "top_n"
        return "single"

    @staticmethod
    def _extract_target(*, query: str, entities: ParsedEntities, operation: str) -> str | None:
        if operation == "chat":
            return None
        if entities.file_names:
            return entities.file_names[0]
        if entities.projects:
            return entities.projects[0]
        if entities.tags:
            return entities.tags[0]
        direct = _TARGET_WITH_DOMAIN_PATTERN.search(query)
        if direct:
            token = (direct.group(1) or "").strip()
            if token and token.lower() not in _TARGET_NOISE:
                return token
        for token in entities.topics:
            cleaned = (token or "").strip()
            if not cleaned:
                continue
            if cleaned.lower() in _TARGET_NOISE:
                continue
            if cleaned.lower() in _SCOPE_ALL_TOKENS:
                continue
            return cleaned
        return None

    @staticmethod
    def _infer_ambiguity(*, operation: str, scope: str, target: str | None, entities: ParsedEntities) -> str:
        if operation == "chat":
            return "clear"
        if target:
            return "clear"
        if entities.file_names or entities.projects or entities.tags:
            return "clear"
        if scope == "all":
            return "unclear"
        return "unclear"

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
    def _confidence(
        *,
        intent: ReasoningIntent,
        entities: ParsedEntities,
        query: str,
        operation: str,
        scope: str,
        ambiguity: str,
    ) -> float:
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
        if operation == "chat":
            score = max(score, 0.72)
        elif operation in {"find", "summarize", "open"}:
            score += 0.06
        if scope == "all":
            score += 0.02
        elif scope == "top_n":
            score += 0.01
        if entities.file_names:
            score += 0.15
        if entities.tags or entities.projects:
            score += 0.1
        if len(query.split()) >= 12:
            score += 0.08
        if ambiguity == "unclear":
            score -= 0.18
        if len(query.split()) <= 3 and ambiguity == "unclear":
            score -= 0.08
        return max(0.2, min(score, 0.98))
