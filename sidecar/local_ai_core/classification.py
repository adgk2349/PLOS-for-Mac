from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .embedding import EmbeddingService
from .models import ClassificationResult

FIXED_CATEGORIES = [
    "학습자료",
    "프로젝트문서",
    "회의록",
    "아이디어",
    "개인메모",
    "참고자료",
    "코드관련",
]


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "학습자료": ("study", "lecture", "class", "course", "learning", "강의", "학습", "정리", "노트"),
    "프로젝트문서": ("project", "spec", "prd", "proposal", "기획", "요구사항", "설계", "로드맵"),
    "회의록": ("meeting", "minutes", "agenda", "sync", "회의", "회의록", "미팅"),
    "아이디어": ("idea", "brainstorm", "concept", "draft", "아이디어", "브레인스토밍"),
    "개인메모": ("memo", "journal", "daily", "private", "개인", "메모", "일기"),
    "참고자료": ("reference", "article", "paper", "guide", "링크", "자료", "참고"),
    "코드관련": ("code", "api", "swift", "python", "typescript", "readme", "source", "리팩토링"),
}

_DOCUMENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "노트형": ("note", "노트", "정리"),
    "회의록형": ("meeting", "minutes", "회의"),
    "기획안형": ("spec", "proposal", "기획", "요구사항"),
    "코드문서형": ("api", "readme", "code", "swift", "python"),
    "참고형": ("reference", "paper", "guide", "참고"),
}

_GENERIC_PATH_WORDS = {
    "users",
    "seungminlee",
    "desktop",
    "development",
    "documents",
    "downloads",
    "workspace",
    "study",
    "project",
}

_TAG_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "note",
    "notes",
    "draft",
    "temp",
    "final",
    "ver",
    "version",
    "문서",
    "정리",
    "파일",
}


@dataclass(slots=True)
class RuleSignal:
    category: str
    document_type: str
    tags: list[str]
    year: int | None
    project: str | None
    summary: str
    importance: float


class DocumentClassifier:
    def __init__(self, embedding_service: EmbeddingService, local_inference):
        self._embedding = embedding_service
        self._local_inference = local_inference
        self._category_vectors = {
            category: self._embedding.embed_query(
                f"{category} " + " ".join(_CATEGORY_KEYWORDS[category])
            )
            for category in FIXED_CATEGORIES
        }

    def classify(self, path: Path, text: str) -> ClassificationResult:
        compact_text = re.sub(r"\s+", " ", text).strip()
        signal = self._rule_based(path, compact_text)
        semantic_category = self._semantic_category(compact_text)
        llm_payload = self._call_local_classifier(path, compact_text, signal, semantic_category)

        category = self._normalize_category(
            self._pick_str(llm_payload, "category") or signal.category or semantic_category
        )
        subcategory = self._normalize_subcategory(self._pick_str(llm_payload, "subcategory"), signal.tags)
        document_type = self._pick_str(llm_payload, "document_type") or signal.document_type
        if not document_type:
            document_type = self._doc_type_from_category(category)

        tags = self._normalize_tags(
            signal.tags
            + self._extract_tags_from_text(compact_text)
            + self._pick_list(llm_payload, "tags")
        )
        year = self._pick_year(llm_payload, signal.year, compact_text)
        project = self._pick_project(llm_payload, signal.project, path)
        summary = self._pick_summary(llm_payload, signal.summary, compact_text)
        importance = self._pick_importance(llm_payload, signal.importance)

        return ClassificationResult(
            summary=summary,
            category=category,
            subcategory=subcategory,
            document_type=document_type,
            tags=tags,
            year=year,
            project=project,
            importance=importance,
        )

    def _rule_based(self, path: Path, text: str) -> RuleSignal:
        lower_path = path.as_posix().lower()
        lower_text = text.lower()

        weights = {category: 0 for category in FIXED_CATEGORIES}
        for category, keywords in _CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lower_path:
                    weights[category] += 3
                if keyword in lower_text[:5000]:
                    weights[category] += 1

        category = max(weights, key=lambda k: weights[k])
        if weights[category] <= 0:
            category = "참고자료"

        document_type = ""
        for doc_type, keywords in _DOCUMENT_TYPE_KEYWORDS.items():
            if any(keyword in lower_path for keyword in keywords) or any(
                keyword in lower_text[:4000] for keyword in keywords
            ):
                document_type = doc_type
                break

        tags = self._normalize_tags(self._filename_tags(path) + self._extract_tags_from_text(text))
        summary = self._summarize(text)
        year = self._extract_year(path.as_posix() + " " + text[:2000])
        project = self._project_from_path(path)
        importance = 0.65 if any(k in lower_text for k in ("핵심", "중요", "must", "critical")) else 0.5

        return RuleSignal(
            category=category,
            document_type=document_type,
            tags=tags,
            year=year,
            project=project,
            summary=summary,
            importance=importance,
        )

    def _semantic_category(self, text: str) -> str:
        if not text.strip():
            return "참고자료"
        vector = self._embedding.embed_query(text[:1800])
        scored: list[tuple[str, float]] = []
        for category, prototype in self._category_vectors.items():
            score = self._cosine(vector, prototype)
            scored.append((category, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        if not scored:
            return "참고자료"
        top_category, top_score = scored[0]
        if top_score < 0.08:
            return "참고자료"
        return top_category

    def _call_local_classifier(
        self,
        path: Path,
        text: str,
        signal: RuleSignal,
        semantic_category: str,
    ) -> dict:
        try:
            payload = self._local_inference.classify_document(
                path=str(path),
                text=text[:6000],
                fixed_categories=FIXED_CATEGORIES,
                fallback={
                    "category": signal.category or semantic_category,
                    "document_type": signal.document_type or self._doc_type_from_category(signal.category),
                    "tags": signal.tags[:5],
                    "summary": signal.summary,
                    "year": signal.year,
                    "project": signal.project,
                    "importance": signal.importance,
                },
            )
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    @staticmethod
    def _pick_str(payload: dict, key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _pick_list(payload: dict, key: str) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    @staticmethod
    def _pick_year(payload: dict, fallback_year: int | None, text: str) -> int | None:
        raw = payload.get("year")
        if isinstance(raw, int) and 1990 <= raw <= 2100:
            return raw
        if isinstance(raw, str):
            match = re.search(r"(19|20)\d{2}", raw)
            if match:
                return int(match.group(0))
        extracted = DocumentClassifier._extract_year(text)
        return extracted or fallback_year

    @staticmethod
    def _pick_project(payload: dict, fallback_project: str | None, path: Path) -> str | None:
        direct = payload.get("project")
        if isinstance(direct, str) and direct.strip():
            return DocumentClassifier._clean_label(direct.strip())
        return fallback_project or DocumentClassifier._project_from_path(path)

    @staticmethod
    def _pick_summary(payload: dict, fallback_summary: str, text: str) -> str:
        direct = payload.get("summary")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()[:260]
        if fallback_summary.strip():
            return fallback_summary.strip()[:260]
        return DocumentClassifier._summarize(text)

    @staticmethod
    def _pick_importance(payload: dict, fallback: float) -> float:
        raw = payload.get("importance")
        try:
            if raw is not None:
                return max(0.0, min(1.0, float(raw)))
        except Exception:
            pass
        return max(0.0, min(1.0, float(fallback)))

    @staticmethod
    def _normalize_category(candidate: str | None) -> str:
        if not candidate:
            return "참고자료"
        normalized = candidate.strip()
        if normalized in FIXED_CATEGORIES:
            return normalized
        lower = normalized.lower()
        for category in FIXED_CATEGORIES:
            if lower == category.lower():
                return category
        for category, keywords in _CATEGORY_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                return category
        return "참고자료"

    @staticmethod
    def _normalize_subcategory(candidate: str | None, tags: list[str]) -> str:
        if candidate and candidate.strip():
            return DocumentClassifier._clean_label(candidate)[:40]
        if tags:
            return DocumentClassifier._clean_label(tags[0])[:40]
        return ""

    @staticmethod
    def _normalize_tags(raw_tags: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in raw_tags:
            token = DocumentClassifier._clean_label(raw)
            if not token:
                continue
            key = token.lower()
            if key in seen or key in _TAG_STOPWORDS:
                continue
            if len(key) < 2:
                continue
            seen.add(key)
            normalized.append(token)
            if len(normalized) >= 8:
                break
        return normalized

    @staticmethod
    def _filename_tags(path: Path) -> list[str]:
        stem = path.stem
        parts = re.split(r"[_\-\s\.\(\)\[\]]+", stem)
        return [part for part in parts if part]

    @staticmethod
    def _extract_tags_from_text(text: str) -> list[str]:
        if not text:
            return []
        words = re.findall(r"[A-Za-z가-힣0-9\+#]{2,24}", text[:2500])
        ranked: list[str] = []
        freq: dict[str, int] = {}
        for word in words:
            lower = word.lower()
            if lower in _TAG_STOPWORDS:
                continue
            freq[word] = freq.get(word, 0) + 1
        for word, _ in sorted(freq.items(), key=lambda item: item[1], reverse=True)[:14]:
            ranked.append(word)
        return ranked

    @staticmethod
    def _extract_year(text: str) -> int | None:
        match = re.search(r"(19|20)\d{2}", text)
        if not match:
            return None
        year = int(match.group(0))
        return year if 1990 <= year <= 2100 else None

    @staticmethod
    def _project_from_path(path: Path) -> str | None:
        for part in reversed(path.parts):
            lower = part.lower()
            if lower in _GENERIC_PATH_WORDS:
                continue
            if lower.startswith("."):
                continue
            cleaned = DocumentClassifier._clean_label(part)
            if cleaned and len(cleaned) >= 3:
                return cleaned[:48]
        return None

    @staticmethod
    def _summarize(text: str) -> str:
        if not text.strip():
            return ""
        compact = re.sub(r"\s+", " ", text).strip()
        sentence = re.split(r"(?<=[.!?])\s+|\n", compact)[0].strip()
        summary = sentence if sentence else compact[:220]
        return summary[:260]

    @staticmethod
    def _doc_type_from_category(category: str) -> str:
        mapping = {
            "학습자료": "노트형",
            "프로젝트문서": "기획안형",
            "회의록": "회의록형",
            "아이디어": "노트형",
            "개인메모": "노트형",
            "참고자료": "참고형",
            "코드관련": "코드문서형",
        }
        return mapping.get(category, "참고형")

    @staticmethod
    def _clean_label(value: str) -> str:
        cleaned = re.sub(r"[^\w가-힣#+\- ]+", " ", value, flags=re.UNICODE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = 0.0
        norm_left = 0.0
        norm_right = 0.0
        for l, r in zip(left, right, strict=True):
            dot += l * r
            norm_left += l * l
            norm_right += r * r
        if norm_left <= 0 or norm_right <= 0:
            return 0.0
        return dot / ((norm_left ** 0.5) * (norm_right ** 0.5))
