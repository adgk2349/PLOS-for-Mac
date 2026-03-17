from pathlib import Path

from local_ai_core.classification import DocumentClassifier, FIXED_CATEGORIES
from local_ai_core.embedding import EmbeddingService
from local_ai_core.local_inference import LocalInferenceEngine


def test_classifier_uses_fixed_category_and_limits_tags():
    classifier = DocumentClassifier(EmbeddingService(dim=128), LocalInferenceEngine())
    text = (
        "회의 agenda와 minutes를 정리한 문서입니다. "
        "프로젝트 일정, API 이슈, action item을 포함합니다. "
        "2025년 3월 17일 회의 기록."
    )
    result = classifier.classify(Path("/tmp/workspace/project_meeting_minutes_2025.md"), text)
    assert result.category in FIXED_CATEGORIES
    assert result.year == 2025
    assert len(result.tags) <= 8
    assert 0.0 <= result.importance <= 1.0
