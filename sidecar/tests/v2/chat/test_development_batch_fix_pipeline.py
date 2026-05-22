from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from local_ai_core.models import Citation, LocalChatRequestV2, SystemFilePermission, WorkMode
from local_ai_core.reasoning.strategies.workspace_rag import WorkspaceRagStrategy
from local_ai_core.reasoning.strategies.workspace_rag_development import WorkspaceRagDevelopment
from local_ai_core.reasoning.strategies.workspace_rag_modules import WorkspaceRagRetriever


def _citation(idx: int) -> Citation:
    return Citation(
        doc_id=f"doc-{idx}",
        chunk_id=f"chunk-{idx}",
        file_path=f"/tmp/file_{idx}.py",
        snippet=f"line {idx}: potential issue",
        score=0.8,
        modified_at=datetime.now(timezone.utc),
    )


def test_large_development_request_detection_by_file_count() -> None:
    strategy = WorkspaceRagStrategy()
    retriever = WorkspaceRagRetriever(
        file_limit=strategy._DEVELOPMENT_BATCH_FILE_LIMIT,
        chunks_per_file=strategy._DEVELOPMENT_BATCH_CHUNKS_PER_FILE,
        char_limit=strategy._DEVELOPMENT_BATCH_CHAR_LIMIT,
        max_files=strategy._DEVELOPMENT_BATCH_MAX_FILES,
    )
    req = LocalChatRequestV2(query="코드리뷰", mode=WorkMode.DEVELOPMENT)
    citations = [_citation(i) for i in range(3)]
    file_doc_ids = [f"doc-{i}" for i in range(strategy._DEVELOPMENT_BATCH_TRIGGER_FILES)]

    assert retriever.is_large_development_review_request(
        req=req,
        citations=citations,
        file_doc_ids=file_doc_ids,
    )


def test_build_patch_plan_requires_evidence_and_limits_items() -> None:
    strategy = WorkspaceRagStrategy()
    issues = []
    for idx in range(12):
        issues.append(
            {
                "severity": "P2",
                "file_path": f"/tmp/f_{idx}.py",
                "line_ref": f"L{idx+1}",
                "summary": "needs fix",
                "fix_hint": "replace call",
                "evidence": "grounded snippet",
            }
        )
    issues.append(
        {
            "severity": "P1",
            "file_path": "/tmp/no_evidence.py",
            "line_ref": "L2",
            "summary": "missing evidence",
            "fix_hint": "no-op",
            "evidence": "",
        }
    )

    plan = WorkspaceRagDevelopment.build_patch_plan(issues, max_items=strategy._DEVELOPMENT_PATCH_MAX_ITEMS)
    assert len(plan) == strategy._DEVELOPMENT_PATCH_MAX_ITEMS
    assert all(str(item.get("evidence") or "").strip() for item in plan)


def test_apply_patch_denied_without_full_access() -> None:
    executor = SimpleNamespace()
    settings = SimpleNamespace(system_file_permission=SystemFilePermission.READ_ONLY)
    workspace = SimpleNamespace()

    applied, failed, logs = WorkspaceRagDevelopment.apply_patch_items(
        executor=executor,
        settings=settings,
        workspace=workspace,
        query="fix it",
        patch_plan=[{"file_path": "/tmp/never_write.py", "evidence": "x"}],
    )
    assert applied == 0
    assert failed == 1
    assert "patch_apply:denied_permission" in logs

