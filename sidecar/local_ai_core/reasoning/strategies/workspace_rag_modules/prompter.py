from __future__ import annotations

import re
import json
from typing import Any

from ....models import Citation


class WorkspaceRagPrompter:
    @staticmethod
    def extract_grounded_line_refs(citations: list[Citation]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for citation in citations[:10]:
            path = str(citation.file_path or "").strip()
            if not path:
                continue
            snippet = str(citation.snippet or "")
            line_match = re.search(r"(?:\bline\s*|라인\s*)(\d{1,6})\b|\bL(\d{1,6})\b", snippet, re.IGNORECASE)
            line_number = ""
            if line_match:
                line_number = str(line_match.group(1) or line_match.group(2) or "").strip()
            ref = f"{path}:L{line_number}" if line_number else f"{path}:{citation.chunk_id}"
            if ref in seen:
                continue
            seen.add(ref)
            output.append(ref)
            if len(output) >= 8:
                break
        return output

    @staticmethod
    def ensure_development_answer_template(
        *,
        query: str,
        answer: str,
        citations: list[Citation],
        response_language: str,
    ) -> str:
        text = str(answer or "").strip()
        if not text:
            return text
        lowered = text.lower()
        if "근거 파일/라인" in text or "evidence files/lines" in lowered:
            return text

        file_refs: list[str] = []
        seen_paths: set[str] = set()
        for citation in citations[:4]:
            path = str(citation.file_path or "").strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            file_refs.append(path)

        if response_language == "en":
            refs_text = "\n".join(f"- {item}" for item in file_refs) or "- (no grounded file references)"
            return (
                f"Issue Summary:\n{query.strip()}\n\n"
                f"Evidence files/lines:\n{refs_text}\n\n"
                f"Proposed change:\n{text}\n\n"
                "Validation:\n- Run impacted unit/integration tests\n- Verify no regression in related paths"
            )
        if response_language == "ja":
            refs_text = "\n".join(f"- {item}" for item in file_refs) or "- （根拠ファイルなし）"
            return (
                f"問題要約:\n{query.strip()}\n\n"
                f"根拠ファイル/行:\n{refs_text}\n\n"
                f"修正提案:\n{text}\n\n"
                "検証:\n- 影響範囲のテストを実行\n- 関連パスの回帰を確認"
            )

        refs_text = "\n".join(f"- {item}" for item in file_refs) or "- (근거 파일 없음)"
        return (
            f"문제 요약:\n{query.strip()}\n\n"
            f"근거 파일/라인:\n{refs_text}\n\n"
            f"수정 제안:\n{text}\n\n"
            "검증:\n- 영향 범위 단위/통합 테스트 실행\n- 관련 경로 회귀 여부 확인"
        )

    @staticmethod
    def build_patch_prompt(*, query: str, issue: dict[str, Any], source_excerpt: str) -> str:
        return (
            "Return JSON only.\n"
            "Schema: {\"line_ref\":\"L12\",\"new_block\":\"...\"}\n"
            "Goal: minimal safe code change based on issue.\n"
            f"User request: {query}\n"
            f"Issue: {json.dumps(issue, ensure_ascii=False)}\n"
            "Current excerpt:\n"
            f"{source_excerpt}\n"
        )

    @staticmethod
    def build_batch_review_text(*, query: str, issues: list[dict[str, Any]], response_language: str) -> str:
        if response_language == "en":
            if not issues:
                return "No grounded issues were found for this development review."
            lines = ["Review findings (grounded):"]
            for idx, row in enumerate(issues[:12], start=1):
                lines.append(
                    f"{idx}. [{row.get('severity','P2')}] {row.get('file_path','')} {row.get('line_ref','')}: {row.get('summary','')}"
                )
            return "\n".join(lines)
        if response_language == "ja":
            if not issues:
                return "根拠付きの指摘は見つかりませんでした。"
            lines = ["レビュー結果（根拠あり）:"]
            for idx, row in enumerate(issues[:12], start=1):
                lines.append(
                    f"{idx}. [{row.get('severity','P2')}] {row.get('file_path','')} {row.get('line_ref','')}: {row.get('summary','')}"
                )
            return "\n".join(lines)
        if not issues:
            return "근거 기반으로 확정할 수 있는 코드 이슈를 찾지 못했습니다."
        lines = ["코드리뷰 결과(근거 기반):"]
        for idx, row in enumerate(issues[:12], start=1):
            lines.append(
                f"{idx}. [{row.get('severity','P2')}] {row.get('file_path','')} {row.get('line_ref','')}: {row.get('summary','')}"
            )
        return "\n".join(lines)

