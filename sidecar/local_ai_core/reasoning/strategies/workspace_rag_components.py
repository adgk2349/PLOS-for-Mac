from __future__ import annotations

import subprocess
from typing import Any


class WorkspaceRagComponents:
    @staticmethod
    def reduce_and_rank_issues(
        items: list[dict[str, Any]],
        *,
        severity_rank,
    ) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for row in items:
            file_path = str(row.get("file_path") or "").strip()
            summary = str(row.get("summary") or "").strip()
            line_ref = str(row.get("line_ref") or "").strip()
            if not file_path or not summary:
                continue
            key = f"{file_path}|{line_ref}|{summary[:80].lower()}"
            prev = deduped.get(key)
            if prev is None:
                deduped[key] = row
                continue
            prev_conf = float(prev.get("confidence") or 0.0)
            cur_conf = float(row.get("confidence") or 0.0)
            if cur_conf > prev_conf:
                deduped[key] = row
        ranked = list(deduped.values())
        ranked.sort(
            key=lambda row: (
                severity_rank(str(row.get("severity") or "P3")),
                -float(row.get("confidence") or 0.0),
                str(row.get("file_path") or ""),
            )
        )
        return ranked[:60]

    @staticmethod
    def run_patch_verification(changed_files: list[str]) -> tuple[str, list[str]]:
        if not changed_files:
            return "skipped", []
        logs: list[str] = []
        py_files = [p for p in changed_files if str(p).lower().endswith(".py")]
        swift_files = [p for p in changed_files if str(p).lower().endswith(".swift")]
        status = "passed"
        if py_files:
            try:
                proc = subprocess.run(
                    ["python", "-m", "compileall", "-q", *py_files[:20]],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if proc.returncode != 0:
                    status = "failed"
                    logs.append(f"python_compile_failed:{proc.stderr.strip()[:240]}")
                else:
                    logs.append("python_compile:ok")
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                status = "failed"
                logs.append(f"python_compile_error:{str(exc)[:180]}")
        if swift_files:
            try:
                proc = subprocess.run(
                    ["xcrun", "swiftc", "-typecheck", *swift_files[:10]],
                    capture_output=True,
                    text=True,
                    timeout=40,
                )
                if proc.returncode != 0:
                    status = "failed"
                    logs.append(f"swift_typecheck_failed:{proc.stderr.strip()[:240]}")
                else:
                    logs.append("swift_typecheck:ok")
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                status = "failed"
                logs.append(f"swift_typecheck_error:{str(exc)[:180]}")
        if not py_files and not swift_files:
            status = "skipped"
        return status, logs

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
