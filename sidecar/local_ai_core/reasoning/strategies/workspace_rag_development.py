from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from ...models import Citation, SystemFilePermission, WorkMode


class WorkspaceRagDevelopment:
    @staticmethod
    def extract_json_value(raw: str) -> dict[str, Any] | list[Any] | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj >= 0 and end_obj > start_obj:
            snippet = text[start_obj : end_obj + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr >= 0 and end_arr > start_arr:
            snippet = text[start_arr : end_arr + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return None

    @staticmethod
    def parse_line_ref(line_ref: str | None) -> int | None:
        text = str(line_ref or "").strip()
        if not text:
            return None
        match = re.search(r"(?:^|[^\d])(\d{1,6})(?:[^\d]|$)", text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    @classmethod
    def issue_from_citation(cls, citation: Citation) -> dict[str, Any]:
        line_number = cls.parse_line_ref(citation.snippet) or 0
        line_ref = f"L{line_number}" if line_number > 0 else ""
        return {
            "severity": "P2",
            "file_path": str(citation.file_path or ""),
            "line_ref": line_ref,
            "summary": str(citation.snippet or "").strip()[:220] or "Potential issue detected from grounded snippet.",
            "fix_hint": "Review this location and apply the smallest safe change with tests.",
            "evidence": str(citation.snippet or "").strip()[:280],
            "confidence": round(float(citation.score or 0.0), 4),
        }

    @classmethod
    def generate_batch_review_issues(
        cls,
        *,
        executor,
        settings,
        workspace,
        batch_citations: list[Citation],
    ) -> tuple[list[dict[str, Any]], str | None, bool]:
        prompt = (
            "You are a strict code reviewer. Return JSON only.\n"
            "Schema: {\"issues\":[{\"severity\":\"P0|P1|P2|P3\",\"file_path\":\"...\",\"line_ref\":\"L12\",\"summary\":\"...\","
            "\"fix_hint\":\"...\",\"evidence\":\"...\",\"confidence\":0.0}]}\n"
            "Rules: grounded only, no hallucinations, no markdown."
        )
        inference = executor._local_inference.generate(
            query=prompt,
            mode=WorkMode.DEVELOPMENT,
            citations=batch_citations,
            profile=workspace.startup_profile.value,
            engine=settings.local_engine,
            mlx_model_path=settings.mlx_model_path,
            llama_model_path=settings.llama_model_path,
            language_preference=settings.language,
            max_tokens=640,
        )
        payload = cls.extract_json_value(inference.answer)
        issues: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            raw_issues = payload.get("issues")
            if isinstance(raw_issues, list):
                for row in raw_issues[:24]:
                    if not isinstance(row, dict):
                        continue
                    file_path = str(row.get("file_path") or "").strip()
                    evidence = str(row.get("evidence") or "").strip()
                    if not file_path or not evidence:
                        continue
                    issues.append(
                        {
                            "severity": str(row.get("severity") or "P2").upper(),
                            "file_path": file_path,
                            "line_ref": str(row.get("line_ref") or "").strip(),
                            "summary": str(row.get("summary") or "").strip()[:260],
                            "fix_hint": str(row.get("fix_hint") or "").strip()[:320],
                            "evidence": evidence[:320],
                            "confidence": float(row.get("confidence") or 0.0),
                        }
                    )
        used_fallback = False
        if not issues:
            used_fallback = True
            issues = [cls.issue_from_citation(item) for item in batch_citations[:5]]
        return issues, inference.detail, bool(inference.used_fallback or used_fallback)

    @staticmethod
    def build_patch_plan(issues: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for item in issues:
            evidence = str(item.get("evidence") or "").strip()
            file_path = str(item.get("file_path") or "").strip()
            if not evidence or not file_path:
                continue
            severity = str(item.get("severity") or "P2").upper()
            risk_level = "low" if severity in {"P2", "P3"} else ("medium" if severity == "P1" else "high")
            output.append(
                {
                    "file_path": file_path,
                    "line_ref": str(item.get("line_ref") or "").strip(),
                    "change_type": "targeted_fix",
                    "proposed_diff_summary": str(item.get("fix_hint") or item.get("summary") or "").strip()[:320],
                    "risk_level": risk_level,
                    "evidence": evidence[:280],
                }
            )
            if len(output) >= max_items:
                break
        return output

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

    @classmethod
    def apply_patch_items(
        cls,
        *,
        executor,
        settings,
        workspace,
        query: str,
        patch_plan: list[dict[str, Any]],
    ) -> tuple[int, int, list[str]]:
        if settings.system_file_permission != SystemFilePermission.FULL_ACCESS:
            return 0, len(patch_plan), ["patch_apply:denied_permission"]

        applied = 0
        failed = 0
        changed_files: list[str] = []
        for item in patch_plan:
            file_path = str(item.get("file_path") or "").strip()
            if not file_path:
                failed += 1
                continue
            try:
                target = Path(file_path).expanduser().resolve()
                if not target.exists() or not target.is_file():
                    failed += 1
                    continue
                text = target.read_text(encoding="utf-8", errors="ignore")
                lines = text.splitlines()
                if not lines:
                    failed += 1
                    continue
                line_no = cls.parse_line_ref(str(item.get("line_ref") or "")) or 1
                line_no = max(1, min(line_no, len(lines)))
                start = max(1, line_no - 2)
                end = min(len(lines), line_no + 2)
                excerpt = "\n".join(lines[start - 1 : end])
                prompt = cls.build_patch_prompt(query=query, issue=item, source_excerpt=excerpt)
                inference = executor._local_inference.generate(
                    query=prompt,
                    mode=WorkMode.DEVELOPMENT,
                    citations=[],
                    profile=workspace.startup_profile.value,
                    engine=settings.local_engine,
                    mlx_model_path=settings.mlx_model_path,
                    llama_model_path=settings.llama_model_path,
                    language_preference=settings.language,
                    max_tokens=320,
                )
                payload = cls.extract_json_value(inference.answer)
                if not isinstance(payload, dict):
                    failed += 1
                    continue
                new_block = str(payload.get("new_block") or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
                if not new_block:
                    failed += 1
                    continue
                replacement = new_block.split("\n")
                new_lines = list(lines)
                new_lines[start - 1 : end] = replacement
                updated = "\n".join(new_lines)
                if text.endswith("\n"):
                    updated += "\n"
                if updated == text:
                    failed += 1
                    continue
                target.write_text(updated, encoding="utf-8")
                applied += 1
                changed_files.append(str(target))
            except (OSError, ValueError, TypeError):
                failed += 1
        return applied, failed, changed_files
