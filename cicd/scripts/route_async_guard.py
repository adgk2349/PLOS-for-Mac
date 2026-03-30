#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    ROOT / "sidecar/local_ai_core/main.py",
    ROOT / "sidecar/local_ai_core/chat.py",
    ROOT / "sidecar/local_ai_core/reasoning/pipeline.py",
    *sorted((ROOT / "sidecar/local_ai_core/reasoning/strategies").glob("*.py")),
]

DB_WRITE_METHODS = {
    "update_workspace",
    "update_settings",
    "update_document_metadata",
    "record_external_call",
    "writeMemoryEvent",
    "clearMemory",
    "pinMemory",
    "unpinMemory",
}


def _expr_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


class AsyncRouteGuard(ast.NodeVisitor):
    def __init__(self, *, path: Path):
        self.path = path
        self.violations: list[str] = []
        self._async_stack: list[str] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._async_stack.append(node.name)
        self.generic_visit(node)
        self._async_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if not self._async_stack:
            self.generic_visit(node)
            return
        fn = node.func
        if isinstance(fn, ast.Attribute):
            owner_text = _expr_text(fn.value)
            method = str(fn.attr)
            fn_text = _expr_text(fn)
            if "_local_inference" in owner_text:
                self.violations.append(
                    f"{self.path}:{node.lineno} async '{self._async_stack[-1]}' has direct inference call: {fn_text}"
                )
            if method in {"execute_conversation", "execute"} and "executor" in owner_text:
                self.violations.append(
                    f"{self.path}:{node.lineno} async '{self._async_stack[-1]}' has sync executor call: {fn_text}"
                )
            if method in DB_WRITE_METHODS and (".db" in owner_text or owner_text.endswith("_db") or owner_text == "db"):
                self.violations.append(
                    f"{self.path}:{node.lineno} async '{self._async_stack[-1]}' has direct db write call: {fn_text}"
                )
        self.generic_visit(node)


def main() -> int:
    failures: list[str] = []
    for path in TARGETS:
        if not path.exists():
            failures.append(f"missing target: {path}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guard = AsyncRouteGuard(path=path)
        guard.visit(tree)
        failures.extend(guard.violations)
    if failures:
        print("route_async_guard failed:")
        for row in failures:
            print(f"- {row}")
        return 1
    print("route_async_guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
