#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BLOCKED_PREFIXES = (
    "data/",
    "sidecar/data/",
    ".build/",
    "DerivedData/",
    "sidecar/.state/",
    "sidecar/.pytest_cache/",
    "sidecar/.idea/",
    ".venv/",
    "sidecar/.venv/",
)

BLOCKED_EXTENSIONS = {
    ".sqlite",
    ".sqlite3",
    ".db",
    ".gguf",
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
}

BLOCKED_FILENAMES = {
    ".env",
}


def run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def repo_root() -> Path:
    proc = run_git(["rev-parse", "--show-toplevel"], cwd=Path.cwd())
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "failed to resolve git root")
    return Path(proc.stdout.strip())


def list_paths(mode: str, *, root: Path) -> list[str]:
    if mode == "tracked":
        proc = run_git(["ls-files", "-z"], cwd=root)
    else:
        proc = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git list failed for mode={mode}")
    raw = proc.stdout
    if not raw:
        return []
    return [item for item in raw.split("\x00") if item]


def is_ignored_by_gitignore(path: str, *, root: Path) -> bool:
    proc = run_git(["check-ignore", "--no-index", "-q", path], cwd=root)
    return proc.returncode == 0


def size_bytes(path: str, *, root: Path) -> int:
    target = (root / path).resolve()
    if not target.exists() or not target.is_file():
        return 0
    return int(target.stat().st_size)


def normalize(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def check_paths(paths: list[str], *, root: Path, size_limit_bytes: int) -> tuple[list[str], list[str], list[str], list[str]]:
    ignored_conflicts: list[str] = []
    blocked_paths: list[str] = []
    blocked_exts: list[str] = []
    oversized: list[str] = []

    for raw in paths:
        path = normalize(raw)
        lowered = path.lower()
        basename = Path(path).name.lower()

        if is_ignored_by_gitignore(path, root=root):
            ignored_conflicts.append(path)

        if any(path.startswith(prefix) for prefix in BLOCKED_PREFIXES):
            blocked_paths.append(path)

        if basename in BLOCKED_FILENAMES:
            blocked_paths.append(path)

        ext = Path(path).suffix.lower()
        if ext in BLOCKED_EXTENSIONS:
            blocked_exts.append(path)

        if size_bytes(path, root=root) > size_limit_bytes:
            oversized.append(path)

    return ignored_conflicts, blocked_paths, blocked_exts, oversized


def main() -> int:
    parser = argparse.ArgumentParser(description="Repository guard for CI and pre-commit.")
    parser.add_argument("--mode", choices=("tracked", "staged"), default="staged")
    parser.add_argument("--size-limit-mb", type=int, default=25)
    args = parser.parse_args()

    try:
        root = repo_root()
    except Exception as exc:
        print(f"[repo-guard] failed: {exc}", file=sys.stderr)
        return 2

    try:
        paths = list_paths(args.mode, root=root)
    except Exception as exc:
        print(f"[repo-guard] failed: {exc}", file=sys.stderr)
        return 2

    if not paths:
        print(f"[repo-guard] mode={args.mode}: no files to check")
        return 0

    size_limit_bytes = max(1, int(args.size_limit_mb)) * 1024 * 1024
    ignored_conflicts, blocked_paths, blocked_exts, oversized = check_paths(
        paths,
        root=root,
        size_limit_bytes=size_limit_bytes,
    )

    has_error = False
    if ignored_conflicts:
        has_error = True
        print("[repo-guard] tracked/staged files matching .gitignore were found:")
        for path in sorted(set(ignored_conflicts)):
            print(f"  - {path}")

    if blocked_paths:
        has_error = True
        print("[repo-guard] blocked path/file policy violation:")
        for path in sorted(set(blocked_paths)):
            print(f"  - {path}")

    if blocked_exts:
        has_error = True
        print("[repo-guard] blocked extension policy violation:")
        for path in sorted(set(blocked_exts)):
            print(f"  - {path}")

    if oversized:
        has_error = True
        print(f"[repo-guard] file size policy violation (> {args.size_limit_mb} MB):")
        for path in sorted(set(oversized)):
            mb = size_bytes(path, root=root) / (1024 * 1024)
            print(f"  - {path} ({mb:.2f} MB)")

    if has_error:
        print("[repo-guard] failed")
        return 1

    print(f"[repo-guard] ok ({len(paths)} files checked, mode={args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

