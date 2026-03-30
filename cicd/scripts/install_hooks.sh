#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook_src="$repo_root/cicd/hooks/pre-commit"
hook_dst="$repo_root/.git/hooks/pre-commit"

cp "$hook_src" "$hook_dst"
chmod +x "$hook_dst"
echo "Installed pre-commit hook -> $hook_dst"

