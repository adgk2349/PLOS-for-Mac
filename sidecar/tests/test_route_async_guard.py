from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_route_async_guard_script_passes() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "cicd/scripts/route_async_guard.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
