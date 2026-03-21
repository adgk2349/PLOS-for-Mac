from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from statistics import mean, median

from fastapi.testclient import TestClient


QUERY_POOL = [
    "오늘 저녁 뭐 먹을까?",
    "집중 안 될 때 20분 리셋 방법 3개",
    "파이썬 dataclass 언제 쓰면 좋아?",
    "Swift에서 struct와 class 차이 핵심만 알려줘",
    "Git merge conflict 빠르게 푸는 순서",
    "React useEffect 의존성 배열 기준 알려줘",
    "디버깅할 때 print 말고 먼저 볼 것 3가지",
    "면접 전날 준비 체크리스트 3개",
]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return ordered[idx]


def run_benchmark(*, engine: str, model_path: str, turns: int, seed: int) -> dict:
    random.seed(seed)

    with tempfile.TemporaryDirectory(prefix="plos-perf-") as td:
        data_dir = Path(td) / "data"
        workspace = Path(td) / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "perf_dummy.txt").write_text("performance benchmark workspace", encoding="utf-8")

        os.environ["LOCAL_AI_DATA_DIR"] = str(data_dir)
        os.environ["LOCAL_AI_SESSION_TOKEN"] = "perf-token"

        import local_ai_core.main as m

        importlib.reload(m)
        app = m.create_app()

        settings_payload = {
            "privacy_mode": "LOCAL_ONLY",
            "startup_profile": "RECOMMENDED",
            "model_profile": "advanced",
            "local_engine": engine,
            "llama_model_path": model_path if engine == "llama_cpp" else None,
            "mlx_model_path": model_path if engine == "mlx" else None,
            "reindex_policy": "filewatch_incremental",
            "language": "ko",
            "action_permission_mode": "ASK_PER_ACTION",
            "adaptive_personalization_enabled": True,
            "session_memory_enabled": True,
            "workspace_memory_enabled": True,
            "local_memory_only": True,
            "workspace_memory_mode": "normal",
        }

        latencies_ms: list[float] = []
        result_types: dict[str, int] = {}
        success_count = 0

        with TestClient(app) as client:
            headers = {"x-session-token": "perf-token"}

            client.put("/v1/settings", headers=headers, json=settings_payload)
            client.post(
                "/v1/workspaces",
                headers=headers,
                json={
                    "included_paths": [str(workspace)],
                    "excluded_paths": [],
                    "startup_profile": "RECOMMENDED",
                    "default_mode": "GENERAL",
                },
            )
            job = client.post("/v1/index/jobs", headers=headers, json={"scope": "full"})
            job_id = job.json().get("job_id")
            if job_id:
                for _ in range(180):
                    status_payload = client.get(f"/v1/index/jobs/{job_id}", headers=headers).json()
                    if status_payload.get("status") in {"completed", "failed"}:
                        break

            for idx in range(turns):
                query = random.choice(QUERY_POOL)
                started = time.perf_counter()
                res = client.post(
                    "/v2/chat/local",
                    headers=headers,
                    json={
                        "query": query,
                        "mode": "GENERAL",
                        "conversation_id": f"perf-{idx}",
                        "top_k": 6,
                        "filters": None,
                    },
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                latencies_ms.append(elapsed_ms)

                payload = res.json()
                result_type = (payload.get("structured_result") or {}).get("result_type", "unknown")
                result_types[result_type] = result_types.get(result_type, 0) + 1
                if result_type != "runtime_error":
                    success_count += 1

        return {
            "turns": turns,
            "engine": engine,
            "model_path": model_path,
            "avg_ms": round(mean(latencies_ms), 2) if latencies_ms else 0.0,
            "p50_ms": round(median(latencies_ms), 2) if latencies_ms else 0.0,
            "p95_ms": round(_p95(latencies_ms), 2) if latencies_ms else 0.0,
            "success_rate": round(success_count / turns, 4) if turns > 0 else 0.0,
            "result_type_counts": result_types,
            "seed": seed,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="PLOS sidecar conversational latency benchmark")
    parser.add_argument("--engine", choices=["llama_cpp", "mlx"], default="llama_cpp")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_benchmark(
        engine=args.engine,
        model_path=args.model_path,
        turns=max(1, args.turns),
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
