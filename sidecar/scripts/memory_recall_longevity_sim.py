#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from fastapi.testclient import TestClient


def run(*, turns: int, model_path: str, seed_tag: str) -> dict:
    ts = int(time.time())
    root = Path.cwd().parent
    out_path = root / "results" / f"memory_session_longevity_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LOCAL_AI_DATA_DIR"] = str((root / ".tmp-memory-longevity" / f"{seed_tag}-{ts}").resolve())
    os.environ["LOCAL_AI_SESSION_TOKEN"] = "mem-token"
    os.environ["LOCAL_AI_INFERENCE_TIMEOUT_SECONDS"] = "90"
    os.environ["LOCAL_AI_ROUTE_TIMEOUT_SECONDS"] = "300"

    import local_ai_core.main as main_mod

    importlib.reload(main_mod)
    app = main_mod.create_app()
    conv = f"long-mem-{seed_tag}-{ts}"
    seed_turns = [
        "내 이름은 민수야. 기억해줘.",
        "내가 좋아하는 음료는 라떼야.",
        "내 반려묘 이름은 나비야.",
    ]
    filler = [
        "오늘 할 일 정리 도와줘.",
        "집중력 올리는 팁 2개.",
        "Swift enum 설명 짧게.",
        "파이썬 예외처리 기본 예시.",
        "운동 루틴 초보자용 3개.",
        "식단 관리 핵심 3개.",
        "주말 실내 취미 추천.",
        "짧게 다시 요약해줘.",
    ]
    recalls = [
        ("내 이름 뭐야?", ["민수"]),
        ("내가 좋아하는 음료 뭐였지?", ["라떼"]),
        ("내 고양이 이름 기억나?", ["나비"]),
        ("내 개인 정보 3개 요약해줘.", ["민수", "라떼", "나비"]),
    ]
    rows: list[dict[str, object]] = []
    with TestClient(app) as client:
        headers = {"x-session-token": "mem-token"}
        client.put(
            "/v1/settings",
            headers=headers,
            json={
                "privacy_mode": "HYBRID",
                "startup_profile": "RECOMMENDED",
                "model_profile": "advanced",
                "local_engine": "mlx",
                "mlx_model_path": model_path,
                "language": "ko",
                "hybrid_web_search_enabled": True,
                "session_memory_enabled": True,
                "workspace_memory_enabled": True,
                "local_memory_only": True,
                "workspace_memory_mode": "normal",
                "adaptive_personalization_enabled": True,
            },
        )
        turn = 0
        for query in seed_turns:
            turn += 1
            res = client.post(
                "/v2/chat/local",
                headers=headers,
                json={"query": query, "mode": "GENERAL", "conversation_id": conv, "session_id": conv},
            )
            rows.append({"turn": turn, "query": query, "status": res.status_code})
        filler_turns = max(0, int(turns) - len(seed_turns) - len(recalls))
        for idx in range(filler_turns):
            turn += 1
            query = filler[idx % len(filler)]
            res = client.post(
                "/v2/chat/local",
                headers=headers,
                json={"query": query, "mode": "GENERAL", "conversation_id": conv, "session_id": conv},
            )
            rows.append({"turn": turn, "query": query, "status": res.status_code})
        for query, expected in recalls:
            turn += 1
            res = client.post(
                "/v2/chat/local",
                headers=headers,
                json={"query": query, "mode": "GENERAL", "conversation_id": conv, "session_id": conv},
            )
            text = ""
            metadata = {}
            if res.status_code == 200:
                payload = res.json()
                text = str(payload.get("result_summary") or payload.get("lead") or payload.get("generated_text") or "")
                metadata = dict(payload.get("metadata") or {})
            hit = bool(res.status_code == 200 and all(token.lower() in text.lower() for token in expected))
            rows.append(
                {
                    "turn": turn,
                    "query": query,
                    "status": res.status_code,
                    "hit": hit,
                    "expected": expected,
                    "recovery_path": str(metadata.get("recovery_path") or ""),
                    "recall_path": str(metadata.get("recall_path") or ""),
                    "fact_hit_subject": str(metadata.get("fact_hit_subject") or ""),
                    "fact_miss_reason": str(metadata.get("fact_miss_reason") or ""),
                    "fact_overwrite_blocked": int(metadata.get("fact_overwrite_blocked") or 0),
                    "preview": text[:140],
                }
            )

    recall_rows = [row for row in rows if "hit" in row]
    recall_total = len(recall_rows)
    recall_hit = sum(1 for row in recall_rows if bool(row.get("hit")))
    summary = {
        "turns": len(rows),
        "recall_total": recall_total,
        "recall_hit": recall_hit,
        "recall_accuracy": round((recall_hit / recall_total), 4) if recall_total else None,
        "fact_overwrite_blocked_count": sum(int(row.get("fact_overwrite_blocked") or 0) for row in recall_rows),
        "path_distribution": dict(Counter(str(row.get("recall_path") or "") for row in recall_rows)),
    }
    out = {"summary": summary, "rows": rows}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out_path), **summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--model-path", default="mlx-community/gemma-4-e4b-it-4bit")
    parser.add_argument("--seed-tag", default="default")
    args = parser.parse_args()
    print(json.dumps(run(turns=max(8, args.turns), model_path=args.model_path, seed_tag=str(args.seed_tag)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
