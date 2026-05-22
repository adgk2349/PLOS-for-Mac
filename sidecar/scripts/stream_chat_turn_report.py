from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


QUERY_POOL = [
    "안녕",
    "오늘 저녁 뭐 먹을까?",
    "그냥 너가 하나 정해줘",
    "내일 할 일 3개만 짧게 정리해줘",
    "집중 안될 때 10분 리셋 루틴 추천해줘",
    "카페 갈지 말지 결정해줘",
    "방금 답변 한 줄로 다시 말해줘",
    "오늘 회고 질문 2개만 남겨줘",
]


@dataclass
class TurnRow:
    turn: int
    query: str
    ok: bool
    timeout: bool
    latency_ms: float
    chunk_count: int
    first_chunk_ms: float | None
    done: bool
    runtime_error: bool
    recovery_path: str | None
    runtime_detail: str | None
    generated_head: str
    generated_text: str
    error: str


def _post_stream(*, base_url: str, token: str, payload: dict, timeout_seconds: float | None) -> TurnRow:
    started = time.perf_counter()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v2/chat/local/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-session-token": token,
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    chunk_count = 0
    first_chunk_ms: float | None = None
    done = False
    runtime_error = False
    recovery_path: str | None = None
    runtime_detail: str | None = None
    generated_head = ""
    generated_text = ""
    error = ""
    timed_out = False

    try:
        if timeout_seconds is None or timeout_seconds <= 0:
            resp_ctx = urllib.request.urlopen(req)  # nosec B310
        else:
            resp_ctx = urllib.request.urlopen(req, timeout=timeout_seconds)  # nosec B310
        with resp_ctx as resp:
            for _ in range(4000):
                line = resp.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                event_type = str(event.get("type") or "").strip().lower()
                if event_type == "chunk":
                    text = str(event.get("text") or "")
                    if text:
                        chunk_count += 1
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - started) * 1000.0
                elif event_type == "done":
                    done = True
                    result = event.get("result") or {}
                    execution_result = result.get("execution_result") if isinstance(result.get("execution_result"), dict) else {}
                    generated = str(
                        result.get("generated_text")
                        or execution_result.get("generated_text")
                        or ""
                    )
                    generated_text = generated
                    generated_head = generated[:180]
                    runtime_detail = str(
                        result.get("runtime_detail")
                        or execution_result.get("runtime_detail")
                        or ""
                    ) or None
                    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                    recovery_path = str(metadata.get("recovery_path") or "") or None
                    sr = result.get("structured_result") if isinstance(result.get("structured_result"), dict) else {}
                    data = sr.get("data") if isinstance(sr.get("data"), dict) else {}
                    runtime_error = str(data.get("reason") or "") == "generation_retry_exhausted"
                    break
                elif event_type == "error":
                    error = str(event.get("message") or "stream_error")
                    break
    except urllib.error.URLError as exc:
        # Fallback: if local server is unavailable, run through in-process app stream
        # so benchmark status stays consistent with code changes.
        if "connection refused" in str(exc).lower():
            return _post_stream_inprocess(token=token, payload=payload, timeout_seconds=timeout_seconds)
        error = f"url_error:{exc}"
        if "timed out" in str(exc).lower():
            timed_out = True
    except TimeoutError:
        error = "timeout"
        timed_out = True
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        error = f"exception:{text}"
        if "timed out" in text.lower():
            timed_out = True

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ok = bool(done and not runtime_error and not error)
    return TurnRow(
        turn=0,
        query=str(payload.get("query") or ""),
        ok=ok,
        timeout=timed_out,
        latency_ms=round(elapsed_ms, 1),
        chunk_count=chunk_count,
        first_chunk_ms=round(first_chunk_ms, 1) if first_chunk_ms is not None else None,
        done=done,
        runtime_error=runtime_error,
        recovery_path=recovery_path,
        runtime_detail=runtime_detail,
        generated_head=generated_head,
        generated_text=generated_text,
        error=error,
    )


def _post_stream_inprocess(*, token: str, payload: dict, timeout_seconds: float | None) -> TurnRow:
    started = time.perf_counter()
    chunk_count = 0
    first_chunk_ms: float | None = None
    done = False
    runtime_error = False
    recovery_path: str | None = None
    runtime_detail: str | None = None
    generated_head = ""
    generated_text = ""
    error = ""
    timed_out = False
    try:
        import importlib
        import local_ai_core.main as main_mod
        from fastapi.testclient import TestClient

        importlib.reload(main_mod)
        app = main_mod.create_app()
        with TestClient(app) as client:
            headers = {
                "Content-Type": "application/json",
                "x-session-token": token,
                "Accept": "text/event-stream",
            }
            with client.stream("POST", "/v2/chat/local/stream", json=payload, headers=headers, timeout=timeout_seconds) as resp:
                for raw_line in resp.iter_lines():
                    line = str(raw_line or "").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    event_type = str(event.get("type") or "").strip().lower()
                    if event_type == "chunk":
                        text = str(event.get("text") or "")
                        if text:
                            chunk_count += 1
                            if first_chunk_ms is None:
                                first_chunk_ms = (time.perf_counter() - started) * 1000.0
                    elif event_type == "done":
                        done = True
                        result = event.get("result") or {}
                        execution_result = result.get("execution_result") if isinstance(result.get("execution_result"), dict) else {}
                        generated = str(result.get("generated_text") or execution_result.get("generated_text") or "")
                        generated_text = generated
                        generated_head = generated[:180]
                        runtime_detail = str(result.get("runtime_detail") or execution_result.get("runtime_detail") or "") or None
                        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                        recovery_path = str(metadata.get("recovery_path") or "") or None
                        sr = result.get("structured_result") if isinstance(result.get("structured_result"), dict) else {}
                        data = sr.get("data") if isinstance(sr.get("data"), dict) else {}
                        runtime_error = str(data.get("reason") or "") == "generation_retry_exhausted"
                        break
                    elif event_type == "error":
                        error = str(event.get("message") or "stream_error")
                        break
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        error = f"inprocess_exception:{text}"
        if "timed out" in text.lower():
            timed_out = True

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ok = bool(done and not runtime_error and not error)
    return TurnRow(
        turn=0,
        query=str(payload.get("query") or ""),
        ok=ok,
        timeout=timed_out,
        latency_ms=round(elapsed_ms, 1),
        chunk_count=chunk_count,
        first_chunk_ms=round(first_chunk_ms, 1) if first_chunk_ms is not None else None,
        done=done,
        runtime_error=runtime_error,
        recovery_path=recovery_path,
        runtime_detail=runtime_detail,
        generated_head=generated_head,
        generated_text=generated_text,
        error=error,
    )


def run_report(
    *,
    base_url: str,
    token: str,
    turns: int,
    seed: int,
    timeout_seconds: float | None,
    conversation_id: str,
) -> dict:
    random.seed(seed)
    rows: list[TurnRow] = []
    for turn in range(1, max(1, int(turns)) + 1):
        query = random.choice(QUERY_POOL)
        row = _post_stream(
            base_url=base_url,
            token=token,
            timeout_seconds=timeout_seconds,
            payload={
                "query": query,
                "mode": "GENERAL",
                "conversation_id": conversation_id,
                "session_id": conversation_id,
            },
        )
        row.turn = turn
        rows.append(row)

    latencies = [r.latency_ms for r in rows]
    first_chunks = [r.first_chunk_ms for r in rows if r.first_chunk_ms is not None]
    runtime_error_count = sum(1 for r in rows if r.runtime_error)
    timeout_count = sum(1 for r in rows if r.timeout)
    ok_count = sum(1 for r in rows if r.ok)

    report = {
        "summary": {
            "turns": len(rows),
            "ok_count": ok_count,
            "runtime_error_count": runtime_error_count,
            "timeout_count": timeout_count,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "avg_first_chunk_ms": round(sum(first_chunks) / len(first_chunks), 1) if first_chunks else None,
            "seed": seed,
            "per_turn_timeout_sec": timeout_seconds,
        },
        "rows": [row.__dict__ for row in rows],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn-level stream chat reporter (writes JSON report)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--token", default="sidecar-dev-token")
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=float, default=0.0, help="0 means no per-turn timeout")
    parser.add_argument("--conversation-id", default="stream-turn-report")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = run_report(
        base_url=args.base_url,
        token=args.token,
        turns=max(1, args.turns),
        seed=args.seed,
        timeout_seconds=(None if float(args.timeout_sec) <= 0 else max(5.0, float(args.timeout_sec))),
        conversation_id=str(args.conversation_id),
    )

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = Path.cwd() / f"stream_turn_report_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
