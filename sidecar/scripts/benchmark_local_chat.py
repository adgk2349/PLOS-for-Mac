#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


PROMPTS = [
    "Swift에서 배열 합계를 구하는 함수 예시 2개만 보여줘.",
    "아이폰 17 최신 정보 알려줘.",
    "방금 답변 핵심만 3줄로 다시 정리해줘.",
    "Given nums=[2,7,11,15], target=9 two-sum 파이썬 정답 코드만 줘.",
]


@dataclass
class RunResult:
    ok: bool
    latency_ms: float
    output_len: int
    text: str
    error: str = ""


def post_json(url: str, payload: dict, timeout: float, headers: dict[str, str] | None = None) -> dict:
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=merged_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def run_sidecar(base_url: str, prompt: str, timeout: float, conversation_id: str, session_token: str) -> RunResult:
    url = base_url.rstrip("/") + "/v2/chat/local"
    payload = {
        "query": prompt,
        "mode": "GENERAL",
        "conversation_id": conversation_id,
        "session_id": conversation_id,
    }
    started = time.perf_counter()
    try:
        data = post_json(url, payload, timeout, headers={"x-session-token": session_token})
        elapsed = (time.perf_counter() - started) * 1000.0
        text = str(data.get("result_summary") or data.get("lead") or data.get("generated_text") or "").strip()
        if not text:
            structured = data.get("structured_result") or {}
            text = str(structured.get("summary") or "").strip()
        return RunResult(ok=bool(text), latency_ms=elapsed, output_len=len(text), text=text, error="" if text else "empty_response")
    except urllib.error.URLError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return RunResult(ok=False, latency_ms=elapsed, output_len=0, text="", error=f"url_error:{exc}")
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000.0
        return RunResult(ok=False, latency_ms=elapsed, output_len=0, text="", error=f"exception:{exc}")


def run_ollama(base_url: str, model: str, prompt: str, timeout: float) -> RunResult:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    started = time.perf_counter()
    try:
        data = post_json(url, payload, timeout)
        elapsed = (time.perf_counter() - started) * 1000.0
        text = str(data.get("response") or "").strip()
        return RunResult(ok=bool(text), latency_ms=elapsed, output_len=len(text), text=text, error="" if text else "empty_response")
    except urllib.error.URLError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return RunResult(ok=False, latency_ms=elapsed, output_len=0, text="", error=f"url_error:{exc}")
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000.0
        return RunResult(ok=False, latency_ms=elapsed, output_len=0, text="", error=f"exception:{exc}")


def run_lmstudio(base_url: str, model: str, prompt: str, timeout: float) -> RunResult:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    started = time.perf_counter()
    try:
        data = post_json(url, payload, timeout)
        elapsed = (time.perf_counter() - started) * 1000.0
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = str(((choices[0].get("message") or {}).get("content")) or "").strip()
        return RunResult(ok=bool(text), latency_ms=elapsed, output_len=len(text), text=text, error="" if text else "empty_response")
    except urllib.error.URLError as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return RunResult(ok=False, latency_ms=elapsed, output_len=0, text="", error=f"url_error:{exc}")
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000.0
        return RunResult(ok=False, latency_ms=elapsed, output_len=0, text="", error=f"exception:{exc}")


def summarize(name: str, rows: list[RunResult]) -> dict:
    oks = [r for r in rows if r.ok]
    latencies = [r.latency_ms for r in oks]
    outlens = [r.output_len for r in oks]
    return {
        "backend": name,
        "total": len(rows),
        "ok": len(oks),
        "fail": len(rows) - len(oks),
        "p50_ms": round(statistics.median(latencies), 2) if latencies else None,
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "avg_output_len": round(statistics.mean(outlens), 2) if outlens else None,
        "errors": [r.error for r in rows if not r.ok][:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8787")
    parser.add_argument("--conversation-id", default="benchmark-room")
    parser.add_argument("--sidecar-token", default="dev-session-token")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--lmstudio-url", default="http://127.0.0.1:1234")
    parser.add_argument("--lmstudio-model", default="local-model")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--no-lmstudio", action="store_true")
    args = parser.parse_args()

    result_rows: dict[str, list[RunResult]] = {"sidecar": []}
    if not args.no_ollama:
        result_rows["ollama"] = []
    if not args.no_lmstudio:
        result_rows["lmstudio"] = []

    for prompt in PROMPTS:
        result_rows["sidecar"].append(
            run_sidecar(
                base_url=args.sidecar_url,
                prompt=prompt,
                timeout=args.timeout,
                conversation_id=args.conversation_id,
                session_token=args.sidecar_token,
            )
        )
        if "ollama" in result_rows:
            result_rows["ollama"].append(
                run_ollama(
                    base_url=args.ollama_url,
                    model=args.ollama_model,
                    prompt=prompt,
                    timeout=args.timeout,
                )
            )
        if "lmstudio" in result_rows:
            result_rows["lmstudio"].append(
                run_lmstudio(
                    base_url=args.lmstudio_url,
                    model=args.lmstudio_model,
                    prompt=prompt,
                    timeout=args.timeout,
                )
            )

    report = {
        "prompts": PROMPTS,
        "summary": [summarize(name, rows) for name, rows in result_rows.items()],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
