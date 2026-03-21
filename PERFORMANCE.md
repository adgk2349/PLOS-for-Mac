# Performance Testing Guide

This guide provides repeatable checks for local model latency, response quality, and RAG behavior.

## Scope
- General chat latency (`/v2/chat/local`)
- Recommendation/direct-first behavior
- RAG routing and retrieval correctness
- Summary quality for local/hybrid profiles

## Prerequisites
- Sidecar environment prepared (`pip install -e .`)
- A local model path configured (MLX or GGUF)
- Optional: indexed workspace for RAG checks

## 1) Fast Regression (Policy + Quality)
```bash
cd sidecar
source .venv/bin/activate
pytest -q tests/test_v2_pipeline.py tests/test_local_inference_sanitize.py tests/test_memory_service_digest.py
```

## 2) End-to-End API Regression
```bash
cd sidecar
source .venv/bin/activate
pytest -q tests/test_api_flow.py tests/test_memory_api.py tests/test_model_catalog.py
```

## 3) Local Chat Latency Benchmark (Script)
```bash
cd sidecar
source .venv/bin/activate
python scripts/perf_chat_benchmark.py \
  --engine llama_cpp \
  --model-path "/absolute/path/to/model.gguf" \
  --turns 20
```

Example output fields:
- `avg_ms`, `p50_ms`, `p95_ms`
- `success_rate`
- `result_type_counts`

## 4) RAG Exact-Filter Validation
Run week exact/filter tests:
```bash
cd sidecar
source .venv/bin/activate
pytest -q tests/test_v2_pipeline.py -k "week_exact_filter or requested_weeks"
```

## 5) Manual Comparison Matrix (Recommended)
Collect and compare by model profile:
- Engine: `mlx` / `llama_cpp`
- Model: 8B / 12B / 14B / 20B+
- Query sets:
  - daily conversation
  - coding Q&A
  - explicit file requests (`find/summarize/open`)
- Record:
  - latency (avg/p95)
  - question-loop incidence
  - leak/repetition incidence
  - retrieval precision (when RAG is requested)

## Suggested Acceptance Gates
- No meta/prompt leak in user-visible text
- No repetitive loop in 10-turn simulation
- General conversation keeps question count <= 1 after postprocess
- `N주차` exact requests do not mix other weeks
