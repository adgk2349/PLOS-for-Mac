# 성능 테스트 가이드 (한국어)

로컬 모델 지연시간, 응답 품질, RAG 동작을 반복 가능하게 점검하는 가이드입니다.

영문 원본: [PERFORMANCE.md](PERFORMANCE.md)

## 1) 빠른 회귀 (정책 + 품질)
```bash
cd sidecar
source .venv/bin/activate
pytest -q tests/test_v2_pipeline.py tests/test_local_inference_sanitize.py tests/test_memory_service_digest.py
```

## 2) API 종단간 회귀
```bash
cd sidecar
source .venv/bin/activate
pytest -q tests/test_api_flow.py tests/test_memory_api.py tests/test_model_catalog.py
```

## 3) 채팅 지연시간 벤치마크
```bash
cd sidecar
source .venv/bin/activate
python scripts/perf_chat_benchmark.py \
  --engine llama_cpp \
  --model-path "/absolute/path/to/model.gguf" \
  --turns 20
```

출력 지표:
- `avg_ms`, `p50_ms`, `p95_ms`
- `success_rate`
- `result_type_counts`

## 4) RAG exact 필터 검증
```bash
cd sidecar
source .venv/bin/activate
pytest -q tests/test_v2_pipeline.py -k "week_exact_filter or requested_weeks"
```

## 권장 합격 기준
- 사용자 본문에 메타/프롬프트 누출 0회
- 10턴 대화에서 반복 루프 0회
- 일반대화 후처리 기준 질문 수 1개 이하
- `N주차` exact 요청 시 타 주차 혼입 없음
