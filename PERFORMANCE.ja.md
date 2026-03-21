# パフォーマンステストガイド (日本語)

ローカルモデルのレイテンシ、応答品質、RAG挙動を再現可能に検証するためのガイドです。

英語版: [PERFORMANCE.md](PERFORMANCE.md)

## 1) クイック回帰 (ポリシー + 品質)
```bash
cd sidecar
source .venv/bin/activate
pytest -q tests/test_v2_pipeline.py tests/test_local_inference_sanitize.py tests/test_memory_service_digest.py
```

## 2) API E2E 回帰
```bash
cd sidecar
source .venv/bin/activate
pytest -q tests/test_api_flow.py tests/test_memory_api.py tests/test_model_catalog.py
```

## 3) チャットレイテンシ ベンチマーク
```bash
cd sidecar
source .venv/bin/activate
python scripts/perf_chat_benchmark.py \
  --engine llama_cpp \
  --model-path "/absolute/path/to/model.gguf" \
  --turns 20
```

出力指標:
- `avg_ms`, `p50_ms`, `p95_ms`
- `success_rate`
- `result_type_counts`

## 4) RAG exact フィルタ検証
```bash
cd sidecar
source .venv/bin/activate
pytest -q tests/test_v2_pipeline.py -k "week_exact_filter or requested_weeks"
```

## 推奨合格基準
- ユーザー表示テキストにメタ/プロンプト漏洩なし
- 10ターン会話で反復ループなし
- 一般会話で後処理後の質問数 <= 1
- `N週` exact リクエストで他週の混入なし
