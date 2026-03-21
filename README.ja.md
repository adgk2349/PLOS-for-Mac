# PLOS for Mac (日本語)

macOS向けローカルファーストAIワークスペース。

[English](README.md) | [한국어](README.ko.md) | 日本語

## 概要
PLOS は SwiftUI デスクトップアプリと Python FastAPI サイドカーを組み合わせ、ローカル中心の対話・検索・要約ワークフローを提供します。

- ローカル優先チャット + RAG（出典表示）
- ポリシー制御の外部プロバイダ呼び出し（任意）
- メモリ階層（Session / Workspace / Preference / Pinned）
- ハードウェアに応じたモデルカタログ
- 一般会話 Direct-First 応答ポリシー

## リポジトリ構成
- `PLOS/`: SwiftUI macOS アプリ
- `sidecar/local_ai_core/`: FastAPI サイドカー
- `sidecar/tests/`: サイドカーテスト
- `PLOSTests/`, `PLOSUITests/`: Swift テストターゲット

## 要件
- Apple Silicon Mac 推奨（Mシリーズ）
- macOS 14+
- Xcode 15+
- Python 3.11+
- OCR（任意）: `tesseract`, `poppler`

## インストール
### 1) クローン
```bash
git clone https://github.com/adgk2349/PLOS-for-Mac.git
cd PLOS-for-Mac
```

### 2) サイドカー環境準備
```bash
cd sidecar
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[test]'
```

### 3) OCR ツール（任意）
```bash
brew install tesseract poppler
```

### 4) アプリ実行
- Xcode で `PLOS.xcodeproj` を開く
- `PLOS` ターゲットを実行
- アプリ起動/終了に合わせてサイドカーも自動連動

## サイドカー単体起動（開発用）
```bash
cd sidecar
source .venv/bin/activate
export LOCAL_AI_SESSION_TOKEN=dev-token
export LOCAL_AI_DATA_DIR="$(pwd)/data"
uvicorn local_ai_core.main:create_app --factory --host 127.0.0.1 --port 8787
```

## モデル推奨レンジ
現行カタログの実用目安:
- 16GB: 7B/8B中心、12B〜14B上限トライ可
- 64GB+: 20B/70Bクラス
- 256GB+: GPT-OSS 120B
- 500GB+: Kimi 2.5 / Qwen 3.5 397Bクラス

## メモリ構造
- Session memory: チャットルームごとに分離
- Workspace memory: プロジェクト単位の文脈
- Preference/Pinned memory: ユーザーが明示保存した情報

## テスト
### サイドカー
```bash
cd sidecar
source .venv/bin/activate
pytest -q
```

### 重点回帰セット
```bash
pytest -q tests/test_v2_pipeline.py tests/test_local_inference_sanitize.py tests/test_memory_service_digest.py
```

### Swift テスト
```bash
xcodebuild \
  -project PLOS.xcodeproj \
  -scheme PLOS \
  -destination 'platform=macOS' \
  test
```

## パフォーマンステスト
再現可能な測定手順は [PERFORMANCE.ja.md](PERFORMANCE.ja.md) を参照。

## コントリビュート
[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md) または [CONTRIBUTING.md](CONTRIBUTING.md) を参照。

## 変更履歴
- [CHANGELOG.ja.md](CHANGELOG.ja.md)
- [CHANGELOG.en.md](CHANGELOG.en.md)
- [CHANGELOG.ko.md](CHANGELOG.ko.md)

## ライセンス
MIT License（[LICENSE](LICENSE)）。
