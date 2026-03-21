# PLOS for Mac コントリビュートガイド (日本語)

コントリビュートありがとうございます。

英語の正式版は [CONTRIBUTING.md](CONTRIBUTING.md) です。

## 開始前
- [README.ja.md](README.ja.md) / [README.md](README.md) を確認
- 重複防止のため既存 Issue/PR を確認
- 変更はできるだけ小さく分割

## 開発環境
```bash
git clone https://github.com/adgk2349/PLOS-for-Mac.git
cd PLOS-for-Mac

cd sidecar
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[test]'
```

## ブランチ/PR
- 機能ブランチで作業し、`main` 向けに PR を作成
- 推奨命名: `codex/<scope>-<summary>`, `feat/...`, `fix/...`

## テスト
```bash
cd sidecar
source .venv/bin/activate
pytest -q
pytest -q tests/test_v2_pipeline.py tests/test_local_inference_sanitize.py tests/test_memory_service_digest.py
```

Swift:
```bash
xcodebuild -project PLOS.xcodeproj -scheme PLOS -destination 'platform=macOS' test
```

## PR チェックリスト
- [ ] 変更内容を文書化
- [ ] 動作変更に対応するテストを追加/更新
- [ ] 関連テストがローカルで成功
- [ ] ユーザー影響がある場合 README/ドキュメント更新
- [ ] 秘密情報（APIキー等）を含めない

## セキュリティ/プライバシー
- APIキー・シークレットは絶対にコミットしない
- セッションメモリ分離を維持
- 機微情報は最小限のみ保持
