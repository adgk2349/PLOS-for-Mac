# PLOS for Mac (日本語)

PLOS は macOS 向けのローカル優先 AI ワークスペースコアです。  
インデックスと通常応答はローカルで実行し、必要時のみ外部 AI を利用します。

## 主な機能
- ローカルインデックス (txt/md/pdf + OCR fallback)
- ローカル RAG チャット + 出典表示
- 作業モード (general/summary/research/development/writing/planning/strict_search)
- 会話型レスポンスレイヤー (v2)
- メモリレイヤー (Session/Workspace/Preference/Episodic/Pinned)
- ポリシー制御の外部 AI 呼び出し (privacy gate)

## UI/UX 方針
- Apple の glassEffect ベース UI
- カプセル/ラウンド要素の統一
- ライト/ダークで可読性を維持する中立トーン
- チャット中心レイアウト + アクションチップ

## 構成
- `PLOS/`: macOS SwiftUI アプリ
- `sidecar/local_ai_core/`: Python FastAPI sidecar
- `sidecar/tests/`: sidecar テスト

## 実行
1. Xcode で `PLOS.xcodeproj` を開く
2. アプリ実行 (sidecar は自動起動)

## ドキュメント
- 変更履歴: [CHANGELOG.ja.md](CHANGELOG.ja.md)
- 한국어: [README.ko.md](README.ko.md)
- English: [README.en.md](README.en.md)
