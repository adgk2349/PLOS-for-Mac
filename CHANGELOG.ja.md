# 変更履歴 (日本語)

## v0.2.1 (2026-03-19)

### 主な更新
- `/v2/chat/local` を中心とした会話パイプラインを強化
  - フォローアップ解釈、候補優先応答、確認質問の過多抑制を改善
  - 応答契約を `lead + result + actions + metadata` に統一
- ローカルメモリ層（Session / Workspace / Preference / Episodic / Pinned）を統合
  - 質問・アクション・ファイル選択時の書き込みフックを追加
  - 各段階で関連メモリのみを選択注入
- ローカル/外部ルーティングを安定化
  - Privacy gate（`LOCAL_ONLY`, `HYBRID`, `CONFIRM`）の分岐を強化
  - 一般会話はローカル LLM 優先、失敗時の処理を改善

### UI/UX
- macOS メインワークスペースのレイアウトを整理
  - ヘッダー/サイドバー/チャット間の境界線・重なり問題を緩和
  - カプセル/ラウンド部品の整列一貫性を改善
- 設定/状態/オンボーディング画面の導線と見た目を調整

### インデックス/検索
- 文書メタデータ（カテゴリ/タグ/サブカテゴリ/重要度）処理を強化
- ワークスペース境界に基づく検索フィルタを改善
- PDF/OCR の安定性と失敗記録経路を改善

### 安定性/開発
- sidecar 起動とポート競合時の対処を強化
- セッショントークン/認証処理を補強
- テストとモジュール分離（pipeline/memory/verification）を拡張

---

## 参照
- メイン文書: [README.ja.md](README.ja.md)
- 한국어: [CHANGELOG.ko.md](CHANGELOG.ko.md)
- English: [CHANGELOG.en.md](CHANGELOG.en.md)
