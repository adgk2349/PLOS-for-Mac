# PLOS for Mac (English)

PLOS is a local-first AI workspace core for macOS.  
Indexing and default responses run locally, and external AI is used only when needed.

## Key Features
- Local indexing (txt/md/pdf + OCR fallback)
- Local RAG chat with citations
- Work modes (general/summary/research/development/writing/planning/strict_search)
- Conversational response layer (v2)
- Memory layers (Session/Workspace/Preference/Episodic/Pinned)
- Policy-gated external AI calls (privacy gate)

## UI/UX Direction
- Apple glassEffect-based UI
- Unified capsule/rounded components
- Neutral, readable tones in light/dark modes
- Chat-first layout with action chips

## Project Structure
- `PLOS/`: macOS SwiftUI app
- `sidecar/local_ai_core/`: Python FastAPI sidecar
- `sidecar/tests/`: sidecar tests

## Run
1. Open `PLOS.xcodeproj` in Xcode
2. Run app (sidecar bootstraps automatically)

## Docs
- Changelog: [CHANGELOG.en.md](CHANGELOG.en.md)
- 한국어: [README.ko.md](README.ko.md)
- 日本語: [README.ja.md](README.ja.md)
