# Changelog (English)

## v0.2.1 (2026-03-19)

### Highlights
- Strengthened conversational pipeline centered on `/v2/chat/local`
  - Better follow-up resolution, candidate-first behavior, and reduced repetitive clarification loops
  - Unified response contract: `lead + result + actions + metadata`
- Integrated local memory layers (Session / Workspace / Preference / Episodic / Pinned)
  - Added write hooks for queries/actions/file selections
  - Injects only relevant memory per stage (planner/retriever/composer)
- Improved local/external routing behavior
  - Hardened privacy gates (`LOCAL_ONLY`, `HYBRID`, `CONFIRM`)
  - Prioritizes local LLM path for general conversation with improved fallback handling

### UI/UX
- Refined main workspace layout on macOS
  - Reduced seam/overlap issues between header, sidebar, and chat regions
  - Improved consistency of capsule/rounded controls
- Updated settings/status/onboarding visual flow and interaction consistency

### Indexing & Retrieval
- Enhanced document metadata flow (category/tags/subcategory/importance)
- Strengthened workspace-bound retrieval filtering
- Improved PDF/OCR reliability and failure tracking paths

### Stability & Dev
- Better sidecar bootstrap behavior and busy-port handling
- Improved session token/auth handling
- Expanded tests and modularization (pipeline/memory/verification components)

---

## References
- Main docs: [README.en.md](README.en.md)
- Korean: [CHANGELOG.ko.md](CHANGELOG.ko.md)
- Japanese: [CHANGELOG.ja.md](CHANGELOG.ja.md)
