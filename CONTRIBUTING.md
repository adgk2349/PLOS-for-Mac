# Contributing to PLOS for Mac

Thanks for contributing.

## Before You Start
- Read [README.md](README.md) for setup and architecture.
- Search existing issues/PRs to avoid duplicate work.
- Keep changes focused and small when possible.

## Development Setup
```bash
git clone https://github.com/adgk2349/PLOS-for-Mac.git
cd PLOS-for-Mac

cd sidecar
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[test]'
```

## Branching
Use feature branches and open PRs against `main`.

Suggested naming:
- `codex/<scope>-<short-description>`
- `feat/<scope>-<short-description>`
- `fix/<scope>-<short-description>`

## Code Style
### Swift (App)
- Keep UI logic in views; move orchestration to services/view models.
- Prefer small, composable view files over monolith views.
- Preserve public `AppViewModel` interfaces unless a breaking change is intentional.

### Python (Sidecar)
- Keep module boundaries clear (`pipeline`, `inference`, `memory`, `repositories`).
- Avoid prompt-leak/meta text in user-visible output.
- Preserve API schemas unless versioned changes are introduced.

## Testing
Run relevant tests before opening a PR.

### Sidecar full suite
```bash
cd sidecar
source .venv/bin/activate
pytest -q
```

### Core regression suites
```bash
pytest -q tests/test_v2_pipeline.py tests/test_local_inference_sanitize.py tests/test_memory_service_digest.py
```

### Swift tests
```bash
xcodebuild \
  -project PLOS.xcodeproj \
  -scheme PLOS \
  -destination 'platform=macOS' \
  test
```

## Pull Request Checklist
- [ ] Change is scoped and documented
- [ ] Tests added/updated for behavior changes
- [ ] Relevant tests pass locally
- [ ] README/docs updated if user-facing behavior changed
- [ ] No secrets, tokens, or private paths included

## Commit Messages
Use clear, imperative messages.

Examples:
- `feat(sidecar): apply direct-first policy for conversational recommendations`
- `fix(memory): prevent cross-session digest leakage`
- `docs(readme): add multilingual setup and performance guide`

## Security & Privacy
- Never commit API keys or local secrets.
- Treat memory/session data as sensitive by default.
- Preserve session isolation guarantees.

## Communication
When proposing architectural changes, include:
- problem statement
- constraints
- alternatives considered
- migration/rollback plan
