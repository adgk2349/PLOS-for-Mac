# CI/CD Layout

This repository uses a mixed layout:

- Runtime GitHub Actions wrappers: `.github/workflows/*`
- Reusable CI/CD scripts and hook templates: `cicd/*`

Current guard entrypoint:

- `cicd/scripts/repo_guard.py`

Local hook installation:

```bash
bash cicd/scripts/install_hooks.sh
```

