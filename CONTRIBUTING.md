# Contributing

Thank you for your interest in contributing to StockScreenClaude.

## Getting Started

See the [Development Guide](docs/DEVELOPMENT.md) for full from-source setup instructions covering backend, frontend, Redis, and Celery workers.

## Code Structure

- **Backend:** [backend/README.md](backend/README.md) — Architecture, API reference, database schema, service layer
- **Frontend:** [frontend/README.md](frontend/README.md) — Component structure, tech stack, patterns, conventions
- **Architecture overview:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System design, data flow, caching strategy

## Quality Gates

The project runs 5 automated quality gates plus a theme identity gate via CI. Run locally before pushing:

```bash
make gates         # Run all 5 SE quality gates
make all           # Full CI (backend gates + frontend lint/test)
make gate-check    # Verify all test files are assigned to a gate
```

See `make help` for all available targets. Details in [docs/release-checklist.md](docs/release-checklist.md).

## Git Conventions

This project uses **Conventional Commits**:

```
<type>[optional scope]: <description>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

**Scopes:** `api`, `scanner`, `chatbot`, `frontend`, `celery`, `db`, `cache`, `themes`, `signals`

Examples:
```
feat(scanner): add volume breakthrough screener
fix(chatbot): handle empty response from LLM provider
refactor(api): consolidate stock data fetching logic
```

## Branching Model

The repo uses a two-branch flow:

- **`develop`** — long-lived integration branch. Feature branches branch off `develop`, get merged back into `develop` via PR after CI passes.
- **`main`** — production source of truth. Only `develop` (or hotfix branches) get merged into `main`. Production-grade workflows fire off `main` only: GHCR `latest` images, semver `v*` release tags, the static site (GitHub Pages), and the weekly reference data refresh.

PRs into either `develop` or `main` run the full CI quality gate suite. Image publishing is gated to direct pushes on `main` and `v*` tags only — it never fires from develop or from PRs.

GitHub branch protection should require PRs (no direct pushes) and passing CI on both `main` and `develop`.

## Pull Requests

1. Create a feature branch from `develop`
2. Make your changes with clear, conventional commit messages
3. Ensure all quality gates pass locally (`make all`)
4. Open a PR against `develop` with a description of what changed and why
5. CI runs automatically on all PRs
6. Once accumulated work in `develop` is validated, open a `develop` → `main` PR to ship a release-ready slice

## Testing

- **Backend tests:** `pytest` (see [Development Guide](docs/DEVELOPMENT.md#running-tests))
- **Frontend tests:** `npm run test:run` in `frontend/`
- **Integration tests:** require a running server (`pytest tests/integration/ -m integration`)

## Environment Variables

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the full configuration reference.
