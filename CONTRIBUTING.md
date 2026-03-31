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

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with clear, conventional commit messages
3. Ensure all quality gates pass locally (`make all`)
4. Open a PR against `main` with a description of what changed and why
5. CI runs automatically on all PRs

## Testing

- **Backend tests:** `pytest` (see [Development Guide](docs/DEVELOPMENT.md#running-tests))
- **Frontend tests:** `npm run test:run` in `frontend/`
- **Integration tests:** require a running server (`pytest tests/integration/ -m integration`)

## Environment Variables

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the full configuration reference.
