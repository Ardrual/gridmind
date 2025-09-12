# Repository Guidelines

## Project Structure & Module Organization
- `app/` — FastAPI backend (RAG endpoints WIP).
- `scripts/` — data ingest utilities (see `ingest.py`).
- `data/manifest.json` — source document list; downloads stored in `data/raw/`.
- `web/` — React + Vite frontend.
- `tests/` — Python tests and fixtures.

## Build, Test, and Development Commands
- Backend (dev): `uvicorn app.main:app --reload` or `make run-api` (API is scaffolded).
- Ingest PDFs: `python -m scripts.ingest --download` (reads `data/manifest.json`, writes `data/raw/`).
- Frontend (dev): `cd web && npm install && npm run dev` or `make web`.
- Full dev loop: `make dev` (runs API + web concurrently).
- Note: `make data/ingest` entries reference future flags; prefer the explicit ingest command above for now.

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indent, type hints required for new/changed code.
  - Files/modules `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- TypeScript/React: follow ESLint rules in `web/`.
  - Components `PascalCase` (e.g., `SearchPanel.tsx`), hooks `useX`, props/interfaces typed.

## Testing Guidelines
- Framework: `pytest` (install for dev: `pip install pytest`).
- Test files: `tests/test_*.py`; keep unit tests fast and deterministic (no network).
- Run: `python -m pytest -q` from repo root.

## Commit & Pull Request Guidelines
- Before each push: redocument changes in `README.md` (human-facing, descriptive) and `AGENTS.md` (agent notes, minimal). Reflect new commands, flags, routes, or layout.
- Commits: concise, imperative mood; group related changes.
  - Example: `ingest: add --download to fetch manifest PDFs`.
- PRs: clear description (what/why), steps to validate, screenshots for UI, and linked issues.
- Ensure README/AGENTS are consistent with any CLI, API, or directory changes.

## Security & Configuration Tips
- Never commit secrets; use `.env` (local) and keep `.env.example` updated.
- Respect document licenses in `manifest.json`; avoid adding proprietary PDFs to the repo—reference by URL.
