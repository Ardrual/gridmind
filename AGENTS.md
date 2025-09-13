# Repository Guidelines

## Project Structure & Module Organization
- `app/` — FastAPI backend (RAG endpoints WIP).
- `scripts/` — data ingest utilities (see `ingest.py`).
- `data/manifest.json` — source document list; downloads stored in `data/raw/`.
- `web/` — React + Vite frontend.
- `tests/` — Python tests and fixtures.

## Build, Test, and Development Commands
- Backend (dev): `uvicorn app.main:app --reload` or `make run-api`.
- Ingest PDFs: `python -m scripts.ingest --download` (reads `data/manifest.json`, writes `data/raw/`).
- Vectorize PDFs: `python -m scripts.ingest --vectorize` (conditionally downloads if `data/raw/` is empty; then parses with PyMuPDF and embeds into Chroma at `data/chroma/` using Gemini embeddings).
- RAG demo query: `python -m scripts.query_demo --query "What are the 10 Standard Fire Orders?" --k 5` (requires API key and a populated Chroma DB).
- Chroma sanity test (no network): `python -m pytest -q tests/test_chroma_sanity.py`
- Manual DB check: `python -m scripts.check_chroma` (optional `--query` requires API key)
- Frontend (dev): `cd web && npm install && npm run dev` or `make web`.
- Configure `web/.env` with `VITE_API_BASE_URL` for the FastAPI host (default `http://localhost:8000`).
 - Styling: Tailwind CSS v4 (config-less). `web/src/index.css` uses `@import "tailwindcss";` and PostCSS plugin `@tailwindcss/postcss` expands styles.
- Full dev loop: `make dev` (runs API + web concurrently).
- Note: prefer explicit `python -m scripts.ingest` commands as shown.

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
 - Ingest auto-loads `.env` at runtime; ensure `GOOGLE_API_KEY` or `GEMINI_API_KEY` is present for Gemini embeddings.
 - If both `GOOGLE_API_KEY` and `GEMINI_API_KEY` are set, `GOOGLE_API_KEY` is used. If `GEMINI_EMBEDDING_MODEL` is empty or unset, default `gemini-embedding-001` is used.
  - Query-time env (used by `app/rag.py` and `scripts/query_demo.py`):
    - `CHROMA_DB_DIR` (default `data/chroma`), `CHROMA_COLLECTION` (default `docs`).
    - `GEMINI_EMBEDDING_MODEL` (default `gemini-embedding-001`), `GEMINI_EMBEDDING_DIM` (optional int to match ingest dimensionality).
    - `GEMINI_LLM_MODEL` (default `gemini-1.5-flash`).

## Citation Metadata
- Retain `page` and `file_id` in metadata for human-readable citations.
- Consider enriching with `title` and `url` from `manifest.json` for linkable citations (e.g., `url#page=<page>`), when modifying ingest.
