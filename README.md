Gridmind — FastAPI + React/Vite RAG Playground

- Backend: FastAPI service (planned) to query a small document set defined in `data/manifest.json`.
- Frontend: simple React/Vite app in `web/` (for querying and UI experiments).
- Ingest: a minimal script that currently downloads PDFs listed in the manifest into `data/raw/`.

Overview
- Documents are declared in `data/manifest.json` with fields like `id`, `title`, `url`, `license`, `sha256`.
- The ingest step fetches each `url` and stores it locally as `data/raw/<id>.pdf`.
- API endpoints and retrieval logic (RAG) are scaffolded and will be added under `app/`.

Repo Structure
- `app/` — FastAPI app scaffolding (routes coming soon)
- `data/manifest.json` — list of source documents to ingest
- `data/raw/` — downloaded PDFs (created by the ingest script)
- `scripts/ingest.py` — ingest utilities (currently only download)
- `web/` — React/Vite frontend

Setup
1) Python environment
   - Create and activate a virtualenv, then install deps:
     - `python -m venv .venv && source .venv/bin/activate`
     - `pip install -r requirements.txt`
2) Frontend (optional, for local UI):
   - `cd web && npm install`
   - `npm run dev`

Ingest PDFs (current functionality)
- Command: `python -m scripts.ingest --download`
- Behavior:
  - Reads `data/manifest.json`.
  - Downloads each PDF `url` and saves it as `data/raw/<id>.pdf`.
- Flags: the script accepts flags, but right now only `--download` is implemented. Additional flags referenced elsewhere (e.g., in the `Makefile`) are placeholders and not yet wired up.

Notes
- Ensure `data/manifest.json` entries include a stable `id` and valid PDF `url`.
- The API layer in `app/` is a placeholder; endpoints and retrieval logic will follow as the project evolves.
