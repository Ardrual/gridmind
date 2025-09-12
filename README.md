Gridmind — FastAPI + React/Vite RAG Playground

- Backend: FastAPI service (planned) to query a small document set defined in `data/manifest.json`.
- Frontend: simple React/Vite app in `web/` (for querying and UI experiments).
- Ingest: a script that downloads PDFs listed in the manifest into `data/raw/` and can parse/vectorize them into a local ChromaDB store.

Overview
- Documents are declared in `data/manifest.json` with fields like `id`, `title`, `url`, `license`, `sha256`.
- The ingest step fetches each `url` and stores it locally as `data/raw/<id>.pdf`.
- API endpoints and retrieval logic (RAG) are scaffolded and will be added under `app/`.

Repo Structure
- `app/` — FastAPI app scaffolding (routes coming soon)
- `data/manifest.json` — list of source documents to ingest
- `data/raw/` — downloaded PDFs (created by the ingest script)
- `scripts/ingest.py` — ingest utilities (download + parse/vectorize)
- `web/` — React/Vite frontend

Setup
1) Python environment
   - Create and activate a virtualenv, then install deps:
     - `python -m venv .venv && source .venv/bin/activate`
     - `pip install -r requirements.txt`
2) Frontend (optional, for local UI):
   - `cd web && npm install`
   - `npm run dev`

Ingest & Vectorize
- Download only: `python -m scripts.ingest --download`
  - Reads `data/manifest.json` and saves PDFs to `data/raw/<id>.pdf`.
- Vectorize: `python -m scripts.ingest --vectorize`
  - If `data/raw/` has no PDFs, downloads from `data/manifest.json` first; otherwise skips download.
  - Parses PDFs with PyMuPDF, chunks text, and embeds into a persistent ChromaDB store at `data/chroma/` using Gemini embeddings (default: `gemini-embedding-001`).
- Common flags:
  - `--manifest data/manifest.json` (default)
  - `--raw-dir data/raw` (default)
  - `--db-dir data/chroma` (default)
  - `--embedding-model gemini-embedding-001` (default or env `GEMINI_EMBEDDING_MODEL`)
  - `--embedding-dim 3072` (optional output dimensionality)

Gemini Setup
- Create an API key in Google AI Studio and place it in `.env` as `GOOGLE_API_KEY` (or `GEMINI_API_KEY`).
- The ingest script auto-loads `.env` and uses the Google Gen AI SDK (`google-genai`) to call the Gemini Embedding API.
- If no key is found, vectorization fails with a clear error. Ensure one of `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set.
- At query time, use the same embedding model for the user query to ensure vector-space compatibility; your generator LLM (Gemini 1.x) can be different from the embedding model.

Notes
- Ensure `data/manifest.json` entries include a stable `id` and valid PDF `url`.
- The API layer in `app/` is a placeholder; endpoints and retrieval logic will follow as the project evolves.
