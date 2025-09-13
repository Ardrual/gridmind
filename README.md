Gridmind — FastAPI + React/Vite RAG Playground

- Backend: FastAPI service to query a small document set defined in `data/manifest.json`.
- Frontend: React/Vite UI in `web/` that calls the FastAPI `/query` endpoint.
- Ingest: a script that downloads PDFs listed in the manifest into `data/raw/` and can parse/vectorize them into a local ChromaDB store.

Overview
- Documents are declared in `data/manifest.json` with fields like `id`, `title`, `url`, `license`, `sha256`.
- The ingest step fetches each `url` and stores it locally as `data/raw/<id>.pdf`.
- API endpoints and retrieval logic (RAG) live under `app/`.

Repo Structure
- `app/` — FastAPI app (RAG endpoints)
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
   - Copy `web/.env.example` to `web/.env` and set `VITE_API_BASE_URL` to your FastAPI host (defaults to http://localhost:8000).
   - `npm run dev`
   - Styling is handled by Tailwind CSS utility classes.
   - Tailwind CSS v4 is config-less here: `web/src/index.css` contains `@import "tailwindcss";` and PostCSS runs with `@tailwindcss/postcss` + `autoprefixer` (see `web/postcss.config.js`). No `tailwind.config.js` is required unless you customize.

Make Targets
- `make dev` — runs API and web concurrently.
- `make web` — installs web deps if needed and starts the React/Vite dev server in `web/`.
- `make web-install` — installs web deps.
- `make web-build` — builds the web app for production.
- `make run-api` — runs `uvicorn app.main:app --reload`.

API Status
- The FastAPI app under `app/` is implemented. It exposes `POST /query` and `GET /healthz`.
- CORS is enabled for the Vite dev origin (`http://localhost:5173`) and optionally `FRONTEND_ORIGIN` from the environment.

RAG Query Demo
- This repo now includes a minimal LangChain RetrievalQA pipeline using Gemini for generation and a local Chroma DB for retrieval.
- Populate the vector store first, then run the demo script to ask a question (e.g., the 10 Standard Fire Orders):
  - `python -m scripts.ingest --vectorize`
  - `python -m scripts.query_demo --query "What are the 10 Standard Fire Orders?" --k 5`
- The demo prints the final answer, latency, and top-k citations (title, page, URL when available, plus a snippet).

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

Citations
- Each stored chunk includes metadata: `file_id`, `source` (PDF path), `page` (1‑based), and `chunk`.
- UIs can render citations like “Clayton CA Wildfire Tips (p. 3)” or “data/raw/clayton_evacuate_tips.pdf — p. 3”.
- For linkable citations, consider enriching metadata (future change) with `title` and `url` from the manifest and linking to `url#page=<page>`.

Manifest Schema
- Documents are declared in `data/manifest.json` with fields:
  ```json
  {
    "id": "clayton_evacuate_tips",
    "title": "Clayton CA Fire: Wildfire Preparation & Evacuation Tips",
    "url": "https://claytonca.gov/fc/police/Wildfire-Preparation-and-Evacuation-Tips.pdf",
    "license": "municipal release",
    "sha256": ""
  }
  ```
- The `id` becomes `file_id` in vector store metadata and the PDF filename (`data/raw/<id>.pdf`).

Vector Store Details
- Store: Chroma persistent DB at `data/chroma/`.
- Document IDs: `<file_id>-<page:04d>-<chunk:04d>` (e.g., `clayton_evacuate_tips-0003-0001`).
- Reset: delete `data/chroma/` to fully rebuild the vector store.

Gemini Setup
- Create an API key in Google AI Studio and place it in `.env` as `GOOGLE_API_KEY` (or `GEMINI_API_KEY`).
- The ingest script auto-loads `.env` and uses the Google Gen AI SDK (`google-genai`) to call the Gemini Embedding API.
- If no key is found, vectorization fails with a clear error. Ensure one of `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set. If both are set, `GOOGLE_API_KEY` is used.
- The embedding model defaults to `gemini-embedding-001`. If `GEMINI_EMBEDDING_MODEL` is unset or empty, the default is used.
- At query time, use the same embedding model for the user query to ensure vector-space compatibility; your generator LLM (Gemini 1.x) can be different from the embedding model.

Query-Time Config (env)
- `CHROMA_DB_DIR` (default: `data/chroma`) — path to Chroma persistence.
- `CHROMA_COLLECTION` (default: `docs`) — collection name.
- `GEMINI_EMBEDDING_MODEL` (default: `gemini-embedding-001`) — must match ingest.
- `GEMINI_EMBEDDING_DIM` (optional int) — set if you used `output_dimensionality` during ingest.
- `GEMINI_LLM_MODEL` (default: `gemini-1.5-flash`) — the generation model for answers.

Troubleshooting
- Missing API key: set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env` (or export it in your shell).
- “model is required”: ensure `GEMINI_EMBEDDING_MODEL` is not an empty string in `.env`, or pass `--embedding-model`.
- Network required: embeddings call the Google AI API. Ensure your environment has outbound network access.

Verify Vector Store
```python
import chromadb, os
client = chromadb.PersistentClient(path=os.path.join('data', 'chroma'))
col = client.get_or_create_collection('docs')
print('count:', col.count())
res = col.get(include=['metadatas','documents','embeddings'], limit=1)
print('sample_meta:', (res.get('metadatas') or [''])[0])
```

Sanity Check Commands
- Pytest (network-free): `python -m pytest -q tests/test_chroma_sanity.py`
  - Skips gracefully if `data/chroma/` or the `docs` collection is missing or empty.
- Manual check script: `python -m scripts.check_chroma`
  - Prints collection size and peeks a few items (ids, file_id, page, source).
  - Optional query (requires Gemini API key):
    - `python -m scripts.check_chroma --query "what is this corpus about?" --k 5`
  - Options: `--db-dir data/chroma`, `--collection docs`, `--embedding-model`, `--embedding-dim`.

RAG Internals
- Retrieval: a custom retriever queries Chroma using a Gemini query embedding (via the same embedding path as ingest) for consistency.
- Generation: a minimal Gemini LLM wrapper calls `google-genai` for text generation.
- Chain: LangChain `RetrievalQA` with `chain_type="stuff"` and `return_source_documents=True`.

Notes
- Ensure `data/manifest.json` entries include a stable `id` and valid PDF `url`.
- The API layer in `app/` is a placeholder; endpoints and retrieval logic will follow as the project evolves.
