
import os
import json
import argparse
from typing import Iterable, List, Tuple, Optional

# Load environment variables from a local .env file if present
try:  # Best-effort; do not hard-require in runtime environments without dotenv
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import requests

# Heavy deps are imported lazily inside functions to avoid import costs

def download_manifest_pdfs(manifest_path: str, raw_dir: str) -> None:
    """Download all PDFs listed in manifest into ``raw_dir``.

    Args:
        manifest_path: Path to ``manifest.json`` describing documents.
        raw_dir: Destination directory for downloaded PDFs.
    """
    os.makedirs(raw_dir, exist_ok=True)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    for entry in manifest:
        url = entry['url']
        file_id = entry['id']
        filename = f"{file_id}.pdf"
        out_path = os.path.join(raw_dir, filename)
        print(f"Downloading {url} -> {out_path}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(out_path, 'wb') as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f_out.write(chunk)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")


def _pdfs_in_dir(raw_dir: str) -> List[str]:
    """Return a sorted list of absolute paths to PDFs in ``raw_dir``."""
    if not os.path.isdir(raw_dir):
        return []
    pdfs = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(raw_dir, f))
    ]
    return sorted(pdfs)


def _extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """Extract page-wise text from a PDF using PyMuPDF.

    Returns a list of (page_number, text) tuples, with natural reading order
    sorting applied where possible.
    """
    import fitz  # PyMuPDF

    texts: List[Tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:  # type: ignore[attr-defined]
            # Use "text" with sort=True for a more natural reading order
            # per PyMuPDF docs.
            txt = page.get_text("text", sort=True)
            if txt:
                texts.append((page.number, txt))
    return texts


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple character-based chunker with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def _embed_texts_gemini(
    texts: List[str],
    *,
    model: str = "gemini-embedding-001",
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = None,
    batch_size: int = 100,
) -> List[List[float]]:
    """Embed a list of texts with the Gemini Embedding API using google-genai.

    Requires one of GOOGLE_API_KEY or GEMINI_API_KEY to be set in the env.
    """
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "google-genai is required for Gemini embeddings. Install with `pip install google-genai`."
        ) from e

    # Prefer explicit API key wiring so we can surface a clearer error.
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Google API key. Set GOOGLE_API_KEY or GEMINI_API_KEY (e.g., in a .env file)."
        )

    client = genai.Client(api_key=api_key)

    all_embeddings: List[List[float]] = []

    def batched(seq: List[str], size: int) -> Iterable[List[str]]:
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    for chunk in batched(texts, batch_size):
        cfg_kwargs = {"task_type": task_type}
        if output_dimensionality is not None:
            cfg_kwargs["output_dimensionality"] = output_dimensionality
        cfg = types.EmbedContentConfig(**cfg_kwargs)  # type: ignore[arg-type]

        # Batch embed with a single call; SDK accepts list for `contents`.
        resp = client.models.embed_content(
            model=model,
            contents=chunk,
            config=cfg,
        )
        # resp.embeddings is a list of ContentEmbedding with .values
        all_embeddings.extend([list(e.values) for e in resp.embeddings])

    return all_embeddings


def _vectorize_pdfs(
    raw_dir: str,
    db_dir: str,
    collection_name: str = "docs",
    chunk_size: int = 1000,
    overlap: int = 200,
    batch_size: int = 64,
    *,
    embedding_model: str = "gemini-embedding-001",
    output_dimensionality: Optional[int] = None,
) -> None:
    """Parse PDFs in ``raw_dir`` and store embeddings in ChromaDB at ``db_dir``.

    - Uses PyMuPDF to extract text (per-page, sorted for reading order).
    - Chunks text and upserts into a persistent ChromaDB collection.
    """
    import chromadb

    pdf_paths = _pdfs_in_dir(raw_dir)
    if not pdf_paths:
        print(f"No PDFs found in {raw_dir}; nothing to vectorize.")
        return

    os.makedirs(db_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=db_dir)
    # Important: we compute embeddings externally with Gemini for compatibility
    # with your query-time embedding model. Do not set an internal EF.
    collection = client.get_or_create_collection(name=collection_name)

    for pdf_path in pdf_paths:
        file_stem = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Vectorizing: {pdf_path}")

        # Build deterministic chunk IDs based on file stem and chunk index.
        page_texts = _extract_text_from_pdf(pdf_path)
        all_chunks: List[Tuple[str, dict]] = []  # (text, metadata)
        for page_no, text in page_texts:
            chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for idx, chunk in enumerate(chunks):
                meta = {
                    "source": os.path.relpath(pdf_path),
                    "page": page_no + 1,  # 1-based
                    "chunk": idx,
                    "file_id": file_stem,
                }
                all_chunks.append((chunk, meta))

        if not all_chunks:
            print(f"No text extracted from {pdf_path}; skipping.")
            continue

        # Upsert in batches to control memory and handle re-runs gracefully.
        def iter_batches(items: List[Tuple[str, dict]], size: int) -> Iterable[List[Tuple[str, dict]]]:
            for i in range(0, len(items), size):
                yield items[i:i + size]

        for batch_idx, batch in enumerate(iter_batches(all_chunks, batch_size)):
            ids = [f"{file_stem}-{m['page']:04d}-{m['chunk']:04d}" for _, m in batch]
            docs = [t for t, _ in batch]
            metas = [m for _, m in batch]
            embeds = _embed_texts_gemini(
                docs,
                model=embedding_model,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=output_dimensionality,
            )
            # Use upsert to avoid duplicate ID errors on repeated runs.
            collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
            print(
                f"  upserted batch {batch_idx + 1}/{(len(all_chunks) + batch_size - 1)//batch_size} ({len(ids)} items)"
            )

    print(f"Vectorization complete. Chroma persisted at: {os.path.abspath(db_dir)}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and process data files.")
    parser.add_argument('--download', action='store_true', help='Download PDFs from manifest into data/raw')
    parser.add_argument('--vectorize', action='store_true', help='Parse PDFs in data/raw and embed into local ChromaDB')
    parser.add_argument('--manifest', default=os.path.join('data', 'manifest.json'), help='Path to manifest.json')
    parser.add_argument('--raw-dir', default=os.path.join('data', 'raw'), help='Directory for raw PDFs')
    parser.add_argument('--db-dir', default=os.path.join('data', 'chroma'), help='Chroma persistence directory')
    # If GEMINI_EMBEDDING_MODEL is set but empty, fall back to default
    _env_model = os.getenv('GEMINI_EMBEDDING_MODEL')
    parser.add_argument(
        '--embedding-model',
        default=_env_model or 'gemini-embedding-001',
        help='Gemini embedding model id',
    )
    parser.add_argument('--embedding-dim', type=int, default=None, help='Optional embedding output dimensionality (e.g., 768, 1536, 3072)')
    args = parser.parse_args()

    if args.download and args.vectorize:
        # Run download first, then vectorize whatever is in raw-dir.
        download_manifest_pdfs(args.manifest, args.raw_dir)
        _vectorize_pdfs(
            args.raw_dir,
            args.db_dir,
            embedding_model=(args.embedding_model or 'gemini-embedding-001'),
            output_dimensionality=args.embedding_dim,
        )
        return

    if args.download:
        download_manifest_pdfs(args.manifest, args.raw_dir)
        return

    if args.vectorize:
        # Conditional download: only if raw-dir has no PDFs.
        if not _pdfs_in_dir(args.raw_dir):
            print(f"No PDFs found in {args.raw_dir}. Downloading from manifest...")
            download_manifest_pdfs(args.manifest, args.raw_dir)
        else:
            print(f"Found existing PDFs in {args.raw_dir}; skipping download.")
        _vectorize_pdfs(
            args.raw_dir,
            args.db_dir,
            embedding_model=(args.embedding_model or 'gemini-embedding-001'),
            output_dimensionality=args.embedding_dim,
        )
        return

    parser.print_help()

if __name__ == "__main__":
	main()
