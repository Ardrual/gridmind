import argparse
import os
from typing import Optional


def _embed_query(text: str, model: str, output_dimensionality: Optional[int]) -> list[float]:
    """Embed a single query string using the same Gemini path as ingest.

    Requires GOOGLE_API_KEY or GEMINI_API_KEY in the environment.
    """
    from scripts.ingest import _embed_texts_gemini  # reuse existing helper

    [embedding] = _embed_texts_gemini(
        [text],
        model=model,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=output_dimensionality,
        batch_size=1,
    )
    return embedding


def main() -> None:
    # Best-effort .env loading for local runs
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    try:
        import chromadb  # type: ignore
    except Exception as e:
        raise SystemExit(f"chromadb is required for this script: {e}")

    parser = argparse.ArgumentParser(description="Check persisted Chroma DB and optionally run a query.")
    parser.add_argument("--db-dir", default=os.path.join("data", "chroma"), help="Chroma persistence directory")
    parser.add_argument("--collection", default="docs", help="Collection name to use")
    parser.add_argument("--peek", type=int, default=3, help="How many items to peek for structural checks")
    parser.add_argument("--query", default=None, help="Optional natural language query to run")
    parser.add_argument("--embedding-model", default=os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Optional embedding output dimensionality")
    parser.add_argument("--k", type=int, default=5, help="Top-k results to return for the query")
    args = parser.parse_args()

    if not os.path.isdir(args.db_dir):
        raise SystemExit(f"Chroma directory not found: {args.db_dir}. Vectorize first.")

    client = chromadb.PersistentClient(path=args.db_dir)

    collections = {c.name for c in client.list_collections()}
    if args.collection not in collections:
        raise SystemExit(
            f"Collection '{args.collection}' not found in {args.db_dir}. Vectorize PDFs first."
        )

    col = client.get_collection(name=args.collection)
    total = col.count()
    if total == 0:
        raise SystemExit("Collection is empty; vectorize PDFs first.")

    print(f"Chroma OK: {args.collection} has {total} items at {os.path.abspath(args.db_dir)}")

    # Structural peek
    res = col.get(limit=args.peek, include=["ids", "metadatas", "documents"])
    ids = res.get("ids", [])
    metas = res.get("metadatas", [])
    docs = res.get("documents", [])
    print(f"Peek {len(ids)} items:")
    for i, (id_, meta) in enumerate(zip(ids, metas)):
        src = meta.get("source")
        page = meta.get("page")
        file_id = meta.get("file_id")
        print(f"  {i+1}. id={id_} file_id={file_id} page={page} source={src}")

    # Optional semantic query (requires Gemini API key)
    if args.query:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "Missing GOOGLE_API_KEY/GEMINI_API_KEY for query embedding. Set it or omit --query."
            )

        q_emb = _embed_query(args.query, args.embedding_model, args.embedding_dim)
        qres = col.query(
            query_embeddings=[q_emb],
            n_results=args.k,
            include=["documents", "metadatas", "distances", "ids"],
        )

        print(f"\nTop-{args.k} results for query: {args.query!r}")
        for rank, (id_, meta, dist) in enumerate(
            zip(qres.get("ids", [[]])[0], qres.get("metadatas", [[]])[0], qres.get("distances", [[]])[0]),
            start=1,
        ):
            title = meta.get("file_id")
            page = meta.get("page")
            print(f"  {rank}. id={id_} file_id={title} page={page} distance={dist:.4f}")


if __name__ == "__main__":
    main()

