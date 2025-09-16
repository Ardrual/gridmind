import argparse
import os
from typing import Optional
import sqlalchemy as sa


def main() -> None:
    # Best-effort .env loading for local runs
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    try:
        from langchain_postgres import PGVector  # type: ignore
    except Exception as e:
        raise SystemExit(
            f"langchain-postgres is required for this script: {e}. Install with `pip install langchain-postgres psycopg[binary]`"
        )

    # Lazy import to avoid heavy deps if not querying
    def _embed_query(text: str, model: str, output_dimensionality: Optional[int]) -> list[float]:
        from app.embeddings import embed_texts_gemini

        [emb] = embed_texts_gemini(
            [text],
            model=model,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=output_dimensionality,
            batch_size=1,
        )
        return emb

    parser = argparse.ArgumentParser(description="Check pgvector collection and optionally run a query.")
    parser.add_argument("--pg-uri", default=os.getenv("PGVECTOR_URL") or os.getenv("DATABASE_URL") or "", help="Postgres URI (postgresql+psycopg://user:pass@host:port/db)")
    parser.add_argument("--collection", default=os.getenv("PGVECTOR_COLLECTION") or os.getenv("CHROMA_COLLECTION") or "docs", help="Collection name")
    parser.add_argument("--peek", type=int, default=3, help="How many items to peek for structural checks")
    parser.add_argument("--query", default=None, help="Optional natural language query to run")
    parser.add_argument("--embedding-model", default=os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Optional embedding output dimensionality")
    parser.add_argument("--k", type=int, default=5, help="Top-k results to return for the query")
    args = parser.parse_args()

    if not args.pg_uri:
        raise SystemExit("--pg-uri or PGVECTOR_URL is required")

    # We instantiate without embeddings for structural operations; add embeddings only if querying
    store = None
    try:
        store = PGVector(
            embeddings=None,  # type: ignore[arg-type]
            collection_name=args.collection,
            connection=args.pg_uri,
            use_jsonb=True,
        )
    except TypeError:
        try:
            store = PGVector(
                embeddings=None,  # type: ignore[arg-type]
                collection_name=args.collection,
                connection_string=args.pg_uri,
                use_jsonb=True,
            )
        except TypeError:
            try:
                from langchain_postgres.vectorstores import ConnectionArgs  # type: ignore
                conn = ConnectionArgs.from_uri(args.pg_uri)
            except Exception:
                from langchain_postgres.vectorstores import Connection  # type: ignore
                conn = Connection.from_uri(args.pg_uri)  # type: ignore[attr-defined]

            store = PGVector(
                embeddings=None,  # type: ignore[arg-type]
                collection_name=args.collection,
                connection=conn,
                use_jsonb=True,
            )

    # Use the public session helpers to inspect the collection
    with store._make_sync_session() as session:  # type: ignore[attr-defined]
        collection = store.get_collection(session)
        if not collection:
            raise SystemExit(f"Collection not found: {args.collection}")

        # Count total embeddings in this collection
        count_stmt = (
            sa.select(sa.func.count())
            .select_from(store.EmbeddingStore)
            .where(store.EmbeddingStore.collection_id == collection.uuid)
        )
        total = session.execute(count_stmt).scalar() or 0
        if total == 0:
            raise SystemExit("Collection is empty; vectorize PDFs first.")

        print(f"pgvector OK: '{args.collection}' has {total} items in {args.pg_uri}")

        # Peek a few rows directly from the embeddings table
        peek_stmt = (
            sa.select(store.EmbeddingStore.id, store.EmbeddingStore.cmetadata)
            .where(store.EmbeddingStore.collection_id == collection.uuid)
            .limit(args.peek)
        )
        results = session.execute(peek_stmt).all()

        print(f"Peek {len(results)} items:")
        for i, (id_, meta) in enumerate(results, start=1):
            meta = meta or {}
            src = meta.get("source")
            page = meta.get("page")
            file_id = meta.get("file_id")
            print(f"  {i}. id={id_} file_id={file_id} page={page} source={src}")

    if args.query:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "Missing GOOGLE_API_KEY/GEMINI_API_KEY for query embedding. Set it or omit --query."
            )
        q_emb = _embed_query(args.query, args.embedding_model, args.embedding_dim)
        # Newer interfaces expose similarity_search_by_vector; fall back to search if absent
        try:
            docs = store.similarity_search_by_vector(q_emb, k=args.k)  # type: ignore[attr-defined]
        except Exception:
            # fallback to constructing a retriever with embeddings
            from app.embeddings import GeminiEmbeddings
            store = PGVector(
                embeddings=GeminiEmbeddings(
                    model=args.embedding_model,
                    output_dimensionality=args.embedding_dim,
                ),
                collection_name=args.collection,
                connection=args.pg_uri,
                use_jsonb=True,
            )
            docs = store.similarity_search(args.query, k=args.k)

        print(f"\nTop-{args.k} results for query: {args.query!r}")
        for rank, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            title = meta.get("file_id")
            page = meta.get("page")
            print(f"  {rank}. id={meta.get('id','')} file_id={title} page={page}")


if __name__ == "__main__":
    main()
