import os
import sys
import argparse
from typing import Optional


def main() -> None:
    # Load .env if present for local runs
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Simple RAG query demo using Gemini + Chroma")
    parser.add_argument("--query", default="What are the 10 Standard Fire Orders?", help="Query text")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents to retrieve")
    args = parser.parse_args()

    # Quick sanity on keys; retrieval also needs network for embeddings
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file.")
        sys.exit(1)

    from app.models import QueryRequest
    from app.rag import run_rag

    try:
        req = QueryRequest(query=args.query, k=args.k)
        ans = run_rag(req)
    except Exception as e:
        print(f"Query failed: {e}")
        sys.exit(2)

    print("=== Answer ===")
    print(ans.answer)
    print()
    print(f"Latency: {ans.latency_ms} ms\n")

    if ans.citations:
        print("=== Citations ===")
        for i, c in enumerate(ans.citations, start=1):
            title = c.title or c.source_id
            suffix = f" (p.{c.page})" if c.page else ""
            url = f"\n  URL: {c.url}" if c.url else ""
            print(f"{i}. {title}{suffix}{url}\n  Snippet: {c.snippet}\n")


if __name__ == "__main__":
    main()

