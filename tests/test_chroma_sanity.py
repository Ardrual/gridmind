import os
from typing import List

import pytest


def test_chroma_persistence_sanity() -> None:
    """
    Sanity-check the persisted Chroma DB without requiring network access.

    - Verifies the `docs` collection exists under `data/chroma`.
    - Ensures the collection has at least one record.
    - Fetches a few items and checks expected metadata fields.
    """
    try:
        import chromadb  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"chromadb not installed: {e}")

    db_dir = os.path.join("data", "chroma")
    collection_name = "docs"

    if not os.path.isdir(db_dir):
        pytest.skip("Chroma directory not found; run vectorization first.")

    client = chromadb.PersistentClient(path=db_dir)

    # Ensure the collection exists (avoid implicitly creating it on get-or-create).
    names: List[str] = [c.name for c in client.list_collections()]
    if collection_name not in names:
        pytest.skip(
            f"Collection '{collection_name}' not found. Vectorize PDFs with `python -m scripts.ingest --vectorize`."
        )

    collection = client.get_collection(name=collection_name)

    count = collection.count()
    if count == 0:
        pytest.skip("Collection is empty; vectorize PDFs first.")

    # Pull a small sample and verify structure
    # Note: "ids" is always returned by get; do not request it in include.
    res = collection.get(limit=5, include=["metadatas", "documents"]) 

    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    assert isinstance(ids, list) and len(ids) > 0, "Expected at least one id from Chroma"
    assert isinstance(docs, list) and len(docs) == len(ids), "documents length should match ids"
    assert isinstance(metas, list) and len(metas) == len(ids), "metadatas length should match ids"

    # Check expected metadata keys on at least one item
    sample_meta = metas[0]
    for key in ("file_id", "page", "chunk"):
        assert key in sample_meta, f"missing metadata key: {key}"
