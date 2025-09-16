from __future__ import annotations

import os
from typing import Iterable, List, Optional


def embed_texts_gemini(
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

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Google API key. Set GOOGLE_API_KEY or GEMINI_API_KEY (e.g., in a .env file)."
        )

    client = genai.Client(api_key=api_key)

    def batched(seq: List[str], size: int) -> Iterable[List[str]]:
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    all_embeddings: List[List[float]] = []
    for chunk in batched(texts, batch_size):
        cfg_kwargs = {"task_type": task_type}
        if output_dimensionality is not None:
            cfg_kwargs["output_dimensionality"] = output_dimensionality
        cfg = types.EmbedContentConfig(**cfg_kwargs)  # type: ignore[arg-type]

        resp = client.models.embed_content(
            model=model,
            contents=chunk,
            config=cfg,
        )
        all_embeddings.extend([list(e.values) for e in resp.embeddings])

    return all_embeddings


# Minimal Embeddings wrapper so LangChain integrations (e.g., PGVector)
# can compute embeddings with the same Gemini pathway used in ingest/query.
try:
    from langchain_core.embeddings import Embeddings  # type: ignore
except Exception:  # pragma: no cover
    from langchain.embeddings.base import Embeddings  # type: ignore


class GeminiEmbeddings(Embeddings):
    model: str = "gemini-embedding-001"
    output_dimensionality: Optional[int] = None

    def __init__(
        self,
        *,
        model: str = "gemini-embedding-001",
        output_dimensionality: Optional[int] = None,
    ) -> None:
        self.model = model
        self.output_dimensionality = output_dimensionality

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts_gemini(
            texts,
            model=self.model,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self.output_dimensionality,
        )

    def embed_query(self, text: str) -> List[float]:
        [vec] = embed_texts_gemini(
            [text],
            model=self.model,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.output_dimensionality,
            batch_size=1,
        )
        return vec


__all__ = ["embed_texts_gemini", "GeminiEmbeddings"]

