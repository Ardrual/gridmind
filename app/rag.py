"""
RAG pipeline using LangChain RetrievalQA + Gemini over a local Chroma DB.

This module exposes a single entry point:

    run_rag(req: QueryRequest) -> Answer

It wires a minimal LangChain-compatible Gemini LLM wrapper and a custom
retriever that queries a persisted Chroma collection using Gemini embeddings
for the query vector. Citations are enriched with `title` and `url` from
`data/manifest.json` when available.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import PrivateAttr

from app.models import Answer, Citation, QueryRequest


# Lazy imports of heavy dependencies inside functions/classes where possible


def _load_manifest_index(path: str) -> Dict[str, Dict[str, str]]:
    """Load manifest and return a map: file_id -> {title, url}.

    If the manifest file is missing or invalid, returns an empty mapping.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        index: Dict[str, Dict[str, str]] = {}
        for entry in data:
            fid = entry.get("id")
            if not fid:
                continue
            index[fid] = {
                "title": entry.get("title") or "",
                "url": entry.get("url") or "",
            }
        return index
    except Exception:
        return {}


# --- Gemini LLM wrapper ----------------------------------------------------

try:
    # langchain >= 0.1 ships core interfaces in langchain_core
    from langchain_core.language_models.llms import LLM
    from langchain_core.callbacks import CallbackManagerForLLMRun
except Exception:  # pragma: no cover
    # Fallback for older langchain versions, though our requirements pin modern
    from langchain.llms.base import LLM  # type: ignore
    from langchain.callbacks.manager import (
        CallbackManagerForLLMRun,  # type: ignore
    )


class GeminiLLM(LLM):
    """Minimal LangChain LLM wrapper around google-genai text generation.

    Expects one of GOOGLE_API_KEY or GEMINI_API_KEY in the environment.
    """

    model: str = "gemini-1.5-flash"

    @property
    def _llm_type(self) -> str:  # pragma: no cover - metadata
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            from google import genai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "google-genai is required for Gemini LLM. Install with `pip install google-genai`."
            ) from e

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing Google API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment."
            )

        client = genai.Client(api_key=api_key)

        # Basic single-turn generation; LangChain handles stop tokens if needed.
        resp = client.models.generate_content(model=self.model, contents=prompt)
        text = getattr(resp, "text", None)
        return text or ""


# --- Chroma retriever that uses Gemini for query embeddings ----------------

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore
    from langchain.retrievers.base import BaseRetriever  # type: ignore


class ChromaGeminiRetriever(BaseRetriever):
    """Retriever that queries a persisted Chroma collection.

    - Computes the query embedding using the same Gemini embedding path as ingest
      to ensure vector-space compatibility.
    - Requires the Chroma DB directory to exist with a collection name.
    """

    # Pydantic model fields
    db_dir: str
    collection_name: str = "docs"
    k: int = 5
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensionality: Optional[int] = None

    # Private attrs for runtime handles
    _client: Any = PrivateAttr(default=None)
    _collection: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        try:
            import chromadb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "chromadb is required for retrieval. Install with `pip install chromadb`."
            ) from e

        if not os.path.isdir(self.db_dir):
            raise FileNotFoundError(
                f"Chroma directory not found: {self.db_dir}. Vectorize PDFs first."
            )
        self._client = chromadb.PersistentClient(path=self.db_dir)

        names = {c.name for c in self._client.list_collections()}
        if self.collection_name not in names:
            raise RuntimeError(
                f"Collection '{self.collection_name}' not found in {self.db_dir}. Vectorize PDFs first."
            )
        self._collection = self._client.get_collection(name=self.collection_name)

    def _embed_query(self, text: str) -> List[float]:
        # Reuse the existing ingest helper for consistent embeddings.
        from scripts.ingest import _embed_texts_gemini

        [vec] = _embed_texts_gemini(
            [text],
            model=self.embedding_model,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.embedding_dimensionality,
            batch_size=1,
        )
        return vec

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> List[Document]:
        q_vec = self._embed_query(query)

        res = self._collection.query(
            query_embeddings=[q_vec],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        out: List[Document] = []
        for _id, content, meta in zip(ids, docs, metas):
            md = dict(meta or {})
            md["id"] = _id
            out.append(Document(page_content=content or "", metadata=md))
        return out


# --- Public API ------------------------------------------------------------


def run_rag(req: QueryRequest) -> Answer:
    """Run RetrievalQA using the configured vector store backend.

    - Supports `VECTOR_BACKEND=chroma` (default) or `pgvector`.
    - Uses a minimal Gemini LLM wrapper for generation.
    - Returns the answer text, top-k citations, and overall latency in ms.
    """
    # Best-effort .env for local dev
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    # Config from env with sensible defaults
    vector_backend = (os.getenv("VECTOR_BACKEND") or "chroma").lower()
    db_dir = os.getenv("CHROMA_DB_DIR", os.path.join("data", "chroma"))
    collection = os.getenv("CHROMA_COLLECTION") or os.getenv("PGVECTOR_COLLECTION") or "docs"
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001"
    # Optional override if ingest used output_dimensionality
    emb_dim_env = os.getenv("GEMINI_EMBEDDING_DIM")
    embedding_dim = int(emb_dim_env) if emb_dim_env else None
    llm_model = os.getenv("GEMINI_LLM_MODEL") or "gemini-1.5-flash"

    # Build retriever based on backend
    if vector_backend == "pgvector":
        try:
            from langchain_postgres import PGVector  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "langchain-postgres is required for VECTOR_BACKEND=pgvector. Install with `pip install langchain-postgres psycopg[binary]`."
            ) from e

        from app.embeddings import GeminiEmbeddings

        pg_uri = os.getenv("PGVECTOR_URL") or os.getenv("DATABASE_URL")
        if not pg_uri:
            raise ValueError("PGVECTOR_URL (or DATABASE_URL) must be set for pgvector backend")

        embeddings = GeminiEmbeddings(
            model=embedding_model,
            output_dimensionality=embedding_dim,
        )

        vs = None
        # 1) Try connection as a string via `connection`
        try:
            vs = PGVector(
                embeddings=embeddings,
                collection_name=collection,
                connection=pg_uri,
                use_jsonb=True,
            )
        except TypeError:
            # 2) Try `connection_string`
            try:
                vs = PGVector(
                    embeddings=embeddings,
                    collection_name=collection,
                    connection_string=pg_uri,
                    use_jsonb=True,
                )
            except TypeError:
                # 3) Try helper dataclasses
                try:
                    from langchain_postgres.vectorstores import ConnectionArgs  # type: ignore
                    conn = ConnectionArgs.from_uri(pg_uri)
                except Exception:
                    from langchain_postgres.vectorstores import Connection  # type: ignore
                    conn = Connection.from_uri(pg_uri)  # type: ignore[attr-defined]

                vs = PGVector(
                    embeddings=embeddings,
                    collection_name=collection,
                    connection=conn,
                    use_jsonb=True,
                )

        retriever = vs.as_retriever(search_kwargs={"k": req.k})
    else:
        retriever = ChromaGeminiRetriever(
            db_dir=db_dir,
            collection_name=collection,
            k=req.k,
            embedding_model=embedding_model,
            embedding_dimensionality=embedding_dim,
        )
    llm = GeminiLLM(model=llm_model)

    # Compose RetrievalQA chain
    from langchain.chains import RetrievalQA

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # Run chain and time it
    t0 = time.perf_counter()
    result = chain.invoke({"query": req.query})
    dt_ms = int((time.perf_counter() - t0) * 1000)

    answer_text: str = result.get("result") or ""
    source_docs: List[Any] = result.get("source_documents") or []

    # Build citations enriched with manifest metadata when possible
    manifest_idx = _load_manifest_index(os.path.join("data", "manifest.json"))
    citations: List[Citation] = []
    for doc in source_docs:
        # Works with both langchain_core.documents.Document and duck-typed objects
        meta: Dict[str, Any] = getattr(doc, "metadata", {}) or {}
        content: str = getattr(doc, "page_content", "") or ""

        file_id = str(meta.get("file_id") or meta.get("id") or "")
        page = int(meta.get("page") or 0)
        cid = str(meta.get("id") or meta.get("source_id") or "")

        manifest_meta = manifest_idx.get(file_id, {"title": "", "url": ""})

        snippet = content.strip()
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "..."

        citations.append(
            Citation(
                source_id=cid or file_id,
                title=manifest_meta.get("title", ""),
                url=manifest_meta.get("url", ""),
                page=page,
                snippet=snippet,
            )
        )

    return Answer(answer=answer_text, citations=citations[: req.k], latency_ms=dt_ms)


__all__ = ["run_rag", "GeminiLLM", "ChromaGeminiRetriever"]
