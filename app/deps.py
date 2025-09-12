from typing import Callable

from app.models import Answer, QueryRequest
from app.rag import run_rag


# Type alias for the callable that executes a RAG query
RagRunner = Callable[[QueryRequest], Answer]


def get_rag_runner() -> RagRunner:
    """Provide a RAG runner callable.

    Keeping this in deps.py allows easy swapping/mocking in tests
    and centralizes construction if we later introduce shared clients.
    """
    return run_rag
