from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class Citation(BaseModel):
    source_id: str
    title: str
    url: str
    page: int
    snippet: str

class Answer(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: int
