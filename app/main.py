from fastapi import Depends, FastAPI, HTTPException

from app.deps import RagRunner, get_rag_runner
from app.models import Answer, QueryRequest


app = FastAPI(title="Gridmind API", version="0.1.0")


@app.post("/query", response_model=Answer)
def query_endpoint(req: QueryRequest, rag_runner: RagRunner = Depends(get_rag_runner)) -> Answer:
    """Run a RAG query over the local Chroma DB using Gemini.

    Expects a JSON body matching QueryRequest and returns an Answer.
    """
    try:
        return rag_runner(req)
    except (ValueError, FileNotFoundError) as e:
        # Common misconfiguration cases (missing API key, missing DB)
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Likely missing dependencies or collection issues
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # pragma: no cover - safety net
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/healthz")
def healthcheck() -> dict:
    return {"ok": True}
