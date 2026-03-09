"""Hellobooks AI — FastAPI REST API."""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.config import settings
from src.rag_engine import RAGEngine, QueryResponse

logging.basicConfig(level=getattr(logging, settings.log_level),
                    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

_engine: RAGEngine | None = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Hellobooks AI starting up…")
    engine = get_engine()
    chunks = engine.ingest()
    logger.info("Index ready with %d chunks", chunks)
    yield
    logger.info("Hellobooks AI shutting down.")

app = FastAPI(title="Hellobooks AI", description="RAG-powered accounting assistant.",
              version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["What is a balance sheet?"])
    top_k: int = Field(default=3, ge=1, le=10)

class ChunkPreview(BaseModel):
    source: str
    chunk_id: int
    score: float
    content_preview: str

class QueryResponseModel(BaseModel):
    question: str
    answer: str
    sources: list[str]
    latency_ms: float
    model_used: str
    top_k: int
    chunks: list[ChunkPreview]

class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str

class HealthResponse(BaseModel):
    status: str
    index_size: int | None = None
    embedding_model: str | None = None
    llm_model: str | None = None
    top_k: int | None = None
    error: str | None = None

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return get_engine().health()

@app.post("/query", response_model=QueryResponseModel, tags=["Q&A"])
async def query(request: QueryRequest):
    try:
        response: QueryResponse = get_engine().query(question=request.question, top_k=request.top_k)
        data = response.to_dict()
        return QueryResponseModel(**data)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during query")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest(force_rebuild: Annotated[bool, Query()] = False):
    try:
        chunks = get_engine().ingest(force_rebuild=force_rebuild)
        return IngestResponse(status="ok", chunks_indexed=chunks, message=f"Index built with {chunks} chunks.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/", tags=["System"])
async def root():
    return {"service": "Hellobooks AI", "version": "1.0.0", "docs": "/docs", "health": "/health"}
