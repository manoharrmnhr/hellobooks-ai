"""RAG Engine — central orchestrator for Hellobooks AI."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from src.config import settings
from src.document_loader import Document, KnowledgeBaseLoader
from src.embeddings import EmbeddingEngine, get_embedding_engine
from src.llm_client import LLMClient
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    document: Document
    score: float

    @property
    def source_label(self):
        return self.document.source.replace("_", " ").title()

@dataclass
class QueryResponse:
    question: str
    answer: str
    retrieved_chunks: list
    latency_ms: float
    model_used: str
    top_k: int

    @property
    def sources(self):
        seen = set()
        result = []
        for chunk in self.retrieved_chunks:
            label = chunk.source_label
            if label not in seen:
                seen.add(label)
                result.append(label)
        return result

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "latency_ms": round(self.latency_ms, 1),
            "model_used": self.model_used,
            "top_k": self.top_k,
            "chunks": [
                {"source": c.source_label, "chunk_id": c.document.chunk_id,
                 "score": round(c.score, 4), "content_preview": c.document.content[:120] + "…"}
                for c in self.retrieved_chunks
            ],
        }

class RAGEngine:
    def __init__(self, embedding_engine=None, vector_store=None, llm_client=None, top_k=None):
        self.embedder = embedding_engine or get_embedding_engine()
        self.vector_store = vector_store or VectorStore(dimension=self.embedder.dimension)
        self._llm_client = llm_client
        self.top_k = top_k or settings.top_k
        self._ready = False

    @property
    def llm(self):
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def ingest(self, force_rebuild=False):
        if not force_rebuild and self.vector_store.load():
            logger.info("Loaded existing index with %d chunks", self.vector_store.size)
            self._ready = True
            return self.vector_store.size
        logger.info("Building index from knowledge base...")
        loader = KnowledgeBaseLoader()
        documents = loader.load()
        logger.info("Loaded %d chunks from %d docs", len(documents), len({d.source for d in documents}))
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.embed(texts)
        if embeddings.shape[1] != self.vector_store.dimension:
            self.vector_store.dimension = embeddings.shape[1]
        self.vector_store.reset()
        self.vector_store.add(documents, embeddings)
        self.vector_store.save()
        self._ready = True
        return self.vector_store.size

    def _ensure_ready(self):
        if not self._ready:
            loaded = self.vector_store.load()
            if loaded:
                self._ready = True
            else:
                raise RuntimeError("RAG engine not initialised. Call engine.ingest() first.")

    def query(self, question, top_k=None):
        self._ensure_ready()
        k = top_k or self.top_k
        t0 = time.perf_counter()
        query_vec = self.embedder.embed_single(question.strip())
        raw_results = self.vector_store.search(query_vec, top_k=k)
        retrieved = [RetrievedChunk(document=r["document"], score=r["score"]) for r in raw_results]
        logger.info("Retrieved %d chunks for query: '%.60s'", len(retrieved), question)
        answer = self.llm.generate_answer(question, raw_results)
        latency_ms = (time.perf_counter() - t0) * 1000
        return QueryResponse(question=question, answer=answer, retrieved_chunks=retrieved,
                             latency_ms=latency_ms, model_used=self.llm.model, top_k=k)

    def health(self):
        try:
            self._ensure_ready()
            return {"status": "ok", "index_size": self.vector_store.size,
                    "embedding_model": self.embedder.model_name, "llm_model": self.llm.model, "top_k": self.top_k}
        except RuntimeError as e:
            return {"status": "degraded", "error": str(e)}
