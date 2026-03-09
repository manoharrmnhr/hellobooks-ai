"""Integration tests for RAGEngine with mocked LLM."""
from __future__ import annotations
import numpy as np
import pytest
from src.document_loader import Document
from src.embeddings import EmbeddingEngine
from src.rag_engine import RAGEngine, QueryResponse
from src.vector_store import VectorStore

DIM = 384

class MockEmbeddingEngine(EmbeddingEngine):
    def __init__(self):
        self.model_name = "mock-model"
        self._model = None
        self._dim = DIM

    def embed(self, texts):
        rng = np.random.default_rng(sum(ord(c) for c in texts[0]))
        vecs = rng.random((len(texts), DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    @property
    def dimension(self): return self._dim

class MockLLMClient:
    model = "mock-claude"
    def generate_answer(self, question, context_chunks):
        return f"Mock answer for: {question}"

@pytest.fixture
def engine(tmp_path):
    embedder = MockEmbeddingEngine()
    store = VectorStore(dimension=DIM, index_path=tmp_path / "t.faiss", metadata_path=tmp_path / "m.json")
    return RAGEngine(embedding_engine=embedder, vector_store=store, llm_client=MockLLMClient())

@pytest.fixture
def ingested_engine(engine):
    engine.ingest()
    return engine

class TestRAGEngineIngest:
    def test_returns_positive_chunk_count(self, engine):
        assert engine.ingest() > 0

    def test_uses_cache_on_second_call(self, engine):
        c1 = engine.ingest()
        c2 = engine.ingest(force_rebuild=False)
        assert c1 == c2

    def test_force_rebuild(self, engine):
        engine.ingest()
        assert engine.ingest(force_rebuild=True) > 0

class TestRAGEngineQuery:
    def test_returns_response(self, ingested_engine):
        r = ingested_engine.query("What is a balance sheet?")
        assert isinstance(r, QueryResponse)
        assert len(r.answer) > 0

    def test_correct_chunk_count(self, ingested_engine):
        r = ingested_engine.query("What is bookkeeping?", top_k=3)
        assert len(r.retrieved_chunks) == 3

    def test_positive_latency(self, ingested_engine):
        assert ingested_engine.query("Cash flow").latency_ms > 0

    def test_sources_are_strings(self, ingested_engine):
        r = ingested_engine.query("Explain invoices")
        assert all(isinstance(s, str) for s in r.sources)

    def test_to_dict_keys(self, ingested_engine):
        d = ingested_engine.query("GST").to_dict()
        assert {"question","answer","sources","latency_ms","model_used","top_k","chunks"}.issubset(d)

    def test_query_before_ingest_raises(self, engine):
        with pytest.raises(RuntimeError, match="not initialised"):
            engine.query("test")

class TestRAGEngineHealth:
    def test_ok_after_ingest(self, ingested_engine):
        h = ingested_engine.health()
        assert h["status"] == "ok"
        assert h["index_size"] > 0

    def test_degraded_before_ingest(self, engine):
        assert engine.health()["status"] == "degraded"
