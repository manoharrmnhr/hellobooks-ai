"""Tests for vector_store module."""
import numpy as np
import pytest
from src.document_loader import Document
from src.vector_store import VectorStore

DIM = 8

def make_docs(n):
    return [Document(content=f"Document {i}.", source="test", chunk_id=i) for i in range(n)]

def make_embeddings(n, dim=DIM):
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

@pytest.fixture
def store(tmp_path):
    return VectorStore(dimension=DIM, index_path=tmp_path / "t.faiss", metadata_path=tmp_path / "m.json")

class TestVectorStore:
    def test_initial_size_zero(self, store):
        assert store.size == 0

    def test_add_and_search(self, store):
        docs = make_docs(5)
        embs = make_embeddings(5)
        store.add(docs, embs)
        assert store.size == 5
        results = store.search(embs[[0]], top_k=3)
        assert len(results) == 3
        assert results[0]["score"] > 0.99

    def test_dimension_mismatch_raises(self, store):
        with pytest.raises(ValueError, match="Embedding dimension"):
            store.add(make_docs(2), np.zeros((2, DIM+1), dtype=np.float32))

    def test_doc_count_mismatch_raises(self, store):
        with pytest.raises(ValueError, match="Mismatch"):
            store.add(make_docs(3), make_embeddings(2))

    def test_search_empty_raises(self, store):
        with pytest.raises(RuntimeError, match="empty"):
            store.search(make_embeddings(1))

    def test_persist_and_reload(self, store, tmp_path):
        docs = make_docs(4)
        embs = make_embeddings(4)
        store.add(docs, embs)
        store.save()
        new_store = VectorStore(dimension=DIM, index_path=tmp_path / "t.faiss", metadata_path=tmp_path / "m.json")
        assert new_store.load() is True
        assert new_store.size == 4
        assert new_store.search(embs[[1]], top_k=1)[0]["document"].chunk_id == 1

    def test_load_false_if_no_files(self, store):
        assert store.load() is False

    def test_reset_clears(self, store):
        store.add(make_docs(3), make_embeddings(3))
        store.reset()
        assert store.size == 0
