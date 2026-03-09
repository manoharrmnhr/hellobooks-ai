"""FAISS-backed vector store for semantic document retrieval."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import faiss
import numpy as np
from src.config import settings
from src.document_loader import Document

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages a FAISS IndexFlatIP index with JSON metadata.
    IndexFlatIP on L2-normalised vectors = cosine similarity.
    """
    def __init__(self, dimension=None, index_path=None, metadata_path=None):
        self.dimension = dimension or settings.embedding_dimension
        self.index_path = index_path or settings.faiss_index_path
        self.metadata_path = metadata_path or settings.metadata_path
        self._index = None
        self._documents = []

    def _build_new_index(self):
        return faiss.IndexFlatIP(self.dimension)

    @property
    def index(self):
        if self._index is None:
            self._index = self._build_new_index()
        return self._index

    @property
    def size(self):
        return self.index.ntotal

    def add(self, documents, embeddings):
        if len(documents) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(documents)} documents but {embeddings.shape[0]} embeddings")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != expected {self.dimension}")
        self.index.add(embeddings)
        self._documents.extend(documents)
        logger.info("Added %d chunks → total: %d", len(documents), self.size)

    def search(self, query_embedding, top_k=None):
        k = top_k or settings.top_k
        if self.size == 0:
            raise RuntimeError("Vector store is empty. Run `python main.py ingest` first.")
        k = min(k, self.size)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"document": self._documents[idx], "score": float(score)})
        return results

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        metadata = [doc.to_dict() for doc in self._documents]
        self.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved index (%d vectors) → %s", self.size, self.index_path)

    def load(self):
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        self._index = faiss.read_index(str(self.index_path))
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._documents = [Document.from_dict(m) for m in metadata]
        logger.info("Loaded index (%d vectors) from %s", self.size, self.index_path)
        return True

    def reset(self):
        self._index = self._build_new_index()
        self._documents = []
