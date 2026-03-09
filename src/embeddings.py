"""Embedding engine using Sentence-Transformers (runs locally)."""
from __future__ import annotations
import logging
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import settings

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    def __init__(self, model_name=None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts):
        if not texts:
            raise ValueError("Cannot embed empty list")
        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=len(texts) > 100,
                                        normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def embed_single(self, text):
        return self.embed([text])

    @property
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()

@lru_cache(maxsize=1)
def get_embedding_engine():
    return EmbeddingEngine()
