"""Tests for FastAPI endpoints."""
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

mock_engine = MagicMock()
mock_engine.ingest.return_value = 42
mock_engine.health.return_value = {
    "status": "ok", "index_size": 42, "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "claude-sonnet-4-20250514", "top_k": 3,
}
mock_engine.query.return_value = MagicMock(to_dict=lambda: {
    "question": "What is a balance sheet?",
    "answer": "A balance sheet shows assets, liabilities and equity.",
    "sources": ["Balance Sheet"],
    "latency_ms": 150.0,
    "model_used": "claude-sonnet-4-20250514",
    "top_k": 3,
    "chunks": [{"source": "Balance Sheet", "chunk_id": 0, "score": 0.95, "content_preview": "A balance sheet…"}],
})

@pytest.fixture
def client():
    with patch("src.api.get_engine", return_value=mock_engine), patch("src.api._engine", mock_engine):
        from src.api import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_has_status(self, client):
        assert "status" in client.get("/health").json()

class TestQueryEndpoint:
    def test_valid_question(self, client):
        r = client.post("/query", json={"question": "What is a balance sheet?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data and "sources" in data

    def test_too_short_rejected(self, client):
        assert client.post("/query", json={"question": "Hi"}).status_code == 422

    def test_missing_field(self, client):
        assert client.post("/query", json={}).status_code == 422

class TestIngestEndpoint:
    def test_returns_ok(self, client):
        r = client.post("/ingest")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

class TestRoot:
    def test_service_info(self, client):
        assert client.get("/").json()["service"] == "Hellobooks AI"
