"""Tests for document_loader module."""
from pathlib import Path
import pytest
from src.document_loader import Document, KnowledgeBaseLoader, MarkdownLoader, TextChunker


class TestMarkdownLoader:
    def test_strips_headings(self):
        raw = "# Main Heading\n## Sub\nBody text."
        cleaned = MarkdownLoader.clean(raw)
        assert "#" not in cleaned
        assert "Main Heading" in cleaned

    def test_strips_bold_italic(self):
        raw = "This is **bold** and *italic* text."
        cleaned = MarkdownLoader.clean(raw)
        assert "**" not in cleaned
        assert "bold" in cleaned

    def test_strips_links(self):
        raw = "See [our guide](https://example.com) for details."
        cleaned = MarkdownLoader.clean(raw)
        assert "https://example.com" not in cleaned
        assert "our guide" in cleaned

    def test_preserves_content(self):
        cleaned = MarkdownLoader.clean("Net Profit = Revenue minus Expenses")
        assert "Net Profit" in cleaned


class TestTextChunker:
    def test_short_text_single_chunk(self):
        chunker = TextChunker(chunk_size=1000, chunk_overlap=50)
        chunks = chunker.split("This is short. It fits.")
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = " ".join(f"Sentence number {i} here." for i in range(30))
        chunks = chunker.split(text)
        assert len(chunks) > 1

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_empty_text(self):
        assert TextChunker().split("   \n\n   ") == []


class TestDocument:
    def test_doc_id_generation(self):
        doc = Document(content="Test", source="balance_sheet", chunk_id=2)
        assert doc.doc_id == "balance_sheet::chunk_2"

    def test_roundtrip_serialisation(self):
        doc = Document(content="Test.", source="invoices", chunk_id=0)
        restored = Document.from_dict(doc.to_dict())
        assert restored.content == doc.content
        assert restored.source == doc.source
        assert restored.doc_id == doc.doc_id


class TestKnowledgeBaseLoader:
    def test_loads_real_knowledge_base(self):
        loader = KnowledgeBaseLoader()
        docs = loader.load()
        assert len(docs) > 0
        sources = {d.source for d in docs}
        expected = {"bookkeeping", "invoices", "profit_and_loss", "balance_sheet", "cash_flow"}
        assert expected.issubset(sources)

    def test_raises_on_missing_dir(self, tmp_path):
        loader = KnowledgeBaseLoader(kb_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_raises_on_empty_dir(self, tmp_path):
        loader = KnowledgeBaseLoader(kb_dir=tmp_path)
        with pytest.raises(ValueError, match="No documents loaded"):
            loader.load()

    def test_chunk_ids_sequential(self):
        loader = KnowledgeBaseLoader()
        docs = loader.load()
        by_source = {}
        for doc in docs:
            by_source.setdefault(doc.source, []).append(doc.chunk_id)
        for source, ids in by_source.items():
            assert ids == list(range(len(ids)))
