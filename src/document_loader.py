"""Document loader and text chunker for the Hellobooks knowledge base."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from src.config import settings


@dataclass(frozen=True)
class Document:
    """A single text chunk with associated metadata."""
    content: str
    source: str
    chunk_id: int
    doc_id: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "doc_id", f"{self.source}::chunk_{self.chunk_id}")

    def to_dict(self):
        return {"content": self.content, "source": self.source, "chunk_id": self.chunk_id, "doc_id": self.doc_id}

    @classmethod
    def from_dict(cls, data):
        return cls(content=data["content"], source=data["source"], chunk_id=data["chunk_id"])


class MarkdownLoader:
    _MD_TABLE_SEP = re.compile(r"^\|[-:| ]+\|$", re.MULTILINE)
    _MD_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    _MD_INLINE_CODE = re.compile(r"`[^`]+`")
    _MD_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
    _MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
    _MD_ITALIC = re.compile(r"\*(.+?)\*")
    _MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    _WHITESPACE = re.compile(r"\n{3,}")

    @classmethod
    def clean(cls, text):
        text = cls._MD_CODE_BLOCK.sub(lambda m: "\n" + m.group(0)[3:-3].strip() + "\n", text)
        text = cls._MD_TABLE_SEP.sub("", text)
        text = cls._MD_HEADING.sub("", text)
        text = cls._MD_BOLD.sub(r"\1", text)
        text = cls._MD_ITALIC.sub(r"\1", text)
        text = cls._MD_LINK.sub(r"\1", text)
        text = cls._MD_INLINE_CODE.sub(lambda m: m.group(0).strip("`"), text)
        text = cls._WHITESPACE.sub("\n\n", text)
        return text.strip()

    def load(self, path):
        return self.clean(path.read_text(encoding="utf-8"))


class TextChunker:
    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size=512, chunk_overlap=64):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text):
        sentences = self._SENTENCE_END.split(text)
        chunks = []
        current = []
        current_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            slen = len(sentence)
            if current_len + slen + 1 > self.chunk_size and current:
                chunks.append(" ".join(current))
                overlap_text = " ".join(current)[-self.chunk_overlap:]
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
            current.append(sentence)
            current_len += slen + 1
        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if c.strip()]


class KnowledgeBaseLoader:
    def __init__(self, kb_dir=None, chunk_size=None, chunk_overlap=None):
        self.kb_dir = kb_dir or settings.knowledge_base_dir
        self.loader = MarkdownLoader()
        self.chunker = TextChunker(chunk_size or settings.chunk_size, chunk_overlap or settings.chunk_overlap)

    def _iter_md_files(self):
        if not self.kb_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {self.kb_dir}")
        yield from sorted(self.kb_dir.glob("*.md"))

    def load(self):
        documents = []
        for md_file in self._iter_md_files():
            source = md_file.stem
            text = self.loader.load(md_file)
            chunks = self.chunker.split(text)
            for chunk_id, chunk in enumerate(chunks):
                documents.append(Document(content=chunk, source=source, chunk_id=chunk_id))
        if not documents:
            raise ValueError(f"No documents loaded from {self.kb_dir}.")
        return documents
