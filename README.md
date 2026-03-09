# 🏦 Hellobooks AI

> **RAG-powered accounting assistant** — ask natural language questions about bookkeeping, invoices, P&L, balance sheets, cash flow, and more.

Built for the **Hellobooks** AI-powered bookkeeping platform.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│         RAG Engine              │
│                                 │
│  1. EmbeddingEngine             │
│     sentence-transformers       │
│     (all-MiniLM-L6-v2, local)  │
│              │                  │
│              ▼                  │
│  2. VectorStore (FAISS)         │
│     IndexFlatIP (cosine sim)    │
│     Top-K retrieval             │
│              │                  │
│              ▼                  │
│  3. LLMClient (Anthropic)       │
│     Claude Sonnet               │
│     Grounded answer generation  │
└─────────────────────────────────┘
      │
      ▼
  Structured Answer + Sources
```

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | `sentence-transformers` (local) | Free, fast, no API quota |
| Vector Store | FAISS `IndexFlatIP` | Exact cosine search, zero infra |
| LLM | Anthropic Claude Sonnet | Best-in-class reasoning |
| API | FastAPI + uvicorn | Async, auto-docs, production-ready |
| CLI | Typer + Rich | Beautiful terminal UX |

---

## Features

- 9 rich accounting knowledge base documents (Markdown)
- Semantic embedding with `sentence-transformers` (fully local)
- FAISS vector index with cosine similarity search
- Claude-powered grounded answer generation
- FastAPI REST API with Swagger UI at `/docs`
- Interactive CLI with REPL chat mode
- Persistent FAISS index (survives restarts)
- Docker + Docker Compose for one-command deployment
- Full test suite (unit + integration, no API key needed for tests)
- Pydantic-validated configuration from environment variables

---

## Prerequisites

- Python **3.11+**
- An **Anthropic API key** — [get one here](https://console.anthropic.com/)
- Docker (optional, for containerised deployment)

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/your-username/hellobooks-ai.git
cd hellobooks-ai
```

### 2. Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Build the index

```bash
python main.py ingest
```

### 6. Ask your first question

```bash
python main.py query "What is a balance sheet?"
```

---

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Anthropic model |
| `LLM_MAX_TOKENS` | `1024` | Max response tokens |
| `LLM_TEMPERATURE` | `0.2` | Temperature (lower = more focused) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `TOP_K` | `3` | Chunks retrieved per query |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `API_HOST` | `0.0.0.0` | Server host |
| `API_PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Usage

### CLI

```bash
# Build / rebuild the FAISS index
python main.py ingest

# Force full rebuild
python main.py ingest --force

# Ask a question
python main.py query "What is the difference between bookkeeping and accounting?"

# Show retrieved chunks
python main.py query "How do I calculate gross profit margin?" --verbose

# Interactive REPL chat
python main.py chat

# Start REST API
python main.py serve

# Custom host/port
python main.py serve --host 127.0.0.1 --port 9000
```

#### Chat REPL commands

| Command | Action |
|---------|--------|
| `/verbose` | Toggle display of retrieved chunks |
| `/help` | Show commands |
| `exit` / `quit` | Exit |

---

### REST API

```bash
python main.py serve
# Docs at: http://localhost:8000/docs
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health and index status |
| `POST` | `/query` | Ask a question |
| `POST` | `/ingest` | Rebuild index |
| `GET` | `/docs` | Swagger UI |

#### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is accounts receivable?", "top_k": 3}'
```

**Response:**
```json
{
  "question": "What is accounts receivable?",
  "answer": "Accounts Receivable (AR) represents money owed to a business...",
  "sources": ["Accounts Receivable"],
  "latency_ms": 1240.5,
  "model_used": "claude-sonnet-4-20250514",
  "top_k": 3,
  "chunks": [
    {
      "source": "Accounts Receivable",
      "chunk_id": 0,
      "score": 0.9213,
      "content_preview": "Accounts Receivable (AR) represents money owed..."
    }
  ]
}
```

---

## Docker

### Docker Compose (recommended)

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# Build and start
docker compose up --build

# Background
docker compose up -d --build

# Logs
docker compose logs -f

# Stop
docker compose down
```

### Docker directly

```bash
# Build
docker build -t hellobooks-ai .

# Run API server
docker run -d \
  --name hellobooks-ai \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-your-key-here \
  -v hellobooks_index:/app/vector_store \
  -v hellobooks_models:/app/model_cache \
  hellobooks-ai

# Run CLI inside container
docker run --rm -it \
  -e ANTHROPIC_API_KEY=sk-ant-your-key-here \
  hellobooks-ai python main.py chat
```

The first startup downloads the embedding model (~90MB) and builds the FAISS index. Subsequent startups load from the volume cache.

---

## Project Structure

```
hellobooks-ai/
├── knowledge_base/             # Accounting knowledge docs (Markdown)
│   ├── bookkeeping.md
│   ├── invoices.md
│   ├── profit_and_loss.md
│   ├── balance_sheet.md
│   ├── cash_flow.md
│   ├── accounts_payable.md
│   ├── accounts_receivable.md
│   ├── tax_basics.md
│   └── financial_ratios.md
│
├── src/                        # Application source code
│   ├── __init__.py
│   ├── config.py               # Pydantic settings (env vars)
│   ├── document_loader.py      # Markdown loader + text chunker
│   ├── embeddings.py           # Sentence-Transformers wrapper
│   ├── vector_store.py         # FAISS index management
│   ├── llm_client.py           # Anthropic Claude client
│   ├── rag_engine.py           # Core RAG pipeline orchestrator
│   └── api.py                  # FastAPI REST application
│
├── tests/                      # Test suite
│   ├── test_document_loader.py
│   ├── test_vector_store.py
│   ├── test_rag_engine.py
│   └── test_api.py
│
├── vector_store/               # Generated at runtime (gitignored)
│
├── cli.py                      # Typer CLI commands
├── main.py                     # Entry point
├── requirements.txt
├── pytest.ini
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## Knowledge Base

| Document | Topics |
|----------|--------|
| `bookkeeping.md` | Double-entry, chart of accounts, cash vs accrual |
| `invoices.md` | Invoice types, lifecycle, GST, payment terms |
| `profit_and_loss.md` | P&L structure, margins, EBIT, net profit |
| `balance_sheet.md` | Assets, liabilities, equity, ratios |
| `cash_flow.md` | Operating/Investing/Financing, FCF, DSO/DPO |
| `accounts_payable.md` | AP process, aging, DPO, controls |
| `accounts_receivable.md` | AR process, aging, DSO, bad debt |
| `tax_basics.md` | GST, TDS, income tax, advance tax |
| `financial_ratios.md` | Liquidity, profitability, leverage, efficiency |

To add new topics: create a `.md` file in `knowledge_base/` and run `python main.py ingest --force`.

---

## Running Tests

```bash
# Run all tests (no API key required)
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Verbose
pytest -v
```

All tests use mocked embeddings and LLM clients — no Anthropic API key needed.

---

## Design Decisions

**Why FAISS over ChromaDB?**
FAISS `IndexFlatIP` gives exact nearest-neighbour search with zero infra overhead. For ~100 chunks, exact search is faster than approximate. ChromaDB is better for >100k chunks or when metadata filtering is needed.

**Why local embeddings over OpenAI Embeddings?**
`all-MiniLM-L6-v2` runs on CPU with no API cost and no rate limits. Swap to `text-embedding-3-small` by changing `EMBEDDING_MODEL` in `.env` if needed.

**Why Claude for generation?**
Claude's instruction-following and grounding reduces hallucination when context is explicitly provided. The RAG prompt instructs the model to answer only from the retrieved context.

**Chunk size 512 chars with 64-char overlap:**
Balances context density with retrieval precision. Overlap prevents information loss at chunk boundaries.

---

## License

MIT © Hellobooks AI
