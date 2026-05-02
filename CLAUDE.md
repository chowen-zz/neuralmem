# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Run all tests
pytest

# Run only fast unit tests (skip slow/integration/stub)
pytest tests/unit/ -m "not stub and not slow"

# Run a single test file
pytest tests/unit/test_retrieval.py -v

# Run a single test by name
pytest tests/unit/test_retrieval.py::test_rrf_fusion -v

# Run with coverage
pytest --cov=neuralmem --cov-report=term-missing

# Skip integration tests that require real storage
pytest -m "not integration"

# Start MCP server (stdio transport)
neuralmem mcp

# Start MCP server with HTTP transport
neuralmem mcp --http
```

## Architecture

NeuralMem is a local-first, MCP-native agent memory library. The public surface is `NeuralMem` (in `core/memory.py`), which orchestrates all subsystems behind four sync methods: `remember()`, `recall()`, `reflect()`, `forget()`.

### Module map

```
src/neuralmem/
├── core/
│   ├── protocols.py    # Protocol interfaces for all swappable backends (import this, not implementations)
│   ├── types.py        # All frozen Pydantic models: Memory, Entity, Relation, SearchResult, SearchQuery
│   ├── config.py       # NeuralMemConfig — also reads NEURALMEM_* env vars via from_env()
│   ├── memory.py       # NeuralMem facade — the only public entry point
│   └── exceptions.py
├── storage/
│   └── sqlite.py       # SQLiteStorage — uses sqlite-vec for vector column; implements StorageProtocol
├── embedding/
│   ├── base.py          (EmbeddingBackend ABC — all backends inherit this)
│   ├── registry.py      (factory: embedding_provider string → backend instance)
│   ├── local.py         (FastEmbed, default)
│   ├── openai.py        (OpenAI text-embedding-3-*, requires neuralmem[openai])
│   ├── cohere.py        (Cohere embed-multilingual-v3.0, requires neuralmem[cohere])
│   ├── gemini.py        (Google text-embedding-004, requires neuralmem[gemini])
│   ├── huggingface.py   (HF Inference API, uses httpx — no extra needed)
│   └── azure_openai.py  (Azure OpenAI, requires neuralmem[openai])
├── extraction/
│   ├── base_llm_extractor.py  (ABC with shared JSON-parse + fallback logic)
│   ├── extractor_registry.py  (factory: llm_extractor string → extractor instance)
│   ├── extractor.py           (rule-based, default)
│   ├── llm_extractor.py       (Ollama, llm_extractor="ollama")
│   ├── openai_extractor.py    (OpenAI gpt-4o-mini, llm_extractor="openai")
│   ├── anthropic_extractor.py (Anthropic claude-haiku, llm_extractor="anthropic")
│   └── entity_resolver.py
├── graph/
│   └── knowledge_graph.py # NetworkX-backed graph; persisted as JSON snapshot in SQLite
├── retrieval/
│   ├── engine.py       # RetrievalEngine — runs 4 strategies in parallel, RRF-merges, optional rerank
│   ├── semantic.py     # Vector similarity via sqlite-vec
│   ├── keyword.py      # BM25 via rank-bm25
│   ├── graph.py        # Graph traversal strategy
│   ├── temporal.py     # Recency-weighted vector search
│   ├── fusion.py       # RRFMerger (k=60 default)
│   └── reranker.py     # CrossEncoderReranker (sentence-transformers, opt-in)
├── lifecycle/
│   ├── decay.py        # Importance decay over time
│   └── consolidation.py # Merge similar memories (stub in v0.1)
├── mcp/
│   └── server.py       # FastMCP server exposing remember/recall/reflect/forget/consolidate as tools
└── cli/
    └── main.py         # `neuralmem` CLI entrypoint
```

### Data flow: `remember()`

```
content → extractor.extract() → [ExtractedItem list]
  ↓ for each item:
embedding.encode_one() → vector
storage.find_similar(threshold=0.95) → skip if duplicate
entity_resolver.resolve() → canonical entities
Memory(frozen) → storage.save_memory()
                → graph.upsert_entity() + link_memory_to_entity()
```

### Data flow: `recall()`

```
query → SearchQuery
  ↓ RetrievalEngine.search() via ThreadPoolExecutor(4):
    ├── SemanticStrategy  → vector_search in sqlite-vec
    ├── KeywordStrategy   → BM25 over in-memory corpus
    ├── GraphStrategy     → graph traversal from matched entities
    └── TemporalStrategy  → recency-weighted vector search
  ↓ RRFMerger.merge()  → unified ranked list
  ↓ CrossEncoderReranker.rerank() (optional, enable_reranker=True)
  ↓ load full Memory objects + build SearchResult list
```

### Protocol contracts

All subsystems depend only on the Protocol types in `core/protocols.py`:
- `StorageProtocol` — vector_search, keyword_search, temporal_search, find_similar, etc.
- `EmbedderProtocol` — `dimension`, `encode(texts)`, `encode_one(text)`
- `GraphStoreProtocol` — upsert_entity, add_relation, get_neighbors, traverse_for_memories
- `LifecycleProtocol` — apply_decay, remove_forgotten

To swap a backend, implement the corresponding Protocol and inject into `NeuralMem.__init__()`.

### Configuration

All config fields in `NeuralMemConfig` have `NEURALMEM_` env var equivalents. Key toggles:
- `enable_llm_extraction=True` — switches to Ollama-backed extractor (requires local Ollama)
- `enable_reranker=True` — adds Cross-Encoder reranking (requires `pip install neuralmem[reranker]`)
- `embedding_provider` — `local` (default) | `openai` | `ollama`

### Provider Selection Quick Reference

| `embedding_provider` | Extra | Default model | Dim |
|---|---|---|---|
| `local` (default) | — | all-MiniLM-L6-v2 | 384 |
| `openai` | `neuralmem[openai]` | text-embedding-3-small | 1536 |
| `cohere` | `neuralmem[cohere]` | embed-multilingual-v3.0 | 1024 |
| `gemini` | `neuralmem[gemini]` | text-embedding-004 | 768 |
| `huggingface` | — | BAAI/bge-m3 | 1024 |
| `azure_openai` | `neuralmem[openai]` | text-embedding-3-small | 1536 |

| `llm_extractor` | Extra | Model |
|---|---|---|
| `none` (default) | — | rule-based |
| `ollama` | — | llama3.2:3b (local) |
| `openai` | `neuralmem[openai]` | gpt-4o-mini |
| `anthropic` | `neuralmem[anthropic]` | claude-haiku-4-5-20251001 |

### Test structure

```
tests/
├── unit/        # Fast, isolated, no real embedding model — use mock_embedder fixture
├── integration/ # Requires real SQLiteStorage (uses tmp_path fixtures)
├── contract/    # Protocol conformance tests for Storage/Embedder/Extractor
└── smoke/       # End-to-end sanity checks
```

pytest markers: `stub` (placeholder implementations), `integration` (real storage), `slow`.
The `conftest.py` provides `mock_embedder` (deterministic, 4-dim) and `tmp_db_path` fixtures for fast unit tests.
