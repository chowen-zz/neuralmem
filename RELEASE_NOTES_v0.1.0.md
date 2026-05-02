# NeuralMem v0.1.0 Release Notes

**Release Date:** April 30, 2026

NeuralMem v0.1.0 is the inaugural public release of an autonomous agent memory system built on Ebbinghaus forgetting curves and knowledge graphs.

## Installation

```bash
pip install neuralmem
```

For optional cross-encoder reranking support:

```bash
pip install neuralmem[reranker]
```

## Quick Start

```python
from neuralmem import NeuralMem

# Initialize (creates SQLite db in memory)
mem = NeuralMem()

# Remember a fact
mem.remember("Claude is an AI assistant made by Anthropic")

# Recall with 4-strategy retrieval
results = mem.recall("Who made Claude?")
# Returns: [Memory(content="Claude is an AI...", relevance=0.95)]

# Reflect to build knowledge graph
reasoning = mem.reflect("What companies make AI?")
# Uses graph traversal + semantic search

# Forget irrelevant memories
mem.forget("outdated_memory_id")

# Maintain memory health
mem.consolidate()  # Compact storage, apply forgetting curves
```

## Key Features

### 3-Line Integration
```python
mem = NeuralMem()
mem.remember("Important fact")
results = mem.recall("question")
```

### Zero External Dependencies
- No Docker required
- No API keys needed
- No remote service calls
- Local-first design with SQLite + sqlite-vec

### Four-Strategy Retrieval
1. **Semantic Search** — Embedding-based similarity (FastEmbed ONNX)
2. **BM25 Keyword Matching** — Exact term matching with TF-IDF
3. **Graph Traversal** — Entity relationship reasoning
4. **Temporal Search** — Recent memory prioritization

Results merged via RRF (Reciprocal Rank Fusion) for optimal ranking.

### MCP Integration (30 seconds)
Integrate with Claude Desktop / Cursor:

```bash
neuralmem mcp
```

Exposes 5 tools: `remember`, `recall`, `reflect`, `forget`, `consolidate`

### Ebbinghaus Forgetting Curves
Automatic memory decay based on:
- Importance score (user-assigned)
- Access frequency (implicit)
- Time elapsed since last review

Formula: `R = e^(-t/S)` where S = importance × 10 × 1.5^access_count

### Knowledge Graph
Automatic entity + relation extraction:
- NetworkX in-memory graph
- SQLite persistence snapshots
- Multi-hop reasoning for `reflect()`
- Entity deduplication prevents node pollution

### CLI Commands
```bash
# Add memory
neuralmem add "Fact to remember"

# Search with strategy selection
neuralmem search "question" --strategy semantic

# View statistics
neuralmem stats

# Start MCP server
neuralmem mcp
```

## Architecture Highlights

### Storage Layer
- **Primary:** SQLite with sqlite-vec for vector storage
- **Fallback:** NumPy for systems without sqlite-vec
- **Snapshots:** Periodic SQLite exports for disaster recovery

### Embedding Model
- **FastEmbed ONNX:** all-MiniLM-L6-v2 (384-dim, 33MB)
- **Zero Setup:** Pre-trained, no downloads needed
- **Privacy:** 100% local, no telemetry

### Immutability
- Pydantic v2 with `frozen=True`
- No mutation of original objects
- Thread-safe via `threading.Lock`

### Extensibility
All components use `typing.Protocol`:
- `StorageBackend` — custom persistence
- `EmbeddingModel` — custom embeddings
- `Extractor` — custom fact extraction
- Swap implementations without changing core API

## Known Limitations

### v0.1.0 Scope
- **Single-user only** — No multi-user workspace isolation yet (v0.2.0)
- **Cross-Encoder optional** — Install `neuralmem[reranker]` for enhanced ranking
- **Rule-based extraction** — NLP extraction limited for non-English text (Ollama can enhance)
- **In-memory graph** — Large knowledge graphs (>100K entities) may consume memory

### Workarounds
- Use Ollama locally for LLM-based extraction: `mem = NeuralMem(extractor=OllamaExtractor())`
- Implement custom `StorageBackend` for scale-out storage
- Call `consolidate()` periodically to manage memory footprint

## Testing & Quality

- **160 tests** passing, 83% code coverage
- **TDD approach:** All code written test-first
- **Immutable models:** Pydantic v2 `frozen=True`
- **Thread-safe:** Concurrent access via locks
- **Type-safe:** Full type hints, Protocol-based abstractions

Run tests:
```bash
pytest tests/ -v --cov=src/neuralmem --cov-report=term-out
```

## Next Release (v0.2.0)

- REST API server for web integration
- TypeScript/JavaScript SDK
- LangChain & CrewAI adapter plugins
- Web dashboard for memory visualization
- Multi-user workspace isolation
- Default Cross-Encoder reranking (no optional install)
- Memory tagging and advanced filtering

## Support

- **GitHub:** https://github.com/your-org/neuralmem
- **Docs:** https://neuralmem.readthedocs.io/
- **Issues:** GitHub Issues for bug reports
- **Contributing:** See CONTRIBUTING.md

## License

Apache License 2.0 — See LICENSE file for details.

---

**Happy memorizing! 🧠**
