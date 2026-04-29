# Changelog

All notable changes to NeuralMem will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-30

### Added

#### Core Engine
- `NeuralMem` main class with `remember()`, `recall()`, `reflect()`, `forget()`, `consolidate()` API
- SQLite + sqlite-vec vector storage with numpy fallback for local-first deployment
- FastEmbed ONNX local embedding (all-MiniLM-L6-v2 model, zero API keys required)
- Rule-based memory extraction with optional Ollama LLM enhancement

#### Retrieval System
- 4-strategy parallel retrieval: semantic search, BM25 keyword matching, graph traversal, temporal search
- RRF (Reciprocal Rank Fusion) result merging for optimal ranking
- Optional Cross-Encoder reranking via `pip install neuralmem[reranker]`
- Configurable retrieval strategies and fusion weights

#### Knowledge Graph
- NetworkX in-memory graph with SQLite snapshot persistence
- Automatic entity and relation extraction from memory text
- Multi-hop reasoning via `reflect()` method
- Entity disambiguation to prevent duplicate nodes

#### Memory Lifecycle
- Ebbinghaus forgetting curve implementation: `R = e^(-t/S)`
- Spaced repetition strength calculation: `S = importance × 10 × 1.5^access_count`
- `consolidate()` for periodic memory maintenance and cleanup
- Automatic memory decay based on temporal patterns

#### Entity Disambiguation
- Two-stage disambiguation: rule filtering (substring/edit-distance) + embedding cosine similarity
- Prevents duplicate graph nodes for same-entity mentions
- Configurable similarity thresholds

#### MCP Server Integration
- FastMCP Server with 5 tools: `remember`, `recall`, `reflect`, `forget`, `consolidate`
- 1 resource: `neuralmem://stats/{user_id}` for memory statistics
- `asyncio.to_thread` bridge for sync core + async MCP compatibility
- Claude Desktop / Cursor plugin-ready

#### CLI Tools
- `neuralmem mcp` — start MCP server (stdio transport)
- `neuralmem add` — add memory with optional metadata
- `neuralmem search` — search memories with strategy selection
- `neuralmem stats` — show memory statistics and graph info

#### Testing & Quality
- 160 tests passing, 83% code coverage
- Pydantic v2 models with `frozen=True` for immutability
- `typing.Protocol` interfaces for all backend abstractions
- Thread-safe storage with `threading.Lock`
- Comprehensive test suite: unit, integration, and edge cases

### Quality Attributes
- **Zero external dependencies** — no Docker, API keys, or remote services required
- **Local-first design** — all processing happens locally with sqlite-vec
- **Thread-safe** — concurrent access support via locks and immutable models
- **Extensible** — Protocol-based backend interface for custom storage implementations
- **Well-tested** — 83% coverage with TDD approach

## Upcoming (v0.2.0 Roadmap)

- REST API Server for web integration
- TypeScript SDK for Node.js environments
- LangChain & CrewAI integration adapters
- Web Dashboard for memory visualization
- Multi-user workspace isolation
- Cross-Encoder support by default (not optional)
- Memory tagging and filtering system
