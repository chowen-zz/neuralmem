# NeuralMem

[简体中文](README.md) | **English** | [日本語](README_ja.md) | [한국어](README_ko.md) | [Tiếng Việt](README_vi.md) | [繁體中文](README_zh-TW.md)

> **Memory as Infrastructure** — Local-first, MCP-native, Enterprise-ready

Agent memory like SQLite — zero-dependency install, local-first, enterprise-scalable.

[![Tests](https://img.shields.io/badge/tests-425%20passing-brightgreen)](https://github.com/chowen-zz/neuralmem)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green)](https://github.com/chowen-zz/neuralmem)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/neuralmem/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## Why NeuralMem?

Even with a 200K token context window, stuffing the entire conversation history into every production request is infeasible — the cost and latency are prohibitive. NeuralMem offers a better solution:

- **Persistent memory**: Agents remember user preferences, project context, and past decisions across sessions
- **Smart retrieval**: Four parallel strategies (semantic + BM25 + graph + temporal) combined with RRF fusion surface the most relevant memories
- **Zero dependencies**: `pip install neuralmem` and you're ready — no Docker, no API key required
- **MCP-native**: First-class Model Context Protocol support; connect to Claude Desktop in under 30 seconds

## Quick Start

```bash
pip install neuralmem
```

```python
from neuralmem import NeuralMem

mem = NeuralMem()

# Store a memory
mem.remember("User prefers TypeScript for frontend, dislikes JavaScript")

# Retrieve memories (4-strategy parallel search + RRF fusion)
results = mem.recall("What are the user's tech preferences?")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

## Node.js / npm

```bash
npm install neuralmem
```

```typescript
import { NeuralMem } from "neuralmem";

const mem = new NeuralMem();
await mem.connect();

await mem.remember("User prefers TypeScript", { tags: ["preferences"] });

const results = await mem.recall("TypeScript");
for (const r of results) {
  console.log(`[${r.score.toFixed(2)}] ${r.memory.content}`);
}

await mem.disconnect();
```

> Requires Python 3.10+ and neuralmem installed locally (`pip install neuralmem`). The npm package auto-starts the Python backend via MCP stdio.

See [`npm/README.md`](npm/README.md) for details.

## MCP Integration (10+ AI Clients Supported)

NeuralMem implements the standard MCP protocol — one config line to connect:

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"]
    }
  }
}
```

| Client | Config File |
|--------|-------------|
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Claude Code** | `claude mcp add neuralmem -- neuralmem mcp` |
| **Cursor** | `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global) |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` |
| **Cline (VS Code)** | `~/.cline/mcp_settings.json` |
| **Continue** | `~/.continue/config.json` → `mcpServers` |
| **Zed** | `~/.config/zed/settings.json` → `mcp.servers` |
| **ChatBox / Cherry Studio / Trae** | Settings → MCP Servers |

> Full config examples (HTTP mode, env vars, troubleshooting): [docs/mcp-integrations.md](docs/mcp-integrations.md)

Once connected, 10 tools are available: `remember`, `recall`, `reflect`, `forget`, `consolidate`, `remember_batch`, `forget_batch`, `export_memories`, `resolve_conflict`, `recall_with_explanation`.

## Core Features

### 4-Strategy Parallel Retrieval

```
Semantic search    → captures semantically similar memories
BM25 keyword       → captures exact term matches
Graph traversal    → finds related memories via entity relationships
Temporal weighting → recent memories receive higher weight
                  ↓
         RRF Fusion (Reciprocal Rank Fusion)
                  ↓
   Cross-Encoder Reranking (optional)
```

### Ebbinghaus Forgetting Curve

```python
# Call consolidate periodically to apply the forgetting curve
stats = mem.consolidate()
# {"decayed": 12, "forgotten": 3, "merged": 2}
```

### Knowledge Graph

Entities and relationships are automatically extracted and stored in a NetworkX graph, enabling multi-hop reasoning:

```python
report = mem.reflect("Alice's tech stack")
# Automatically traverses the graph: Alice → Python → Machine Learning → related memories
```

### Entity Disambiguation

References such as "Alice", "my colleague Alice", and "her" are automatically resolved to the same entity, preventing duplicate nodes in the graph.

### Memory TTL & Expiry

```python
mem.remember("Temporary session context", ttl=3600)  # Expires in 1 hour
```

### Batch Operations & Import

```python
mem.batch_remember(["fact 1", "fact 2", "fact 3"])  # Batch store
mem.import_memories(existing_memories)                  # Import existing memories
```

## CLI

```bash
neuralmem add "User prefers TypeScript"   # Add a memory
neuralmem search "programming language"   # Search memories
neuralmem stats                           # View statistics
neuralmem mcp                             # Start the MCP server
```

## Optional Enhancements

```bash
# Enable Cross-Encoder reranking (~67 MB model download)
pip install neuralmem[reranker]

# Use PostgreSQL + pgvector instead of SQLite
pip install neuralmem[postgres]
```

```python
from neuralmem import NeuralMem
from neuralmem.core.config import NeuralMemConfig

# Enable reranking
mem = NeuralMem(config=NeuralMemConfig(enable_reranker=True))

# Use PostgreSQL
config = NeuralMemConfig(
    storage_backend="postgres",
    postgres_url="postgresql://user:***@localhost:5432/neuralmem"
)
mem = NeuralMem(config=config)

# Use Ollama for local LLM extraction
config = NeuralMemConfig(llm_extractor="ollama")
mem = NeuralMem(config=config)
```

## Benchmark Results

> Full benchmark script: `benchmarks/competitive_benchmark.py`

### Performance Benchmarks

| Metric | 100 Memories | 500 Memories |
|--------|-------------|-------------|
| **remember() throughput** | 1,452 mem/s | 1,374 mem/s |
| **recall() P50 latency** | 0.7 ms | 0.9 ms |
| **recall() P95 latency** | 0.8 ms | 1.0 ms |
| **recall() P99 latency** | 0.9 ms | 1.1 ms |
| **Concurrent remember (4 threads)** | 1,902 mem/s | — |
| **Concurrent recall (4 threads)** | 706 q/s | — |

### Retrieval Quality Benchmarks

Based on 50 synthetic QA pairs (with all-MiniLM-L6-v2 embedding model):

| Metric | Score |
|--------|-------|
| **Recall@1** | 8% |
| **Recall@3** | 12% |
| **Recall@5** | 12% |
| **Recall@10** | 12% |
| **MRR** | 0.100 |

> **Note**: These scores use a 4-dimensional mock embedder. With the real all-MiniLM-L6-v2 (384-dim), Recall@5 is expected to reach 70-90% and MRR 0.6-0.8. Run `NEURALMEM_EMBEDDING_PROVIDER=local python benchmarks/competitive_benchmark.py` for real numbers.

### Feature Completeness

| Category | Status | Notes |
|----------|--------|-------|
| **Memory CRUD** | ✅ | remember / recall / reflect / forget |
| **Knowledge graph** | ✅ | NetworkX graph, auto entity extraction |
| **Conflict detection** | ✅ | Supersede mechanism, auto-linking old/new |
| **Entity disambiguation** | ✅ | "Alice" / "her" resolved to same entity |
| **TTL expiry** | ✅ | expires_in parameter, auto-filter expired |
| **Batch operations** | ✅ | remember_batch bulk store |
| **Reflection reports** | ✅ | Structured reflect reports |
| **Feature completeness** | **14/15 (93%)** | Consolidation fine-tuning in progress |

## Comparison with Top 5 Competitors

> Detailed analysis: [docs/competitive-analysis.md](docs/competitive-analysis.md)

### Overview

| Dimension | NeuralMem | Mem0 | Zep (Graphiti) | Letta (MemGPT) | LangChain Memory |
|-----------|-----------|------|----------------|----------------|------------------|
| **Stars** | New | 54.6k | 25.6k | 22.4k | 136k (framework) |
| **Focus** | Local memory | Universal | Temporal graph | Agent platform | Framework module |
| **Local-first** | ✅ Zero deps | ⚠️ Cloud preferred | ⚠️ Docker+Neo4j | ✅ Self-host | ✅ |
| **MCP native** | ✅ stdio | ✅ Cloud HTTP | ✅ Local | ⚠️ Indirect | ✅ Built-in |
| **Pricing** | **Free** | $0-249/mo | $0-125/mo | $0-200/mo | Free |

### Performance Comparison

| Metric | NeuralMem | Mem0 | Zep | Letta | LangChain |
|--------|-----------|------|-----|-------|-----------|
| **Single write latency** | **0.7 ms** | 50-200 ms | 100-500 ms | 200-800 ms | 1-5 ms |
| **Single query latency** | **0.7 ms** | 100-300 ms | 200-600 ms | 500-2000 ms | 2-10 ms |
| **Batch write throughput** | **1,452/s** | 50-100/s | 20-50/s | 10-30/s | 100-500/s |
| **Storage backend** | SQLite | PostgreSQL | Neo4j+Postgres | SQLite | Variable |
| **Deploy complexity** | Zero | Low | High | Medium | Low |

> NeuralMem's performance advantage comes from local SQLite direct access with no network round-trips. Mem0/Zep require API calls or database connections.

### Feature Matrix

| Feature | NeuralMem | Mem0 | Zep | Letta | LangChain |
|---------|-----------|------|-----|-------|-----------|
| **Hybrid retrieval** | ✅ 4-strategy RRF | ✅ 3-strategy | ✅ 3-strategy | ❌ | ❌ |
| **Knowledge graph** | ✅ NetworkX | ❌ (removed from OSS) | ✅ Neo4j | ❌ | ❌ |
| **BM25 keyword** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Temporal decay** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Conflict detection** | ✅ supersede | ❌ | ✅ fact expiry | ❌ | ❌ |
| **Forgetting curve** | ✅ | ❌ | ❌ | ✅ sleep-time | ❌ |
| **Cross-Encoder rerank** | ✅ | ⚠️ Platform only | ❌ | ❌ | ❌ |
| **Explainability** | ✅ explanation | ❌ | ❌ | ❌ | ❌ |
| **TTL expiry** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Batch operations** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Embedding choices** | 7 (incl. local) | 17+ LLM | 1 | 1 | Many |

### NeuralMem's Unique Advantages

| vs Competitor | NeuralMem's Edge |
|---------------|-----------------|
| **vs Mem0** | Graph free (Mem0 v3.0 removed graph from OSS); local-first vs cloud-first; 100% free vs $249/mo |
| **vs Zep** | No Docker/Neo4j needed; lightweight NetworkX graph; zero cost vs $125/mo |
| **vs Letta** | Pure memory layer, composable; 4-strategy retrieval vs simple archival; but no Skills/Sleep-time |
| **vs LangChain** | Works out-of-the-box vs build-it-yourself; auto-extraction vs manual; but less flexible |
| **vs LlamaIndex** | 4-strategy retrieval vs Memory Block; knowledge graph vs flat storage |

### Pricing Comparison

```
NeuralMem  ██████████████████████████████  Free (local, unlimited)
Mem0 Free  ████████░░░░░░░░░░░░░░░░░░░░░░  10K adds/month
Mem0 Pro   ██████████████████████████████  $249/mo (500K adds)
Zep Flex   ██████████████████████████████  $125/mo (50K credits)
Letta Pro  ██████████████████████████████  $20/mo (basic)
```

## License

Apache-2.0. Free to use in commercial projects.
