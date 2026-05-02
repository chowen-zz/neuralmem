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

## MCP Integration with Claude Desktop

Add the following to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

Once connected, Claude gains access to five tools: `remember`, `recall`, `reflect`, `forget`, and `consolidate`.

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
    postgres_url="postgresql://user:pass@localhost:5432/neuralmem"
)
mem = NeuralMem(config=config)

# Use Ollama for local LLM extraction
config = NeuralMemConfig(llm_extractor="ollama")
mem = NeuralMem(config=config)
```

## Comparison with Competitors

> Detailed analysis: [docs/competitive-analysis.md](docs/competitive-analysis.md)

### Overview

| Dimension | NeuralMem | Mem0 | Zep (Graphiti) | Letta (MemGPT) | LangChain Memory |
|-----------|-----------|------|----------------|----------------|------------------|
| **Stars** | New | 54.6k | 25.6k | 22.4k | 136k (framework) |
| **Focus** | Local memory | Universal | Temporal graph | Agent platform | Framework module |
| **Local-first** | ✅ Zero deps | ⚠️ Cloud preferred | ⚠️ Docker+Neo4j | ✅ Self-host | ✅ |
| **MCP native** | ✅ stdio | ✅ Cloud HTTP | ✅ Local | ⚠️ Indirect | ✅ Built-in |
| **Pricing** | **Free** | $0-249/mo | $0-125/mo | $0-200/mo | Free |

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
