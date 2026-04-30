# NeuralMem

[简体中文](README.md) | **English** | [日本語](README_ja.md) | [한국어](README_ko.md) | [Tiếng Việt](README_vi.md) | [繁體中文](README_zh-TW.md)

> **Memory as Infrastructure** — Local-first, MCP-native, Enterprise-ready

Agent memory like SQLite — zero-dependency install, local-first, enterprise-scalable.

[![Tests](https://img.shields.io/badge/tests-160%20passing-brightgreen)](https://github.com/chowen-zz/neuralmem)
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
# {"decayed": 12, "forgotten": 3, "merged": 0}
```

### Knowledge Graph

Entities and relationships are automatically extracted and stored in a NetworkX graph, enabling multi-hop reasoning:

```python
report = mem.reflect("Alice's tech stack")
# Automatically traverses the graph: Alice → Python → Machine Learning → related memories
```

### Entity Disambiguation

References such as "Alice", "my colleague Alice", and "her" are automatically resolved to the same entity, preventing duplicate nodes in the graph.

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
```

```python
from neuralmem import NeuralMem
from neuralmem.core.config import NeuralMemConfig

mem = NeuralMem(config=NeuralMemConfig(enable_reranker=True))
```

## Comparison

| Feature | NeuralMem | Mem0 | Zep |
|---------|-----------|------|-----|
| Local execution | ✅ Zero deps | ❌ Requires Docker | ❌ Requires Neo4j |
| Graph features | ✅ Free | ❌ $249/month | ✅ Requires Neo4j |
| MCP native | ✅ | ✅ | ✅ |
| Forgetting curve | ✅ | ❌ | ❌ |
| Entity disambiguation | ✅ | ❌ | ✅ |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |

## License

Apache-2.0. Free to use in commercial projects.
