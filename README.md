# NeuralMem

> **The Memory Layer AI Agents Deserve** — Local-first. MCP-native. Production-ready.

[![Tests](https://img.shields.io/badge/tests-3500%2B%20passing-brightgreen)](https://github.com/chowen-zz/neuralmem)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/neuralmem/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-native-purple)](docs/mcp-integrations.md)

**English** | [简体中文](README_zh.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [繁體中文](README_zh-TW.md)

---

## Why NeuralMem?

Even with a 200K token context window, stuffing the entire conversation history into every production request is infeasible — the cost and latency are prohibitive. **NeuralMem gives your AI agents a brain that persists across sessions.**

```python
from neuralmem import NeuralMem

mem = NeuralMem()

# Store once, remember forever
mem.remember("User prefers TypeScript for frontend, dislikes JavaScript")

# Retrieve with 4-strategy parallel search + RRF fusion
results = mem.recall("What are the user's tech preferences?")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

### What Makes It Different

| | NeuralMem | Others |
|---|---|---|
| **Install** | `pip install neuralmem` — zero deps | Docker, API keys, cloud signup |
| **Latency** | **0.7ms** recall (local SQLite) | 100-2000ms (network round-trip) |
| **Retrieval** | **4 strategies** in parallel (semantic + BM25 + graph + temporal) | Usually 1-2 strategies |
| **MCP** | **Native** — works with Claude, Cursor, Windsurf, VS Code out of the box | Bolt-on integration |
| **Cost** | **Free**, unlimited local storage | $0-249/month |
| **Graph** | **Built-in** NetworkX knowledge graph | Removed from OSS (Mem0) or requires Neo4j (Zep) |

---

## 30-Second Quick Start

```bash
pip install neuralmem
```

```python
from neuralmem import NeuralMem

mem = NeuralMem()

# Store
mem.remember("User prefers TypeScript for frontend, dislikes JavaScript")

# Recall (4-strategy parallel search + RRF fusion)
results = mem.recall("What are the user's tech preferences?")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

---

## MCP Integration — Connect in 30 Seconds

NeuralMem speaks native MCP. One config line connects to **10+ AI clients**:

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

| Client | Config Location |
|--------|----------------|
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Claude Code** | `claude mcp add neuralmem -- neuralmem mcp` |
| **Cursor** | `.cursor/mcp.json` or `~/.cursor/mcp.json` |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` |
| **Cline (VS Code)** | `~/.cline/mcp_settings.json` |
| **Continue** | `~/.continue/config.json` |
| **Zed** | `~/.config/zed/settings.json` |
| **OpenAI Codex** | `~/.codex/config.toml` |
| **Hermes Agent** | `hermes-agent.yaml` |

10 tools exposed: `remember`, `recall`, `reflect`, `forget`, `consolidate`, `remember_batch`, `forget_batch`, `export_memories`, `resolve_conflict`, `recall_with_explanation`.

[Full MCP docs →](docs/mcp-integrations.md)

---

## Architecture at a Glance

```
┌─────────────────────────────────────────┐
│           NeuralMem (Facade)            │
│  remember()  recall()  reflect() forget()│
└─────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌──────────┐   ┌──────────┐
│Extractor│   │ Retrieval│   │  Graph   │
│(LLM/   │   │ Engine   │   │ (NetworkX)│
│ rule)  │   │4-strategy│   │          │
└─────────┘   │parallel  │   └──────────┘
              │+ RRF     │
              └──────────┘
                    │
              ┌──────────┐
              │ Storage  │
              │SQLite +  │
              │sqlite-vec│
              └──────────┘
```

---

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

### Knowledge Graph with Multi-Hop Reasoning

```python
report = mem.reflect("Alice's tech stack")
# Automatically traverses: Alice → Python → Machine Learning → related memories
```

### Ebbinghaus Forgetting Curve

```python
stats = mem.consolidate()
# {"decayed": 12, "forgotten": 3, "merged": 2}
```

### Entity Disambiguation

"Alice", "my colleague Alice", and "her" resolve to the same entity — no duplicate nodes.

### Memory TTL & Batch Operations

```python
mem.remember("Temporary context", ttl=3600)  # Expires in 1 hour
mem.batch_remember(["fact 1", "fact 2", "fact 3"])
```

---

## CLI

```bash
neuralmem add "User prefers TypeScript"   # Add memory
neuralmem search "programming language"   # Search
neuralmem stats                           # Statistics
neuralmem mcp                             # Start MCP server
```

---

## Optional Enhancements

```bash
# Cross-Encoder reranking (~67 MB model)
pip install neuralmem[reranker]

# PostgreSQL + pgvector backend
pip install neuralmem[postgres]

# Local LLM extraction via Ollama
pip install neuralmem[ollama]
```

```python
from neuralmem import NeuralMem, NeuralMemConfig

# Enable reranking
mem = NeuralMem(config=NeuralMemConfig(enable_reranker=True))

# Use PostgreSQL
config = NeuralMemConfig(
    storage_backend="postgres",
    postgres_url="postgresql://user:***@localhost:5432/neuralmem"
)
mem = NeuralMem(config=config)
```

---

## Performance Benchmarks

| Metric | 100 Memories | 500 Memories |
|--------|-------------|-------------|
| **remember() throughput** | 1,452 mem/s | 1,374 mem/s |
| **recall() P50 latency** | **0.7 ms** | **0.9 ms** |
| **recall() P95 latency** | **0.8 ms** | **1.0 ms** |
| **recall() P99 latency** | **0.9 ms** | **1.1 ms** |
| **Concurrent (4 threads)** | 1,902 mem/s | — |

NeuralMem's speed comes from **local SQLite direct access** — zero network round-trips. Compare to Mem0 (50-200ms) or Zep (100-500ms).

---

## Feature Completeness

| Category | Status |
|----------|--------|
| Memory CRUD | ✅ remember / recall / reflect / forget |
| Knowledge graph | ✅ NetworkX, auto entity extraction |
| Conflict detection | ✅ Supersede mechanism |
| Entity disambiguation | ✅ "Alice" / "her" resolved |
| TTL expiry | ✅ expires_in parameter |
| Batch operations | ✅ remember_batch |
| Reflection reports | ✅ Structured reports |
| **Feature score** | **14/15 (93%)** |

---

## Comparison with Competitors

### Performance

| Metric | NeuralMem | Mem0 | Zep | Letta | LangChain |
|--------|-----------|------|-----|-------|-----------|
| Single write | **0.7 ms** | 50-200 ms | 100-500 ms | 200-800 ms | 1-5 ms |
| Single query | **0.7 ms** | 100-300 ms | 200-600 ms | 500-2000 ms | 2-10 ms |
| Batch write | **1,452/s** | 50-100/s | 20-50/s | 10-30/s | 100-500/s |
| Deploy | **Zero** | Low | High (Neo4j) | Medium | Low |
| Cost | **Free** | $0-249/mo | $0-125/mo | $0-200/mo | Free |

### Feature Matrix

| Feature | NeuralMem | Mem0 | Zep | Letta | LangChain |
|---------|-----------|------|-----|-------|-----------|
| Hybrid retrieval | ✅ **4-strategy RRF** | ✅ 3-strategy | ✅ 3-strategy | ❌ | ❌ |
| Knowledge graph | ✅ **NetworkX** | ❌ (removed v3) | ✅ Neo4j | ❌ | ❌ |
| BM25 keyword | ✅ | ✅ | ✅ | ❌ | ❌ |
| Temporal decay | ✅ | ❌ | ✅ | ❌ | ❌ |
| Conflict detection | ✅ | ❌ | ✅ | ❌ | ❌ |
| Forgetting curve | ✅ | ❌ | ❌ | ✅ | ❌ |
| Cross-Encoder rerank | ✅ | ⚠️ Paid only | ❌ | ❌ | ❌ |
| Explainability | ✅ | ❌ | ❌ | ❌ | ❌ |
| TTL expiry | ✅ | ❌ | ✅ | ❌ | ❌ |
| MCP native | ✅ | ✅ Cloud | ✅ Local | ⚠️ | ✅ |

---

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

await mem.disconnect();
```

> Requires Python 3.10+ with `neuralmem` installed. The npm package auto-starts the Python backend via MCP stdio.

---

## License

Apache-2.0. Free for commercial use.
