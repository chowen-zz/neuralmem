# NeuralMem

> Memory as Infrastructure — Local-first, MCP-native, Enterprise-ready

Agent 记忆领域的 SQLite —— 零依赖安装，本地优先，企业可扩展。

## Quick Start

```bash
pip install neuralmem
```

```python
from neuralmem import NeuralMem

mem = NeuralMem()
mem.remember("User prefers TypeScript for frontend development")
results = mem.recall("What are the user's tech preferences?")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

## MCP Integration (Claude Desktop / Cursor)

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"],
      "env": {
        "NEURALMEM_DB_PATH": "~/.neuralmem/memory.db"
      }
    }
  }
}
```

## Features

- **Local-first**: Zero external API dependencies (no OpenAI key needed)
- **4-Strategy Retrieval**: Semantic + BM25 keyword + graph traversal + temporal filtering with RRF fusion
- **Knowledge Graph**: NetworkX-powered entity/relation extraction
- **MCP Native**: First-class Model Context Protocol support
- **Privacy**: All data stays on your machine

## License

AGPL-3.0. Commercial license available at neuralmem.dev/pricing.
