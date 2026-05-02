# NeuralMem

> **Memory as Infrastructure** — Local-first, MCP-native, Enterprise-ready

Agent 记忆领域的 SQLite —— 零依赖安装，本地优先，企业可扩展。

## 3 行代码快速开始

```python
from neuralmem import NeuralMem

mem = NeuralMem()
mem.remember("用户偏好 TypeScript 写前端")
results = mem.recall("用户喜欢什么语言？")
```

## 核心优势

| 特性 | 说明 |
|------|------|
| 零依赖 | `pip install neuralmem`，无需 Docker、无需 API Key |
| 智能检索 | 四策略并行（语义/BM25/图谱/时序）+ RRF 融合 |
| MCP 原生 | 30 秒接入 Claude Desktop / Cursor |
| 遗忘曲线 | Ebbinghaus 算法自动衰减不重要的记忆 |
| 知识图谱 | 实体关系自动提取，支持多跳推理 |
| 本地优先 | 数据默认不离开你的设备 |

## 下一步

- [快速开始](quickstart.md) — 5 分钟内完成第一条记忆存取
- [MCP 集成](guides/claude-code.md) — 接入 Claude Desktop
- [架构概述](concepts/architecture.md) — 了解内部原理
