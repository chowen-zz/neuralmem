# NeuralMem

**简体中文** | [English](README_en.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Tiếng Việt](README_vi.md) | [繁體中文](README_zh-TW.md)

> **Memory as Infrastructure** — Local-first, MCP-native, Enterprise-ready

Agent 记忆领域的 SQLite —— 零依赖安装，本地优先，企业可扩展。

[![Tests](https://img.shields.io/badge/tests-160%20passing-brightgreen)](https://github.com/chowen-zz/neuralmem)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green)](https://github.com/chowen-zz/neuralmem)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/neuralmem/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## 为什么需要 NeuralMem？

即使拥有 200K token 的上下文窗口，在生产环境中塞入全部对话历史也不可行——成本和延迟都无法接受。NeuralMem 提供了更好的解决方案：

- **持久记忆**：Agent 跨会话记住用户偏好、项目背景、历史决策
- **智能检索**：四策略并行（语义 + BM25 + 图谱 + 时序）+ RRF 融合，找到最相关的记忆
- **零依赖**：`pip install neuralmem` 即用，无需 Docker、无需 API Key
- **MCP 原生**：一等公民支持 Model Context Protocol，30 秒接入 Claude Desktop

## 快速开始

```bash
pip install neuralmem
```

```python
from neuralmem import NeuralMem

mem = NeuralMem()

# 存储记忆
mem.remember("用户偏好 TypeScript 写前端，讨厌 JavaScript")

# 检索记忆（四策略并行 + RRF 融合）
results = mem.recall("用户的技术偏好是什么？")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

## MCP 接入 Claude Desktop

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`：

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

接入后 Claude 可以调用 5 个工具：`remember`、`recall`、`reflect`、`forget`、`consolidate`。

## 核心特性

### 四策略并行检索
```
语义搜索 → 捕获语义相近的记忆
BM25 关键词 → 捕获精确术语匹配
图谱遍历 → 通过实体关系找关联记忆
时序加权 → 近期记忆权重更高
         ↓
    RRF 融合（Reciprocal Rank Fusion）
         ↓
  Cross-Encoder 重排序（可选）
```

### Ebbinghaus 遗忘曲线
```python
# 定期调用 consolidate 应用遗忘曲线
stats = mem.consolidate()
# {"decayed": 12, "forgotten": 3, "merged": 0}
```

### 知识图谱
实体和关系自动提取并存储到 NetworkX 图谱，支持多跳推理：
```python
report = mem.reflect("Alice 的技术栈")
# 自动遍历图谱：Alice → Python → 机器学习 → 相关记忆
```

### 实体消歧
"Alice"、"我同事 Alice"、"她" 自动识别为同一实体，图谱不产生重复节点。

## CLI

```bash
neuralmem add "用户偏好 TypeScript"      # 添加记忆
neuralmem search "编程语言"              # 搜索记忆
neuralmem stats                         # 查看统计
neuralmem mcp                           # 启动 MCP Server
```

## 可选增强

```bash
# 启用 Cross-Encoder 重排序（~67MB 模型）
pip install neuralmem[reranker]
```

```python
from neuralmem import NeuralMem
from neuralmem.core.config import NeuralMemConfig

mem = NeuralMem(config=NeuralMemConfig(enable_reranker=True))
```

## 与竞品对比

| 特性 | NeuralMem | Mem0 | Zep |
|------|-----------|------|-----|
| 本地运行 | ✅ 零依赖 | ❌ 需 Docker | ❌ 需 Neo4j |
| 图谱功能 | ✅ 免费 | ❌ $249/月 | ✅ 需 Neo4j |
| MCP 原生 | ✅ | ✅ | ✅ |
| 遗忘曲线 | ✅ | ❌ | ❌ |
| 实体消歧 | ✅ | ❌ | ✅ |
| 开源协议 | Apache-2.0 | Apache-2.0 | Apache-2.0 |

## 许可证

Apache-2.0 开源，可免费用于商业项目。
