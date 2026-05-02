# NeuralMem

**简体中文** | [English](README_en.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Tiếng Việt](README_vi.md) | [繁體中文](README_zh-TW.md)

> **Memory as Infrastructure** — Local-first, MCP-native, Enterprise-ready

Agent 记忆领域的 SQLite —— 零依赖安装，本地优先，企业可扩展。

[![Tests](https://img.shields.io/badge/tests-425%20passing-brightgreen)](https://github.com/chowen-zz/neuralmem)
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
# {"decayed": 12, "forgotten": 3, "merged": 2}
```

### 知识图谱
实体和关系自动提取并存储到 NetworkX 图谱，支持多跳推理：
```python
report = mem.reflect("Alice 的技术栈")
# 自动遍历图谱：Alice → Python → 机器学习 → 相关记忆
```

### 实体消歧
"Alice"、"我同事 Alice"、"她" 自动识别为同一实体，图谱不产生重复节点。

### 记忆 TTL 与过期
```python
mem.remember("临时会话上下文", ttl=3600)  # 1 小时后自动过期
```

### 批量操作与导入
```python
mem.batch_remember(["事实1", "事实2", "事实3"])  # 批量存储
mem.import_memories(existing_memories)               # 导入已有记忆
```

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

# 使用 PostgreSQL + pgvector 替代 SQLite
pip install neuralmem[postgres]
```

```python
from neuralmem import NeuralMem
from neuralmem.core.config import NeuralMemConfig

# 启用重排序
mem = NeuralMem(config=NeuralMemConfig(enable_reranker=True))

# 使用 PostgreSQL
config = NeuralMemConfig(
    storage_backend="postgres",
    postgres_url="postgresql://user:pass@localhost:5432/neuralmem"
)
mem = NeuralMem(config=config)

# 使用 Ollama 本地 LLM 提取
config = NeuralMemConfig(llm_extractor="ollama")
mem = NeuralMem(config=config)
```

## 与竞品对比

> 详细分析见 [docs/competitive-analysis.md](docs/competitive-analysis.md)

### 概览

| 维度 | NeuralMem | Mem0 | Zep (Graphiti) | Letta (MemGPT) | LangChain Memory |
|------|-----------|------|----------------|----------------|------------------|
| **Stars** | 新项目 | 54.6k | 25.6k | 22.4k | 136k (框架) |
| **定位** | 本地记忆库 | 通用记忆层 | 时序图谱 | Agent 平台 | 框架模块 |
| **本地优先** | ✅ 零依赖 | ⚠️ 推荐云端 | ⚠️ 需 Docker+Neo4j | ✅ 可自托管 | ✅ |
| **MCP 原生** | ✅ stdio | ✅ 云端 HTTP | ✅ 本地 | ⚠️ 间接 | ✅ 内置 |
| **定价** | **免费** | $0-249/月 | $0-125/月 | $0-200/月 | 免费 |

### 功能矩阵

| 功能 | NeuralMem | Mem0 | Zep | Letta | LangChain |
|------|-----------|------|-----|-------|-----------|
| **混合检索** | ✅ 4策略 RRF | ✅ 3策略 | ✅ 3策略 | ❌ | ❌ |
| **知识图谱** | ✅ NetworkX | ❌ (已移除) | ✅ Neo4j | ❌ | ❌ |
| **BM25 关键词** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **时间衰减** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **冲突检测** | ✅ supersede | ❌ | ✅ 事实失效 | ❌ | ❌ |
| **遗忘曲线** | ✅ | ❌ | ❌ | ✅ sleep-time | ❌ |
| **Cross-Encoder 重排** | ✅ | ⚠️ 平台 | ❌ | ❌ | ❌ |
| **可解释性** | ✅ explanation | ❌ | ❌ | ❌ | ❌ |
| **TTL 过期** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **批量操作** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Embedding 选择** | 7 种 (含本地) | 17+ LLM | 1 种 | 1 种 | 多种 |

### NeuralMem 的独特优势

| vs 竞品 | NeuralMem 的差异化 |
|---------|-------------------|
| **vs Mem0** | 图谱免费 (Mem0 v3.0 已从 OSS 移除图谱); 本地优先 vs 云优先; 完全免费 vs $249/月 |
| **vs Zep** | 无需 Docker/Neo4j; NetworkX 轻量图谱; 零成本 vs $125/月 |
| **vs Letta** | 纯记忆层可组合; 4 策略检索 vs 简单归档; 但无 Skills/Sleep-time |
| **vs LangChain** | 开箱即用 vs 需自行构建; 自动提取 vs 手动; 但灵活性不如 |
| **vs LlamaIndex** | 4 策略检索 vs Memory Block; 知识图谱 vs 扁平存储 |

### 定价对比

```
NeuralMem  ██████████████████████████████  免费 (本地运行, 无限制)
Mem0 Free  ████████░░░░░░░░░░░░░░░░░░░░░░  10K add/月
Mem0 Pro   ██████████████████████████████  $249/月 (500K add)
Zep Flex   ██████████████████████████████  $125/月 (50K credits)
Letta Pro  ██████████████████████████████  $20/月 (基础)
```

## 许可证

Apache-2.0 开源，可免费用于商业项目。
