# NeuralMem

[简体中文](README_zh.md) | [English](README.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | [Tiếng Việt](README_vi.md) | **繁體中文**

> **Memory as Infrastructure** — Local-first, MCP-native, Enterprise-ready

Agent 記憶領域的 SQLite —— 零依賴安裝，本地優先，企業可擴展。

[![Tests](https://img.shields.io/badge/tests-160%20passing-brightgreen)](https://github.com/chowen-zz/neuralmem)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green)](https://github.com/chowen-zz/neuralmem)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/neuralmem/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## 為什麼需要 NeuralMem？

即使擁有 200K token 的上下文視窗，在生產環境中塞入全部對話歷史也不可行——成本和延遲都無法接受。NeuralMem 提供了更好的解決方案：

- **持久記憶**：Agent 跨會話記住使用者偏好、專案背景、歷史決策
- **智慧檢索**：四策略平行（語義 + BM25 + 圖譜 + 時序）+ RRF 融合，找到最相關的記憶
- **零依賴**：`pip install neuralmem` 即用，無需 Docker、無需 API Key
- **MCP 原生**：一等公民支援 Model Context Protocol，30 秒接入 Claude Desktop

## 快速開始

```bash
pip install neuralmem
```

```python
from neuralmem import NeuralMem

mem = NeuralMem()

# 儲存記憶
mem.remember("使用者偏好 TypeScript 寫前端，討厭 JavaScript")

# 檢索記憶（四策略平行 + RRF 融合）
results = mem.recall("使用者的技術偏好是什麼？")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

## MCP 接入 Claude Desktop

編輯 `~/Library/Application Support/Claude/claude_desktop_config.json`：

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

接入後 Claude 可以呼叫 5 個工具：`remember`、`recall`、`reflect`、`forget`、`consolidate`。

## 核心特性

### 四策略平行檢索
```
語義搜尋 → 捕獲語義相近的記憶
BM25 關鍵詞 → 捕獲精確術語匹配
圖譜遍歷 → 透過實體關係找關聯記憶
時序加權 → 近期記憶權重更高
         ↓
    RRF 融合（Reciprocal Rank Fusion）
         ↓
  Cross-Encoder 重排序（可選）
```

### Ebbinghaus 遺忘曲線
```python
# 定期呼叫 consolidate 應用遺忘曲線
stats = mem.consolidate()
# {"decayed": 12, "forgotten": 3, "merged": 0}
```

### 知識圖譜
實體和關係自動提取並儲存到 NetworkX 圖譜，支援多跳推理：
```python
report = mem.reflect("Alice 的技術棧")
# 自動遍歷圖譜：Alice → Python → 機器學習 → 相關記憶
```

### 實體消歧
「Alice」、「我同事 Alice」、「她」自動識別為同一實體，圖譜中只有一個 Alice 節點。

## CLI

```bash
neuralmem add "使用者偏好 TypeScript"      # 新增記憶
neuralmem search "程式語言"               # 搜尋記憶
neuralmem stats                          # 查看統計
neuralmem mcp                            # 啟動 MCP Server
```

## 可選增強

```bash
# 啟用 Cross-Encoder 重排序（~67MB 模型）
pip install neuralmem[reranker]
```

```python
from neuralmem import NeuralMem
from neuralmem.core.config import NeuralMemConfig

mem = NeuralMem(config=NeuralMemConfig(enable_reranker=True))
```

## 與競品對比

| 特性 | NeuralMem | Mem0 | Zep |
|------|-----------|------|-----|
| 本地執行 | ✅ 零依賴 | ❌ 需 Docker | ❌ 需 Neo4j |
| 圖譜功能 | ✅ 免費 | ❌ $249/月 | ✅ 需 Neo4j |
| MCP 原生 | ✅ | ✅ | ✅ |
| 遺忘曲線 | ✅ | ❌ | ❌ |
| 實體消歧 | ✅ | ❌ | ✅ |
| 開源協議 | Apache-2.0 | Apache-2.0 | Apache-2.0 |

## 授權條款

Apache-2.0 開源，可免費用於商業專案。
