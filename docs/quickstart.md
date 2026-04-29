# 快速开始

## 安装

```bash
pip install neuralmem
```

!!! note "首次运行"
    第一次调用 `remember()` 时会自动下载 FastEmbed 模型（约 80MB），请确保网络连接。

## 基础用法

```python
from neuralmem import NeuralMem

mem = NeuralMem()

# 存储记忆
memories = mem.remember("用户偏好 TypeScript，讨厌 JavaScript")
print(f"存储了 {len(memories)} 条记忆")

# 检索记忆
results = mem.recall("用户的技术偏好")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

## 多用户场景

```python
# 每个用户拥有独立的记忆空间
mem.remember("Alice 负责后端开发", user_id="alice")
mem.remember("Bob 是 UI 设计师", user_id="bob")

alice_memories = mem.recall("职责是什么", user_id="alice")
```

## MCP 接入

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

## 自定义配置

```python
from neuralmem import NeuralMem
from neuralmem.core.config import NeuralMemConfig

mem = NeuralMem(config=NeuralMemConfig(
    db_path="./my_project.db",
    enable_reranker=True,  # 需要 pip install neuralmem[reranker]
))
```
