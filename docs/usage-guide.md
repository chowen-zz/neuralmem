# NeuralMem 完整使用指南

## 安装

```bash
pip install neuralmem
```

## 最简使用（3 行代码）

```python
from neuralmem import NeuralMem

mem = NeuralMem()
mem.remember("用户偏好用 TypeScript 写前端")
results = mem.recall("用户的技术偏好是什么？")
```

---

## 核心 API

### 1. remember() — 存储记忆

```python
memories = mem.remember(
    content: str,
    *,
    user_id: str | None = None,      # 用户标识（多用户隔离）
    agent_id: str | None = None,     # Agent 标识
    session_id: str | None = None,   # 会话标识
    memory_type: MemoryType | None = None,  # 记忆类型（自动推断）
    tags: list[str] | None = None,   # 标签
    importance: float | None = None,  # 重要性（0-1，自动计算）
    expires_at: datetime | None = None,       # 绝对过期时间
    expires_in: timedelta | None = None,      # 相对过期时间
    infer: bool = True,              # 是否自动提取实体
    metadata: dict | None = None,    # 自定义元数据
)
```

**返回**: `list[Memory]` — 提取并存储的记忆列表（一段内容可能提取多条记忆）

**示例**:

```python
from datetime import timedelta

# 基础存储
mem.remember("用户偏好用 TypeScript 写前端")

# 带标签和过期时间
mem.remember(
    "临时会话上下文",
    tags=["session", "temp"],
    expires_in=timedelta(hours=1)
)

# 多用户隔离
mem.remember("Alice 喜欢 Python", user_id="alice")
mem.remember("Bob 喜欢 JavaScript", user_id="bob")
```

---

### 2. recall() — 检索记忆

```python
results = mem.recall(
    query: str,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    memory_types: list[MemoryType] | None = None,
    tags: list[str] | None = None,
    time_range: tuple[datetime, datetime] | None = None,
    limit: int = 10,        # 返回数量
    min_score: float = 0.3, # 最低相关性分数
)
```

**返回**: `list[SearchResult]` — 按相关性排序的搜索结果

**示例**:

```python
# 基础检索
results = mem.recall("用户的技术偏好")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")

# 按标签过滤
results = mem.recall("技术", tags=["preferences"])

# 按时间范围
from datetime import datetime, timedelta
last_week = datetime.now() - timedelta(days=7)
results = mem.recall("会议", time_range=(last_week, datetime.now()))

# 多用户隔离检索
results = mem.recall("偏好", user_id="alice")
```

---

### 3. reflect() — 记忆推理

```python
report = mem.reflect(
    topic: str,
    *,
    user_id: str | None = None,
    depth: int = 2,  # 图谱遍历深度
)
```

**返回**: `str` — Markdown 格式的推理报告

**示例**:

```python
report = mem.reflect("Alice 的技术栈")
print(report)
# # Reflection on: Alice 的技术栈
# 
# ## Direct Memories
# 1. [0.95] Alice 喜欢 Python
# 2. [0.88] Alice 用 Django 做后端
# 
# ## Related Entities
# - Python (language)
# - Django (framework)
# - Machine Learning (topic)
```

---

### 4. forget() — 删除记忆

```python
count = mem.forget(
    memory_id: str | None = None,    # 指定 ID 删除
    *,
    user_id: str | None = None,      # 删除某用户的全部记忆
    before: datetime | None = None,   # 删除某时间之前的记忆
    tags: list[str] | None = None,    # 删除带某标签的记忆
)
```

**返回**: `int` — 删除的记忆数量

**示例**:

```python
# 删除指定记忆
mem.forget("memory_id_xxx")

# 删除临时标签的全部记忆
mem.forget(tags=["temp"])

# 删除一周前的记忆
from datetime import datetime, timedelta
mem.forget(before=datetime.now() - timedelta(days=7))
```

---

### 5. consolidate() — 记忆整理

```python
stats = mem.consolidate(user_id: str | None = None)
```

**返回**: `dict[str, int]` — `{"decayed": N, "forgotten": N, "merged": N}`

**示例**:

```python
stats = mem.consolidate()
print(f"衰减: {stats['decayed']}, 遗忘: {stats['forgotten']}, 合并: {stats['merged']}")
```

---

## 批量操作

### remember_batch() — 批量存储

```python
memories = mem.remember_batch(
    contents: list[str],
    *,
    user_id: str | None = None,
    tags: list[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
)
```

**示例**:

```python
def on_progress(current, total, preview):
    print(f"{current}/{total}: {preview}")

memories = mem.remember_batch(
    ["事实1", "事实2", "事实3"],
    tags=["batch"],
    progress_callback=on_progress,
)
```

### forget_batch() — 批量删除

```python
count = mem.forget_batch(memory_ids: list[str] | None = None)
```

---

## 导入导出

### export_memories() — 导出

```python
data = mem.export_memories(
    *,
    user_id: str | None = None,
    tags: list[str] | None = None,
    format: ExportFormat = ExportFormat.JSON,
)
```

**示例**:

```python
from neuralmem.core.types import ExportFormat

# JSON 导出
json_data = mem.export_memories(format=ExportFormat.JSON)

# CSV 导出
csv_data = mem.export_memories(format=ExportFormat.CSV)

# 按标签导出
data = mem.export_memories(tags=["important"])
```

### import_memories() — 导入

```python
mem.import_memories(data: str | list[dict])
```

**示例**:

```python
# 从 JSON 字符串导入
mem.import_memories(json_data)

# 从列表导入
mem.import_memories([
    {"content": "事实1", "tags": ["tag1"]},
    {"content": "事实2", "tags": ["tag2"]},
])
```

---

## 冲突解决

### resolve_conflict() — 手动解决冲突

```python
success = mem.resolve_conflict(
    memory_id: str,
    *,
    action: str = "reactivate",  # "reactivate" 或 "delete"
)
```

**示例**:

```python
# 重新激活被覆盖的旧记忆
mem.resolve_conflict("old_memory_id", action="reactivate")

# 永久删除冲突记忆
mem.resolve_conflict("conflict_id", action="delete")
```

---

## 统计信息

### get_stats() — 获取统计

```python
stats = mem.get_stats()
```

**返回**: `dict` — 存储统计 + 图谱统计

**示例**:

```python
stats = mem.get_stats()
print(f"记忆总数: {stats.get('total_memories', 0)}")
print(f"实体数量: {stats.get('entity_count', 0)}")
print(f"关系数量: {stats.get('relation_count', 0)}")
```

---

## 配置选项

### NeuralMemConfig

```python
from neuralmem import NeuralMem, NeuralMemConfig

config = NeuralMemConfig(
    db_path="~/.neuralmem/memory.db",     # 数据库路径
    enable_reranker=False,                 # 启用 Cross-Encoder 重排序
    enable_llm_extraction=False,           # 启用 LLM 实体提取（需 Ollama）
    enable_importance_reinforcement=True,  # 访问时提升重要性
    embedding_provider="local",            # local / openai / ollama
    llm_extractor="none",                  # none / ollama / openai
    storage_backend="sqlite",              # sqlite / postgres
)

mem = NeuralMem(config=config)
```

---

## 完整示例：聊天机器人记忆

```python
from neuralmem import NeuralMem
from datetime import timedelta

mem = NeuralMem()

# 存储用户偏好
mem.remember("用户叫 Alice，喜欢 Python 和机器学习", tags=["user_profile"])
mem.remember("用户讨厌 Java，觉得太啰嗦", tags=["user_profile"])

# 存储对话历史
mem.remember("用户问：如何用 PyTorch 训练 CNN？", tags=["conversation"])
mem.remember("用户说：之前的解释太复杂了，简单点", tags=["conversation"])

# 检索相关记忆
results = mem.recall("用户的技术偏好")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")

# 推理总结
report = mem.reflect("Alice 的编程偏好")
print(report)

# 定期整理
stats = mem.consolidate()
print(f"整理了 {stats['merged']} 条相似记忆")

# 导出重要记忆
important = mem.export_memories(tags=["user_profile"])
print(important)
```

---

## MCP 集成

连接 Claude Desktop:

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

连接后可用 10 个工具:
- `remember` — 存储记忆
- `recall` — 检索记忆
- `reflect` — 推理总结
- `forget` — 删除记忆
- `consolidate` — 记忆整理
- `remember_batch` — 批量存储
- `forget_batch` — 批量删除
- `export_memories` — 导出记忆
- `resolve_conflict` — 解决冲突
- `recall_with_explanation` — 带解释的检索

---

## CLI 命令

```bash
neuralmem add "用户偏好 TypeScript"     # 添加记忆
neuralmem search "编程语言"             # 搜索记忆
neuralmem stats                         # 查看统计
neuralmem mcp                           # 启动 MCP 服务器
```
