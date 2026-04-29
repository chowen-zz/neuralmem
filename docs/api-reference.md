# API 参考

## NeuralMem

```python
from neuralmem import NeuralMem

mem = NeuralMem(
    db_path="~/.neuralmem/memory.db",  # 数据库路径
    config=None,                        # NeuralMemConfig 实例
)
```

### remember()
```python
memories = mem.remember(
    content: str,           # 要记住的内容
    user_id: str | None,    # 用户标识
    agent_id: str | None,   # Agent 标识
    memory_type: MemoryType | None,  # 记忆类型（自动推断）
    tags: list[str] | None, # 自定义标签
    importance: float | None,  # 重要性 0-1
) -> list[Memory]
```

### recall()
```python
results = mem.recall(
    query: str,              # 检索查询
    user_id: str | None,     # 用户过滤
    memory_types: list[MemoryType] | None,
    limit: int = 10,         # 最大返回数
    min_score: float = 0.3,  # 最低相关度
) -> list[SearchResult]
```

### reflect()
```python
report = mem.reflect(
    topic: str,         # 反思主题
    user_id: str | None,
    depth: int = 2,     # 图谱遍历深度
) -> str
```

### forget()
```python
count = mem.forget(
    memory_id: str | None,    # 指定记忆 ID
    user_id: str | None,      # 按用户删除
    tags: list[str] | None,   # 按标签删除
) -> int
```

### consolidate()
```python
stats = mem.consolidate(user_id=None) -> dict
# {"decayed": int, "forgotten": int, "merged": int}
```
