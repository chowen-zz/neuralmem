# NeuralMem 最佳实践

## 1. 数据库选择

| 场景 | 推荐后端 | 理由 |
|---|---|---|
| 本地开发/测试 | SQLite (默认) | 零配置，开箱即用 |
| 生产环境 (单节点) | SQLite + WAL | 简单，性能足够 |
| 生产环境 (多节点) | PostgreSQL + PGVector | 并发支持，备份方便 |
| 云原生 | Pinecone / Milvus / Weaviate | 托管服务，自动扩缩容 |
| 边缘设备 | SQLite (内存模式) | 资源占用最小 |

## 2. 配置优化

### 开发环境
```python
from neuralmem import NeuralMem

mem = NeuralMem(db_path="./dev.db")  # 最简配置
```

### 生产环境
```python
from neuralmem import NeuralMem, NeuralMemConfig

cfg = NeuralMemConfig(
    db_path="./prod.db",
    # 检索质量
    enable_reranker=True,
    enable_importance_reinforcement=True,
    # 实体提取
    enable_llm_extraction=True,
    # 生命周期
    enable_decay=True,
    decay_half_life_days=30,
    # 安全
    enable_poison_detection=True,
    max_memory_size=100_000,
)
mem = NeuralMem(config=cfg)
```

## 3. 性能调优

### 批量操作
```python
# 优于逐个 remember
contents = [f"Item {i}" for i in range(1000)]
results = mem.remember_batch(contents, batch_size=50)
```

### 查询计划
```python
from neuralmem.perf import QueryPlanner

planner = QueryPlanner()
# 自动选择最优检索策略组合
```

### 增量索引
```python
from neuralmem.perf import IncrementalIndex

index = IncrementalIndex()
# 只重新索引变更部分，避免全量重建
```

## 4. 数据安全

### 输入净化
```python
# NeuralMem 自动处理 SQL 注入和 XSS
mem.remember("'; DROP TABLE memories; --")  # 安全存储
```

### 审计日志
```python
# 所有操作自动记录到 Evidence Ledger
ledger = mem.get_ledger()
entries = ledger.query(since="2026-01-01")
```

### 访问控制
```python
cfg = NeuralMemConfig(
    enable_rbac=True,
    default_role="user"
)
```

## 5. 监控与运维

### 健康检查
```python
from neuralmem.health import HealthChecker

checker = HealthChecker(mem)
status = checker.check_all()  # 存储、嵌入、图谱状态
```

### 指标导出
```python
from neuralmem.metrics import PrometheusExporter

exporter = PrometheusExporter(mem)
exporter.start_server(port=9090)
```

### 结构化日志
```python
from neuralmem.production import StructuredLogger

logger = StructuredLogger()
logger.info("memory_stored", {"memory_id": mid, "user_id": uid})
# 输出: {"timestamp": "...", "level": "INFO", "event": "memory_stored", ...}
```

## 6. 常见陷阱

### 不要共享 NeuralMem 实例跨线程
```python
# ❌ 错误 — 共享实例跨线程
mem = NeuralMem(db_path="./shared.db")

# ✅ 正确 — 每个线程独立实例
def worker():
    mem = NeuralMem(db_path="./shared.db")  # 独立连接
    mem.remember("...")
```

### 注意内存限制
```python
# ❌ 错误 — 无限制存储
for i in range(1_000_000):
    mem.remember(f"Item {i}")

# ✅ 正确 — 设置上限并清理
mem = NeuralMem(config=NeuralMemConfig(max_memory_size=100_000))
mem.cleanup_expired()
mem.consolidate()
```

### 向量维度匹配
```python
# ❌ 错误 — 混用不同维度模型
mem1 = NeuralMem(embedding_provider="local")      # 384 dim
mem2 = NeuralMem(embedding_provider="openai")       # 1536 dim
# 同一数据库不能混用！

# ✅ 正确 — 固定模型
cfg = NeuralMemConfig(embedding_provider="local", embedding_model="BAAI/bge-small-en-v1.5")
```

## 7. 升级策略

### V0.7 → V0.8
- 新增 LLM Memory Manager — 可选启用
- 新增框架集成 — 按需导入
- 数据库结构自动迁移

### V0.8 → V0.9
- 新增生产加固模块 — 建议生产环境启用
- 新增向量存储后端 — 按需配置
- 性能优化工具 — 可选使用
