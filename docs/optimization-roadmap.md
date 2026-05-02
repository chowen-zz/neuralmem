# NeuralMem 优化项清单

> 生成时间: 2026-05-02 | 代码量: 4,553 行 | 测试: 283 个

---

## 一、性能优化 (P0 — 高优先级)

### 1.1 remember() 中 N 次 embed 调用 → 批量编码
- **位置**: `src/neuralmem/core/memory.py:102-103`
- **现状**: `for item in extracted: vector = self.embedding.encode_one(item.content)` 逐条编码
- **优化**: 提取所有 content 后调用 `self.embedding.encode(texts)` 一次批量编码
- **收益**: N 条记忆从 N 次模型推理降为 1 次，吞吐量提升 N 倍
- **复杂度**: 低

### 1.2 remember() 中 N 次 find_similar → 批量去重
- **位置**: `src/neuralmem/core/memory.py:106-121`
- **现状**: 每条记忆单独调用 `find_similar()` 做去重+冲突检测（2 次 DB 查询）
- **优化**: 先批量编码，再用单次 SQL 查询做批量相似度匹配
- **收益**: N 条记忆从 2N 次 DB 查询降为 1-2 次
- **复杂度**: 中

### 1.3 recall() 中逐条 record_access → 批量更新
- **位置**: `src/neuralmem/core/memory.py:209-214`
- **现状**: `for result in results: self.storage.record_access(...)` 逐条 DB 写入
- **优化**: 收集所有 ID 后用一条 `UPDATE ... WHERE id IN (...)` 批量更新
- **收益**: N 条结果从 N 次写入降为 1 次
- **复杂度**: 低

### 1.4 SQLite 无索引 → 添加关键索引
- **位置**: `src/neuralmem/storage/sqlite.py`
- **现状**: `CREATE INDEX` 语句数为 0，全表扫描
- **优化**: 添加索引：
  ```sql
  CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
  CREATE INDEX IF NOT EXISTS idx_memories_is_active ON memories(is_active);
  CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
  CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
  CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
  ```
- **收益**: user_id/is_active 过滤查询从 O(N) 降为 O(log N)
- **复杂度**: 低

### 1.5 每次 recall 创建新 ThreadPoolExecutor → 复用
- **位置**: `src/neuralmem/retrieval/engine.py`
- **现状**: 每次 `search()` 创建新的 `ThreadPoolExecutor(max_workers=4)`
- **优化**: 在 `RetrievalEngine.__init__()` 中创建持久 executor，`__del__` 中 shutdown
- **收益**: 避免线程创建/销毁开销，高频调用场景提升明显
- **复杂度**: 低

### 1.6 encode_one() 重复编码 → LRU 缓存
- **位置**: `src/neuralmem/embedding/local.py:81-83`
- **现状**: `encode_one(text)` 每次都调用模型，无缓存
- **优化**: 对 `encode_one` 添加 `@functools.lru_cache(maxsize=512)` 或自定义 TTL 缓存
- **收益**: 相同文本重复查询时直接命中缓存
- **复杂度**: 低

---

## 二、架构优化 (P1 — 中优先级)

### 2.1 ConsolidationEngine 仍是 stub → 实现真正合并
- **位置**: `src/neuralmem/lifecycle/consolidation.py:108`
- **现状**: `merge_similar` 无 storage/embedder 时退化为 stub，返回 0
- **优化**: 实现完整的相似记忆合并逻辑：检测→合并内容→保留最高效用→删除旧记录
- **收益**: 长期运行后记忆膨胀问题的根本解决
- **复杂度**: 中

### 2.2 知识图谱全量 JSON 快照 → 增量持久化
- **位置**: `src/neuralmem/graph/knowledge_graph.py`
- **现状**: 每次 `_persist()` 将整个 NetworkX 图序列化为 JSON 存入 SQLite
- **优化**: 
  - 方案 A: 使用 SQLite 存储节点/边表（增量读写）
  - 方案 B: 增量 JSON patch（仅保存 diff）
- **收益**: 图谱大时避免每次全量序列化（当前 O(N) 序列化开销）
- **复杂度**: 高

### 2.3 单线程 SQLite 连接 → 连接池或 WAL 模式
- **位置**: `src/neuralmem/storage/sqlite.py`
- **现状**: 使用 `threading.local()` 为每个线程创建连接，无 WAL
- **优化**: 
  - 启用 WAL 模式：`PRAGMA journal_mode=WAL`
  - 启用内存映射：`PRAGMA mmap_size=268435456`
  - 连接池化（跨线程复用只读连接）
- **收益**: 并发读取性能提升 2-5x
- **复杂度**: 低

### 2.4 四策略检索全部返回过多结果 → 限制候选集
- **位置**: `src/neuralmem/retrieval/engine.py`
- **现状**: 四个策略各自返回结果，RRF 合并后可能有大量低质量候选
- **优化**: 
  - 各策略设置 `max_candidates` 上限（如 50）
  - RRF 合并后先做分数阈值过滤再重排
- **收益**: 减少重排序计算量，降低内存占用
- **复杂度**: 低

### 2.5 实体解析每次调用 graph.get_entities() → 缓存
- **位置**: `src/neuralmem/core/memory.py:98, 126`
- **现状**: `remember()` 中两次调用 `self.graph.get_entities(user_id)` 返回全量实体列表
- **优化**: 在 `remember()` 开始时缓存一次，后续复用
- **收益**: 减少重复图遍历
- **复杂度**: 低

---

## 三、功能补全 (P1 — 中优先级)

### 3.1 缺少 Ollama embedding provider
- **现状**: 支持 local/OpenAI/Cohere/Gemini/HF/Azure，但不支持 Ollama
- **优化**: 添加 `src/neuralmem/embedding/ollama.py`
- **收益**: 竞品 Mem0 已支持；本地 LLM 用户的刚需
- **复杂度**: 低

### 3.2 缺少 PostgreSQL/pgvector 后端
- **现状**: 仅 SQLite，无法支撑生产环境大数据量
- **优化**: 添加 `src/neuralmem/storage/pgvector.py`，实现 StorageProtocol
- **收益**: 解决规模化问题，对标 Mem0 的 25+ 后端
- **复杂度**: 高

### 3.3 缺少异步 API (async/await)
- **现状**: 所有 API 都是同步的，FastAPI/async 场景会阻塞事件循环
- **优化**: 为 NeuralMem 添加 `aremember()`, `arecall()` 等异步方法
- **收益**: 与现代 Python Web 框架兼容
- **复杂度**: 中

### 3.4 缺少记忆导入功能
- **现状**: 有 `export_memories()` 但无对应的 `import_memories()`
- **优化**: 添加 JSON/Markdown/CSV 导入
- **收益**: 数据迁移、备份恢复、跨实例同步
- **复杂度**: 低

### 3.5 缺少记忆过期/TTL 机制
- **现状**: 记忆永不过期，只能手动 forget
- **优化**: 添加 `expires_at` 字段 + 自动清理 job
- **收益**: 自动管理临时性记忆（如会话上下文）
- **复杂度**: 低

---

## 四、代码质量 (P2 — 低优先级)

### 4.1 4 处 bare except → 精确异常处理
- **位置**:
  - `session/context.py:77` — `except Exception:`
  - `session/context.py:242` — `except Exception:`
  - `extraction/llm_extractor.py:26` — `except Exception:`
  - `graph/knowledge_graph.py:210` — `except Exception:  # noqa: BLE001`
- **优化**: 替换为具体异常类型（`sqlite3.Error`, `json.JSONDecodeError` 等）
- **收益**: 避免吞掉意外异常，便于调试
- **复杂度**: 低

### 4.2 记忆 ID 使用 UUID4 → 考虑 ULID/有序 ID
- **现状**: `Memory.id` 使用 `uuid.uuid4()`，无序
- **优化**: 使用 ULID 或 Snowflake ID，保持时间有序
- **收益**: SQLite B-tree 插入性能提升（顺序写 vs 随机写）
- **复杂度**: 低

### 4.3 缺少结构化日志 / metrics
- **现状**: 仅有 `logging.getLogger` 的 debug 日志
- **优化**: 
  - 添加 OpenTelemetry span（remember/recall/retrieve）
  - 暴露 Prometheus metrics（延迟、吞吐、缓存命中率）
- **收益**: 生产环境可观测性
- **复杂度**: 中

### 4.4 测试中 mock embedder 注入方式不规范
- **位置**: `tests/conftest.py` — `mem._embedder = mock_embedder`
- **现状**: 直接覆盖私有属性
- **优化**: 在 NeuralMem 构造函数中支持 `embedder` 参数注入
- **收益**: 更好的可测试性，符合依赖注入原则
- **复杂度**: 低

---

## 五、文档 & DX (P2 — 低优先级)

### 5.1 缺少 Quick Start 可运行示例
- **现状**: `examples/01_quickstart.py` 存在但未验证可直接运行
- **优化**: 提供 `pip install neuralmem && python -c "..."` 的 30 秒体验
- **收益**: 降低上手门槛
- **复杂度**: 低

### 5.2 缺少 benchmark 基准数据
- **现状**: `benchmarks/` 目录存在但无实际运行结果
- **优化**: 在 LoCoMo / LongMemEval 上跑基准，发布结果
- **收益**: 对标 Zep 的 80.32% LoCoMo 准确率，建立可信度
- **复杂度**: 中

### 5.3 缺少 CONTRIBUTING.md / 开发者指南
- **优化**: 添加贡献指南、开发环境搭建、测试运行说明
- **收益**: 吸引外部贡献者
- **复杂度**: 低

---

## 优先级排序

| 优先级 | 编号 | 优化项 | 预估工期 | 收益 |
|--------|------|--------|----------|------|
| P0 | 1.1 | 批量 embedding 编码 | 1h | 吞吐量 Nx |
| P0 | 1.4 | SQLite 索引 | 30min | 查询速度 |
| P0 | 1.5 | 复用 ThreadPoolExecutor | 30min | 避免线程开销 |
| P0 | 1.3 | 批量 record_access | 30min | 写入性能 |
| P0 | 1.2 | 批量去重查询 | 2h | remember 吞吐 |
| P0 | 2.3 | SQLite WAL + pragma | 30min | 并发读性能 |
| P1 | 1.6 | embedding LRU 缓存 | 1h | 重复查询 |
| P1 | 2.1 | ConsolidationEngine 实现 | 4h | 记忆膨胀 |
| P1 | 3.1 | Ollama embedding | 2h | 用户覆盖 |
| P1 | 3.3 | 异步 API | 4h | 框架兼容 |
| P1 | 2.5 | 实体缓存 | 30min | remember 性能 |
| P1 | 3.4 | 记忆导入 | 2h | 数据迁移 |
| P1 | 3.5 | TTL 过期机制 | 2h | 自动清理 |
| P1 | 4.4 | 测试 DI 注入 | 1h | 代码质量 |
| P2 | 3.2 | PostgreSQL 后端 | 8h | 生产可用 |
| P2 | 2.2 | 图谱增量持久化 | 6h | 大规模性能 |
| P2 | 4.1 | 精确异常处理 | 1h | 调试体验 |
| P2 | 4.2 | ULID 有序 ID | 1h | 插入性能 |
| P2 | 4.3 | 结构化日志/metrics | 4h | 可观测性 |
| P2 | 5.2 | LoCoMo 基准测试 | 8h | 可信度 |
| P2 | 5.1 | Quick Start 示例 | 1h | DX |
| P2 | 5.3 | CONTRIBUTING.md | 1h | 社区 |

---

## 建议执行顺序

**第一批 (1-2 天)** — 性能立竿见影：
`1.1 → 1.4 → 2.3 → 1.5 → 1.3 → 2.5`

**第二批 (3-5 天)** — 功能补全：
`1.2 → 1.6 → 3.1 → 3.4 → 3.5 → 4.4`

**第三批 (1-2 周)** — 架构升级：
`2.1 → 3.3 → 4.1 → 4.2`

**第四批 (2-4 周)** — 生产就绪：
`3.2 → 2.2 → 4.3 → 5.2`
