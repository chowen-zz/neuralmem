# Deep Interview Spec: NeuralMem — Agent Memory System

## Metadata
- Interview ID: neuralmem-2026-04-29
- Rounds: 3
- Final Ambiguity Score: 13%
- Type: greenfield
- Generated: 2026-04-29
- Threshold: 20%
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.90 | 40% | 0.360 |
| Constraint Clarity | 0.82 | 30% | 0.246 |
| Success Criteria | 0.88 | 30% | 0.264 |
| **Total Clarity** | | | **0.870** |
| **Ambiguity** | | | **13%** |

## Goal
实现 NeuralMem —— 一个本地优先、MCP 原生的 Agent 长期记忆系统。覆盖 Week 1-8 范围：核心引擎（SQLite + FastEmbed）+ 四策略检索引擎（语义/BM25/图谱/时序 + RRF 融合）+ 知识图谱（NetworkX）+ MCP Server（5 Tool + 1 Resource）+ CLI。Week 9-10 模块（DecayManager、CrossEncoderReranker、entity_resolver）实现为功能正确的最小骨架，确保主类可运行。

## Constraints
- 默认零外部 API 依赖（无需 OpenAI Key、无需 Docker）
- Python 3.10+，使用 `uv` 作为包管理器
- 核心存储：SQLite + sqlite-vec（嵌入式向量扩展）
- 本地 Embedding：FastEmbed（ONNX，all-MiniLM-L6-v2）
- 内存图谱：NetworkX（零依赖）
- LLM 提取：规则+spaCy 为默认路径，Ollama 为可选增强（`llm_extractor.py` 实现 Ollama 接口）
- Week 9-10 模块（lifecycle/decay, retrieval/reranker, extraction/entity_resolver）实现为骨架：接口完整，逻辑占位（return 0 / return placeholder），主类可正常实例化和运行

## Non-Goals
- REST API Server 模式（Week 11-12 / Phase 2）
- Web Dashboard（Phase 2）
- TypeScript/Go SDK（Phase 2）
- PostgreSQL / Redis 后端（企业版）
- Neo4j 图谱后端（企业版）
- SSO/RBAC/审计日志（企业版）
- LongMemEval 基准测试对接
- 数据导入/导出脚本
- Cross-Encoder 重排序完整实现（Week 9-10，当前为骨架）
- 遗忘曲线完整实现（Week 9-10，当前为骨架）

## Acceptance Criteria
- [ ] `pip install neuralmem`（或 `uv add neuralmem`）可成功安装
- [ ] 3行代码示例可运行：`NeuralMem()` → `remember()` → `recall()`
- [ ] SQLite 数据库自动创建于 `~/.neuralmem/memory.db`
- [ ] FastEmbed 本地 Embedding 无需任何 API Key 运行
- [ ] 四策略检索引擎（semantic + keyword + graph + temporal）并行执行并 RRF 融合
- [ ] 知识图谱实体和关系可存储和遍历
- [ ] MCP Server 可通过 `neuralmem mcp` 启动（stdio 传输）
- [ ] MCP 5个工具（remember/recall/reflect/forget/consolidate）功能正确
- [ ] CLI 支持 `neuralmem mcp` 和 `neuralmem serve` 命令
- [ ] 测试覆盖率 ≥ 80%（单元测试 + 集成测试）
- [ ] Week 9-10 骨架模块（DecayManager/CrossEncoderReranker）不影响主类实例化

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| 一次实现全部 12 周 | 实际交付边界？ | 实现 Week 1-8，Week 9-10 为骨架 |
| LLM 提取需要 API Key | 如何处理外部依赖？ | 规则提取为默认，Ollama 为可选 |
| Week 9-10 模块可以不存在 | 主类导入 DecayManager 会崩溃 | 实现最小骨架确保主类可运行 |

## Technical Context
全新项目（greenfield）。当前目录只有设计文档：
- `NeuralMem_落地执行方案.md` — 完整代码骨架 + 12周Sprint计划
- `Agent_Memory_深度调研与完整方案.md` — 市场调研 + 技术架构 + 商业模式

参考文档已提供：
- 完整目录结构（`src/neuralmem/` 为 Python 包）
- 所有核心类的代码骨架
- `pyproject.toml` 依赖配置
- Docker/docker-compose 配置

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| NeuralMem | core domain | config, storage, embedding, extractor, graph, retrieval, decay | 组合所有子系统 |
| SQLiteStorage | storage | db_path, conn | 被 NeuralMem 使用 |
| LocalEmbedding | embedding | model_name, model | 被 RetrievalEngine 使用 |
| MemoryExtractor | extraction | config | 被 NeuralMem.remember() 调用 |
| LLMExtractor(Ollama) | extraction | ollama_url, client | 可选增强 MemoryExtractor |
| RetrievalEngine | retrieval | storage, embedding, graph, config | 被 NeuralMem.recall() 调用 |
| KnowledgeGraph | graph | storage, nx_graph | 被 RetrievalEngine._graph_search() 使用 |
| MCPServer | mcp | FastMCP instance, NeuralMem engine | 包含 5 Tools + 1 Resource |
| DecayManager | lifecycle (stub) | storage, config | 被 NeuralMem.consolidate() 调用 |
| CrossEncoderReranker | retrieval (stub) | model | 被 RetrievalEngine._rerank() 调用 |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 7 | 7 | - | - | N/A |
| 2 | 8 | 1 (LLMExtractor) | 0 | 7 | 87% |
| 3 | 10 | 2 (DecayManager, CrossEncoderReranker) | 0 | 8 | 80% |

## Interview Transcript
<details>
<summary>Full Q&A (3 rounds)</summary>

### Round 1
**Q:** 你期望这次实现的范围是什么？
**A:** 全量实现（Week 1-8）：核心引擎 + 四策略检索引擎 + 知识图谱 + MCP Server + CLI + 完整测试套件
**Ambiguity:** 46% → 29% (Goal: 0.72→0.85, Constraints: 0.55, Criteria: 0.38→0.72)

### Round 2
**Q:** 对于记忆提取模块，你希望如何处理 LLM 依赖？
**A:** 规则提取 + Ollama 可选（推荐）：默认规则路径，llm_extractor.py 实现 Ollama 接口
**Ambiguity:** 29% → 22% (Goal: 0.85→0.88, Constraints: 0.50→0.76, Criteria: 0.72→0.78)

### Round 3
**Q:** 对于 Week 9-10 的模块（DecayManager/Consolidation/Reranker），应该如何处理？
**A:** 包含骨架实现（推荐）：接口完整，逻辑占位，主类可正常运行
**Ambiguity:** 22% → 13% (Goal: 0.90, Constraints: 0.82, Criteria: 0.88) ✅ PASSED

</details>
