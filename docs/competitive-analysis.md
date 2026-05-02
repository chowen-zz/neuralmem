# NeuralMem 竞品分析报告

> 对比时间: 2026-05-02 | 对比范围: AI Agent 长期记忆领域 Top 5 竞品
> 
> 数据来源: GitHub 仓库、官方文档、PyPI/npm，调研日期 2026-05-02

---

## 一、竞品概览

| 维度 | NeuralMem | Mem0 | Zep (Graphiti) | Letta (MemGPT) | LangChain Memory | LlamaIndex Memory |
|------|-----------|------|----------------|----------------|------------------|-------------------|
| **GitHub Stars** | 新项目 | 54.6k ⭐ | 25.6k ⭐ | 22.4k ⭐ | 136k ⭐ (框架) | 49.1k ⭐ (框架) |
| **定位** | 本地优先 MCP 原生记忆库 | 通用记忆层 | 时序知识图谱平台 | 有状态 Agent 平台 | Agent 工程平台记忆模块 | 文档 Agent 记忆模块 |
| **最新版本** | v0.2.0 (本地) | v1.0.11 (2026-04-30) | v0.29.0 (2026-04-27) | 持续更新 (2026-04) | 持续更新 (2026-05) | 持续更新 (2026-05) |
| **开源协议** | Apache-2.0 | Apache-2.0 | MIT (Graphiti) | Apache-2.0 | MIT | MIT |
| **语言** | Python | Python / Node.js | Python + Go | Python + TypeScript | Python | Python |
| **MCP 支持** | ✅ 原生 (stdio) | ✅ 云端 MCP | ✅ 本地 MCP Server | ⚠️ 间接支持 | ✅ 内置 | ✅ 内置 |
| **本地优先** | ✅ 完全本地 | ⚠️ 推荐云端 | ⚠️ 需 Neo4j + Docker | ✅ 可自托管 | ✅ | ✅ |

---

## 二、架构对比

### 2.1 NeuralMem
```
User → remember/recall/reflect/forget
  → Extraction (rule-based / LLM: Ollama/OpenAI/Anthropic)
  → Embedding (FastEmbed / OpenAI / Cohere / Gemini / HF / Azure / Ollama)
  → SQLite + sqlite-vec (默认) / PostgreSQL + pgvector (可选)
  → NetworkX 知识图谱 (增量持久化)
  → 4 策略并行检索 (语义 + BM25 + 图遍历 + 时间衰减)
  → RRF 融合 → 可选 Cross-Encoder 重排
  → ConsolidationEngine (相似度合并 + 遗忘曲线)
```

**核心特点**: 单文件 SQLite 或 PostgreSQL, 零外部依赖, MCP Server 原生, 异步 API, 结构化 metrics

### 2.2 Mem0
```
User → add / search / update / delete
  → LLM 提取记忆 (gpt-5-mini, 单次 ADD-only 算法)
  → Entity linking (跨记忆实体关联)
  → Qdrant (默认本地) / pgvector / Pinecone / Milvus / 25+ 向量数据库
  → SQLite 历史存储
  → 混合检索: semantic + BM25 keyword + entity boost
  → Memory Compression Engine (对话压缩)
```

**核心特点**: v3.0 全新记忆算法 (ADD-only), LoCoMo 91.6% 准确率, 25+ 向量数据库, SOC 2/HIPAA 合规

### 2.3 Zep / Graphiti
```
User → Ingest
  → 自动实体/关系提取 (时序感知)
  → 时序知识图谱 (Neo4j)
  → 事实自动失效 + 时序追踪 + Provenance 溯源
  → 增量更新 (无全量重算)
  → 混合搜索: 语义 + 关键词 + 图遍历
  → Graph RAG
```

**核心特点**: 时序知识图谱, 事实自动失效, 自定义边类型, OTEL 可观测性

### 2.4 Letta (MemGPT)
```
User → Agent (self-editing)
  → Core Memory (始终在上下文中)
  → Archival Memory (可搜索存档)
  → Recall Memory (对话历史)
  → MemFS (git-tracked 文件系统记忆)
  → Skills System (学习/预制技能)
  → Sleep-time Compute (后台学习)
```

**核心特点**: 自编辑记忆, MemFS 文件系统记忆, Skills 技能系统, Letta Code 编码助手, AgentFile 便携格式

### 2.5 LangChain Memory
```
User → Deep Agent + Tools
  → Short-term: chat history buffer + 自动上下文压缩
  → Long-term: 持久化跨会话记忆 (新增!)
  → Context Engineering 能力
  → 虚拟文件系统 + subagent 生成
  → Middleware 系统 (预置 + 自定义)
```

**核心特点**: Deep Agents 全栈 Agent, 新增 Long-term Memory 模块, MCP 内置, Model Profiles

### 2.6 LlamaIndex Memory
```
User → FunctionAgent
  → Short-term: FIFO 消息队列 (token 限制)
  → Long-term: Memory Blocks (可自定义)
    - memory.put() / memory.get() API
    - Token-limit 管理
    - chat_history_token_ratio 可配置
  → MCP Server 内置
```

**核心特点**: 新 Memory 类 (替代 ChatMemoryBuffer), BaseMemory 可扩展, MCP 内置, 文档 Agent 定位

---

## 三、功能矩阵对比

| 功能 | NeuralMem | Mem0 | Zep | Letta | LangChain | LlamaIndex |
|------|-----------|------|-----|-------|-----------|------------|
| **自动记忆提取** | ✅ 规则+LLM | ✅ LLM (ADD-only) | ✅ 图提取 | ✅ 自编辑 | ❌ 手动 | ✅ LLM |
| **向量搜索** | ✅ sqlite-vec/pgvector | ✅ 25+ 后端 | ✅ | ✅ | ⚠️ 可选 | ✅ |
| **关键词搜索** | ✅ BM25 | ✅ BM25 (新增) | ✅ | ❌ | ❌ | ❌ |
| **知识图谱** | ✅ NetworkX (增量) | ❌ (OSS 已移除) | ✅ Neo4j 时序图 | ❌ | ❌ | ❌ |
| **时序感知** | ✅ 时间衰减 | ⚠️ 基础 | ✅ 事实失效+溯源 | ❌ | ❌ | ❌ |
| **混合检索** | ✅ 4策略 RRF | ✅ 3策略融合 | ✅ 3策略 | ❌ | ❌ | ⚠️ 基础 |
| **重排序** | ✅ Cross-Encoder | ⚠️ 平台有 | ❌ | ❌ | ❌ | ❌ |
| **冲突检测** | ✅ supersede | ❌ | ✅ 事实失效 | ❌ | ❌ | ❌ |
| **重要性强化** | ✅ recall 自动提升 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **会话记忆** | ✅ 3 层上下文 | ✅ 4 层 | ✅ | ✅ 3 层 | ✅ 短期+长期 | ✅ Memory |
| **批量操作** | ✅ batch CRUD | ✅ bulk | ❌ | ❌ | ❌ | ❌ |
| **导出** | ✅ JSON/MD/CSV | ✅ | ❌ | ✅ AgentFile | ❌ | ❌ |
| **MCP Server** | ✅ 原生 stdio | ✅ 云端 HTTP | ✅ 本地 | ⚠️ 间接 | ✅ 内置 | ✅ 内置 |
| **CLI 工具** | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **异步 API** | ✅ AsyncNeuralMem | ✅ | ❌ | ✅ | ✅ | ✅ |
| **多用户隔离** | ✅ user_id | ✅ | ✅ | ✅ | ⚠️ 手动 | ⚠️ 手动 |
| **解释性** | ✅ explanation | ❌ | ❌ | ❌ | ❌ | ❌ |
| **TTL 过期** | ✅ | ❌ | ✅ 事实失效 | ❌ | ❌ | ❌ |
| **记忆压缩** | ✅ consolidation | ✅ compression | ❌ | ✅ sleep-time | ✅ 自动压缩 | ❌ |
| **Skills/Agent** | ❌ | ❌ | ❌ | ✅ Skills | ✅ Deep Agents | ✅ LlamaAgents |

---

## 四、存储后端对比

| 存储 | NeuralMem | Mem0 | Zep | Letta | LangChain | LlamaIndex |
|------|-----------|------|-----|-------|-----------|------------|
| SQLite | ✅ 默认 | ✅ 历史 | ❌ | ❌ | ✅ | ✅ |
| PostgreSQL | ✅ pgvector (新增) | ✅ pgvector | ❌ | ✅ 默认 | ✅ | ✅ |
| Qdrant | ❌ | ✅ 默认 | ❌ | ❌ | ✅ | ✅ |
| Neo4j | ❌ | ❌ | ✅ 默认 | ❌ | ❌ | ❌ |
| 向量数据库数量 | 2 (sqlite-vec + pgvector) | 25+ | 1 | 1 | 多 | 多 |
| 图数据库 | NetworkX (增量持久化) | ❌ | Neo4j | ❌ | ❌ | ❌ |

---

## 五、Embedding 支持对比

| Provider | NeuralMem | Mem0 | Zep | Letta | LangChain | LlamaIndex |
|----------|-----------|------|-----|-------|-----------|------------|
| 本地模型 | ✅ FastEmbed | ❌ | ❌ | ❌ | ⚠️ 需配置 | ⚠️ 需配置 |
| OpenAI | ✅ | ✅ 默认 | ✅ | ✅ | ✅ | ✅ |
| Azure OpenAI | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Cohere | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Gemini | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| HuggingFace | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Ollama | ✅ (新增) | ✅ | ❌ | ❌ | ✅ | ✅ |

---

## 六、检索策略深度对比

| 策略 | NeuralMem | Mem0 | Zep | Letta | LangChain | LlamaIndex |
|------|-----------|------|-----|-------|-----------|------------|
| 语义搜索 | ✅ sqlite-vec/pgvector | ✅ 向量DB | ✅ | ✅ 归档记忆 | ⚠️ 可选 | ✅ |
| 关键词搜索 | ✅ BM25 | ✅ BM25 (新增) | ✅ | ❌ | ❌ | ❌ |
| 图遍历 | ✅ BFS 邻居 | ❌ | ✅ 实体关系 | ❌ | ❌ | ❌ |
| 时间衰减 | ✅ 重要性×新鲜度 | ❌ | ✅ 时序事实 | ❌ | ❌ | ❌ |
| 融合策略 | ✅ RRF (k=60) | ✅ 多信号融合 | ✅ 混合排序 | ❌ | ❌ | ❌ |
| 重排序 | ✅ Cross-Encoder | ⚠️ 平台 | ❌ | ❌ | ❌ | ❌ |
| **并行检索** | ✅ ThreadPool(4) | ✅ 并行评分 | ❌ | ❌ | ❌ | ❌ |
| 实体增强 | ✅ 图谱遍历 | ✅ entity boost | ✅ 实体关系 | ❌ | ❌ | ❌ |

---

## 七、定价对比

| 层级 | NeuralMem | Mem0 | Zep | Letta |
|------|-----------|------|-----|-------|
| **免费** | ✅ 完全免费开源 | 10K add + 1K retrieval/月 | 1K credits/月 | 基础访问 |
| **入门** | — | $19/月 (50K add) | — | $20/月 (Pro) |
| **专业** | — | $249/月 (500K add) | $125/月 (50K credits) | $100-200/月 |
| **企业** | — | 定制 (on-prem, SSO) | 定制 (SOC 2, HIPAA) | 定制 |

**NeuralMem 优势**: 零成本, 无 API 调用费用 (使用本地模型时), 无月度限制, 无 vendor lock-in

---

## 八、性能基准对比

| 基准 | NeuralMem | Mem0 v3.0 | Zep |
|------|-----------|-----------|-----|
| **LoCoMo 准确率** | 未测 (待基准测试) | 91.6% | 80.32% |
| **LongMemEval** | 未测 | 93.4% | — |
| **BEAM (1M tokens)** | 未测 | 64.1 | — |
| **检索延迟** | <100ms (本地 SQLite) | ~1s (~7K tokens) | <200ms P95 |
| **Token 效率** | 4 策略精准定位 | ~7K tokens/retrieval | — |

> 注: NeuralMem 基准测试框架已就绪 (`benchmarks/run_benchmark.py`)，待正式跑分。

---

## 九、SWOT 分析 — NeuralMem

### Strengths (优势)
1. **真正的本地优先** — 单文件 SQLite 或 PostgreSQL, 零外部依赖, 无需 Docker/云服务
2. **MCP 原生** — stdio 传输, 作为 MCP Server 直接对接 Claude/Cursor 等客户端
3. **混合检索最强** — 4 策略并行 + RRF 融合 + Cross-Encoder 重排, 所有策略本地可用
4. **知识图谱内置** — NetworkX 增量持久化 + 实体关系提取, 轻量级无需 Neo4j
5. **冲突自动解决** — 独有的 supersede 机制 + 重要性自动强化 + TTL 过期
6. **完全免费** — 无 API 限制, 无月费, Apache-2.0 开源
7. **可解释性** — 独有的 retrieval explanation, 其他竞品均无
8. **7 种 Embedding Provider** — 包括本地 FastEmbed + Ollama, 无需 API Key 即可使用
9. **双存储后端** — SQLite (轻量) + PostgreSQL/pgvector (规模化), 满足不同场景
10. **异步原生** — AsyncNeuralMem 支持高并发场景

### Weaknesses (劣势)
1. **社区规模小** — 新项目, 无社区积累 (Mem0 54.6k, LangChain 136k stars)
2. **无云端平台** — Mem0/Zep 提供托管服务, NeuralMem 仅本地
3. **无 Agent 执行能力** — Letta/LangChain 是完整 Agent 平台, NeuralMem 仅记忆层
4. **向量数据库支持少** — 仅 2 种 (sqlite-vec + pgvector), Mem0 支持 25+
5. **无 TypeScript SDK** — Mem0 有 Python + Node.js 双语言支持
6. **无多模态** — Mem0 支持图像/音频记忆
7. **无 SOC 2/HIPAA** — 企业合规方面落后于 Mem0/Zep
8. **基准测试未跑** — 缺乏与竞品的定量对比数据

### Opportunities (机会)
1. **MCP 生态增长** — MCP 协议快速普及, 4/5 竞品已支持 MCP, 原生支持是关键
2. **隐私需求** — 本地优先方案在隐私敏感场景有刚需, Mem0 图谱已移至云端
3. **Mem0 图谱移除** — Mem0 v3.0 从 OSS 移除了图谱功能, NeuralMem 可承接这部分用户
4. **开发者工具集成** — Claude Code / Cursor / Codex 等工具用户快速增长
5. **轻量级替代** — 对不需要云平台的开发者, NeuralMem 比 Mem0/Zep 更简单
6. **教育/研究** — 完整的架构可作为 Agent Memory 教学案例

### Threats (威胁)
1. **Mem0 快速增长** — 54.6k stars, SDK v3.0 大版本, LoCoMo 91.6% 准确率
2. **LangChain/LlamaIndex 内置** — 框架自带记忆模块 + MCP 支持, 减少独立需求
3. **大厂入场** — OpenAI/Google 可能内置记忆能力
4. **Zep 的时序图谱** — Graphiti 的时序知识图谱是独特优势

---

## 十、差异化定位建议

### NeuralMem 的独特价值主张

> "唯一一个同时具备 4 策略混合检索(RRF)、知识图谱、冲突检测、可解释性,
> 且完全本地运行、零依赖、MCP 原生的 Agent 记忆库 — 完全免费, 无 vendor lock-in"

### 与每个竞品的差异化

| vs 竞品 | NeuralMem 的差异化 |
|---------|-------------------|
| **vs Mem0** | 本地优先 vs 云优先; 4策略混合检索 vs 3策略; 图谱免费 vs OSS 已移除; 完全免费 vs $249/月; 但向量DB支持远少于 Mem0 |
| **vs Zep** | NetworkX 轻量图谱 vs Neo4j 重量级; 无需 Docker; 零成本 vs $125/月; 但时序能力不如 Zep |
| **vs Letta** | 纯记忆层 vs 完整 Agent 平台; 混合检索 vs 简单归档; 可组合性更强; 但无 Skills/Sleep-time |
| **vs LangChain** | 开箱即用 vs 需要自行构建; 自动提取 vs 手动; 4策略检索 vs 基础; 但灵活性不如 LangChain |
| **vs LlamaIndex** | 4 策略检索 vs Memory Block; 知识图谱 vs 扁平存储; BM25 内置; 但 LlamaIndex 生态更大 |

### 建议发展方向

1. **短期 (1-2 月)**: 
   - 基准测试跑分 (对标 Mem0 LoCoMo 91.6%)
   - 增加更多向量数据库后端 (Qdrant, Milvus)
   - TypeScript SDK

2. **中期 (3-6 月)**:
   - 时序事实失效 (对标 Zep Graphiti)
   - 多模态记忆 (对标 Mem0)
   - SOC 2 合规准备

3. **长期 (6-12 月)**:
   - 可选云端托管 (Memory-as-a-Service)
   - 分布式图谱 (解决 NetworkX 规模限制)
   - Agent 技能系统 (对标 Letta Skills)

---

## 十一、总结评分

| 维度 | NeuralMem | Mem0 | Zep | Letta | LangChain | LlamaIndex |
|------|-----------|------|-----|-------|-----------|------------|
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **检索质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **图谱能力** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐ |
| **可扩展性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **生态集成** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **成本效益** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **MCP 支持** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **社区活跃** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **隐私安全** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**综合**: NeuralMem 在**检索质量、易用性、成本效益、MCP 支持、隐私安全**上领先, 在**可扩展性、生态、社区**上落后。适合隐私敏感、本地优先、MCP 生态的开发者。
