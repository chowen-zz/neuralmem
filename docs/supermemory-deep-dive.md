# Supermemory.ai 深度技术分析

## 1. 项目概述

- **仓库**: https://github.com/supermemoryai/supermemory
- **定位**: "Build your own second brain with AI"
- **开源程度**: 部分开源（前端、SDK、文档、扩展开源；核心 API 闭源企业版）
- **技术栈**: TypeScript, Cloudflare Workers, Next.js, PostgreSQL+pgvector

## 2. 完整功能清单

### 2.1 核心记忆引擎
- [x] 文档存储（富文本、URL、文件）
- [x] 自动分块（Chunking）
- [x] 多字段嵌入（summaryEmbedding, embedding, memoryEmbedding）
- [x] 混合搜索（RAG + Memory 统一查询）
- [x] 向量相似度搜索（pgvector）
- [x] 阈值可调搜索（chunkThreshold, documentThreshold）
- [x] 可选重排序（rerank）
- [x] 查询重写（query rewriting, +~400ms）
- [x] 事实提取与跟踪
- [x] 矛盾检测与解决
- [x] 自动遗忘（过期信息）
- [x] 记忆版本控制
- [x] 记忆关系图谱（Update/Extend/Derive）

### 2.2 用户画像
- [x] 自动维护静态事实
- [x] 动态近期上下文

### 2.3 连接器生态
- [x] Google Drive（OAuth + Webhook 实时同步）
- [x] Gmail（OAuth + Webhook）
- [x] Notion（OAuth + Webhook）
- [x] OneDrive（OAuth + Webhook）
- [x] GitHub（OAuth + Webhook）
- [x] Web Crawler（内置）
- [x] S3（企业版）
- [x] Chrome/Firefox 浏览器扩展
- [x] Raycast 扩展（macOS）
- [x] MCP Server（Claude/Cursor/Windsurf/VS Code）

### 2.4 文件处理
- [x] PDF 解析
- [x] 图片 OCR
- [x] 视频转录
- [x] 代码 AST-aware 分块

### 2.5 项目管理（Spaces）
- [x] 项目级记忆容器
- [x] 空间成员管理（Owner/Admin/Editor/Viewer）
- [x] 容器标签
- [x] 可见性控制

### 2.6 可视化
- [x] Canvas 图谱可视化（d3-force）
- [x] 记忆卡片网格（Masonry）
- [x] 文档富预览（PDF/图片/Tweet/YouTube/Notion/Google Docs）

### 2.7 AI 助手（Nova）
- [x] 嵌入式 AI 写作助手
- [x] 记忆上下文聊天

### 2.8 分析
- [x] 聊天分析
- [x] 记忆分析
- [x] 使用分析

### 2.9 部署
- [x] Cloudflare Workers（主部署）
- [x] 企业自托管（编译 JS bundle）
- [x] 定时任务（每4小时连接器导入）

## 3. 技术架构

### 3.1 运行时
- Cloudflare Workers（Serverless Edge）
- Next.js 16（前端 App Router）
- Hono（API/MCP 框架）

### 3.2 数据库
- PostgreSQL + pgvector
- Drizzle ORM
- 核心表：Document, Chunk, MemoryEntry, Space, Connection, ApiRequest

### 3.3 AI/LLM
- Cloudflare AI Gateway
- 多提供商：OpenAI, Anthropic, Gemini, Groq

### 3.4 认证
- Better Auth（支持组织）

## 4. API 设计（v3）

| 端点 | 功能 |
|------|------|
| POST /v3/documents | 添加记忆 |
| POST /v3/search | 混合搜索（RAG+Memory） |
| GET /v3/projects | 项目列表 |
| POST /v3/connections/:provider | 创建连接器 |
| GET /v3/analytics/* | 分析数据 |

## 5. NeuralMem 与 Supermemory 功能对标

| 功能 | Supermemory | NeuralMem V1.8 | 差距 |
|------|-------------|----------------|------|
| 文档存储 | ✅ | ✅ | 持平 |
| 自动分块 | ✅ | ✅ | 持平 |
| 向量搜索 | ✅ pgvector | ✅ sqlite-vec | 持平 |
| 混合搜索 | ✅ RAG+Memory | ✅ 4策略+RRF | NeuralMem 更优 |
| 事实提取 | ✅ | ✅ | 持平 |
| 记忆图谱 | ✅ | ✅ NetworkX | 持平 |
| 自动遗忘 | ✅ | ✅ | 持平 |
| 用户画像 | ✅ | ✅ v2 | 持平 |
| 连接器 | 6+ 带Webhook | 10+ 带自动发现 | NeuralMem 更优 |
| 浏览器扩展 | ✅ | ❌ | **差距** |
| Raycast | ✅ | ❌ | **差距** |
| MCP Server | ✅ | ✅ | 持平 |
| 文件处理 | PDF/OCR/视频/代码 | PDF/图片/音频/视频 | 持平 |
| Spaces | ✅ | ❌ | **差距** |
| Canvas 可视化 | ✅ d3-force | ❌ | **差距** |
| AI 写作助手 | ✅ Nova | ❌ | **差距** |
| 分析面板 | ✅ | ✅ Dashboard | 持平 |
| Cloudflare 部署 | ✅ Workers | ❌ | **差距** |
| 企业自托管 | ✅ | ✅ Docker | 持平 |
| 记忆版本控制 | ✅ | ❌ | **差距** |
| 查询重写 | ✅ | ❌ | **差距** |
| 组织/团队 | ✅ | ❌ | **差距** |
| 定时同步 | ✅ Cron | ❌ | **差距** |

## 6. 关键差距（NeuralMem 缺失）

1. **浏览器扩展** — Chrome/Firefox 插件保存网页
2. **Raycast 扩展** — macOS 快速访问
3. **Spaces/项目** — 项目级记忆容器与权限
4. **Canvas 可视化** — 交互式记忆图谱
5. **AI 写作助手** — 嵌入式 Nova 助手
6. **Cloudflare Workers 部署** — Edge serverless
7. **记忆版本控制** — 版本历史与回滚
8. **查询重写** — AI 查询扩展
9. **组织/团队支持** — 多用户组织
10. **定时任务** — Cron 连接器同步
11. **AST-aware 代码分块** — 代码特殊处理
12. **多字段嵌入** — summary/memory/matryoksha

## 7. 演进建议

基于以上差距，建议按以下版本演进：

- **V1.9**: 记忆增强 — 版本控制、查询重写、AST 代码分块
- **V2.0**: 协作空间 — Spaces/项目、组织/团队、权限管理
- **V2.1**: 可视化 — Canvas 图谱、交互式探索
- **V2.2**: AI 助手 — 嵌入式写作助手、查询重写增强
- **V2.3**: 浏览器生态 — Chrome/Firefox 扩展、Raycast
- **V2.4**: Edge 部署 — Cloudflare Workers 适配、定时任务
