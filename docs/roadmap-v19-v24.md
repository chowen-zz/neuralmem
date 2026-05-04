# NeuralMem 演进路线图 V1.9-V2.4

## 目标
对标 Supermemory 所有功能，自评测、自迭代，直到全面覆盖并超越。

## 当前状态（V1.8）
- NeuralMem 评分: 8.50
- Supermemory 评分: 8.20
- 领先: +0.30

## 差距分析（Supermemory 有，NeuralMem 无）

| # | 功能 | 优先级 | 版本 |
|---|------|--------|------|
| 1 | 记忆版本控制 | 高 | V1.9 |
| 2 | 查询重写（Query Rewriting） | 高 | V1.9 |
| 3 | AST-aware 代码分块 | 中 | V1.9 |
| 4 | Spaces/项目级记忆容器 | 高 | V2.0 |
| 5 | 组织/团队与权限 | 高 | V2.0 |
| 6 | Canvas 交互式图谱可视化 | 中 | V2.1 |
| 7 | AI 写作助手（Nova-like） | 中 | V2.2 |
| 8 | Chrome/Firefox 浏览器扩展 | 中 | V2.3 |
| 9 | Raycast 扩展 | 低 | V2.3 |
| 10 | Cloudflare Workers 部署 | 低 | V2.4 |
| 11 | 定时任务/Cron | 中 | V2.4 |
| 12 | 多字段嵌入策略 | 中 | V1.9 |

## 版本规划

### V1.9 "记忆增强引擎"
**目标**: 填补核心记忆能力差距
**预期评分**: 8.50 → 8.65

新增模块:
- `src/neuralmem/versioning/` — 记忆版本控制
- `src/neuralmem/retrieval/query_rewrite.py` — 查询重写引擎
- `src/neuralmem/extraction/code_chunker.py` — AST-aware 代码分块
- `src/neuralmem/embedding/multi_field.py` — 多字段嵌入策略

### V2.0 "协作空间"
**目标**: 团队级记忆协作
**预期评分**: 8.65 → 8.80

新增模块:
- `src/neuralmem/spaces/` — 项目/空间管理
- `src/neuralmem/organization/` — 组织与团队
- `src/neuralmem/access/` — RBAC 权限系统

### V2.1 "可视化大脑"
**目标**: 交互式记忆图谱
**预期评分**: 8.80 → 8.90

新增模块:
- `src/neuralmem/visualization/canvas.py` — Canvas 图谱引擎
- `src/neuralmem/visualization/renderer.py` — 前端渲染适配

### V2.2 "AI 创作伙伴"
**目标**: 嵌入式 AI 写作助手
**预期评分**: 8.90 → 8.95

新增模块:
- `src/neuralmem/assistant/` — AI 写作助手核心
- `src/neuralmem/assistant/context.py` — 记忆上下文注入

### V2.3 "浏览器生态"
**目标**: 全平台捕获记忆
**预期评分**: 8.95 → 8.98

新增模块:
- `extensions/chrome/` — Chrome 扩展
- `extensions/firefox/` — Firefox 扩展
- `extensions/raycast/` — Raycast 扩展

### V2.4 "Edge 无服务器"
**目标**: Cloudflare Workers 适配
**预期评分**: 8.98 → 9.00+

新增模块:
- `src/neuralmem/edge/` — Workers 运行时适配
- `src/neuralmem/cron/` — 定时任务调度

## 自评测流程（每版本）

1. 实现功能模块 + 测试
2. 运行全部单元测试（无回归）
3. 运行压力测试
4. 自评测打分（10维度）
5. 与 Supermemory 对比
6. 识别剩余差距
7. 规划下一版本

## 最终目标

| 阶段 | 版本 | 评分目标 | Supermemory |
|------|------|----------|-------------|
| 当前 | V1.8 | 8.50 | 8.20 |
| 近期 | V1.9 | 8.65 | 8.20 |
| 中期 | V2.0-V2.2 | 8.95 | 8.20 |
| 远期 | V2.3-V2.4 | 9.00+ | 8.20 |
