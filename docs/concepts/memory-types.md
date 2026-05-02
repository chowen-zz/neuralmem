# 记忆类型

NeuralMem 支持四种认知记忆类型，对应人类记忆的不同功能：

## Semantic（语义记忆）
**事实、偏好、知识**——最常用的类型。

```python
mem.remember("用户偏好 Python", memory_type=MemoryType.SEMANTIC)
```

## Episodic（情节记忆）
**事件、交互、历史记录**——带有时间背景的记忆。

```python
mem.remember("昨天用户完成了认证模块重构", memory_type=MemoryType.EPISODIC)
```

## Procedural（程序记忆）
**工作流、SOP、最佳实践**。

```python
mem.remember("部署流程：先跑测试 → 构建镜像 → 推送到 staging", memory_type=MemoryType.PROCEDURAL)
```

## Working（工作记忆）
**当前会话上下文**，会话结束后可设置为低优先级。

```python
mem.remember("当前正在讨论 API 设计方案", memory_type=MemoryType.WORKING)
```

!!! tip "自动推断"
    不指定 `memory_type` 时，NeuralMem 会根据内容自动推断类型。
