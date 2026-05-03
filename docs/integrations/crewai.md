# CrewAI 集成指南

将 NeuralMem 作为 CrewAI Agent 的记忆系统。

## 安装

```bash
pip install neuralmem crewai
```

## 基本用法

```python
from neuralmem.integrations import CrewAIMemory
from crewai import Agent, Task, Crew

# 创建 NeuralMem 记忆
memory = CrewAIMemory(
    db_path="./crew_memory.db",
    enable_llm_extraction=True
)

# 创建 Agent
researcher = Agent(
    role="AI Researcher",
    goal="Research and summarize AI advancements",
    backstory="You are an expert in artificial intelligence.",
    memory=memory,
    verbose=True
)

# 定义任务
task = Task(
    description="Research the latest developments in LLMs",
    agent=researcher,
    expected_output="A summary of recent LLM developments"
)

# 运行 Crew
crew = Crew(
    agents=[researcher],
    tasks=[task],
    memory=True  # 启用 Crew 级记忆
)
result = crew.kickoff()
```

## 多 Agent 共享记忆

```python
# 所有 Agent 共享同一个 NeuralMem 实例
shared_memory = CrewAIMemory(db_path="./shared.db")

researcher = Agent(role="Researcher", memory=shared_memory, ...)
writer = Agent(role="Writer", memory=shared_memory, ...)
editor = Agent(role="Editor", memory=shared_memory, ...)

# 研究员的发现自动对作者和编辑可见
```

## 长期项目记忆

```python
memory = CrewAIMemory(
    db_path="./project_memory.db",
    project_id="ai_research_2026"  # 项目级隔离
)
```
