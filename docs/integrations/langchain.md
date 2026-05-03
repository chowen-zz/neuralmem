# LangChain 集成指南

将 NeuralMem 作为 LangChain 的持久化记忆后端。

## 安装

```bash
pip install neuralmem langchain
```

## 基本用法

```python
from neuralmem.integrations import NeuralMemLangChainMemory
from langchain import OpenAI, LLMChain, PromptTemplate

# 创建 NeuralMem 记忆
memory = NeuralMemLangChainMemory(
    db_path="./chat_memory.db",
    enable_reranker=True
)

# 配置 LangChain
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""Previous conversation:
{history}

Human: {input}
AI:"""
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# 对话
response = chain.predict(input="What is machine learning?")
response = chain.predict(input="How does it relate to neural networks?")
```

## 多用户支持

```python
memory = NeuralMemLangChainMemory(
    db_path="./multi_user.db",
    user_id="user_123"  # 隔离不同用户的记忆
)
```

## 高级配置

```python
from neuralmem.core.config import NeuralMemConfig

cfg = NeuralMemConfig(
    db_path="./advanced.db",
    enable_llm_extraction=True,      # LLM 驱动的实体提取
    enable_reranker=True,            # 重排序
    enable_importance_reinforcement=True,  # 重要性强化
)
memory = NeuralMemLangChainMemory(config=cfg)
```

## 记忆管理

```python
# 手动添加记忆
memory.save_context(
    {"input": "My favorite color is blue"},
    {"output": "Noted! I'll remember that."}
)

# 检索相关记忆
related = memory.load_memory_variables({"input": "What do I like?"})
```
