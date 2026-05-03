# LlamaIndex 集成指南

将 NeuralMem 作为 LlamaIndex 的记忆和向量存储后端。

## 安装

```bash
pip install neuralmem llama-index
```

## 基本用法

```python
from neuralmem.integrations import NeuralMemLlamaIndexMemory
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 创建 NeuralMem 记忆
memory = NeuralMemLlamaIndexMemory(
    db_path="./llamaindex_memory.db"
)

# 加载文档
docs = SimpleDirectoryReader("./data").load_data()

# 创建索引（使用 NeuralMem 作为记忆后端）
index = VectorStoreIndex.from_documents(
    docs,
    memory=memory
)

# 查询（自动利用对话历史）
query_engine = index.as_query_engine()
response = query_engine.query("What did we discuss about AI safety?")
```

## 对话记忆

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine

chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=index.as_retriever(),
    memory=memory,
    verbose=True
)

response = chat_engine.chat("Tell me more about that topic")
```

## 跨会话持久化

```python
# 第一次会话
memory1 = NeuralMemLlamaIndexMemory(db_path="./shared.db", session_id="session_1")
index1 = VectorStoreIndex.from_documents(docs, memory=memory1)

# 第二次会话 — 自动继承之前的记忆
memory2 = NeuralMemLlamaIndexMemory(db_path="./shared.db", session_id="session_2")
index2 = VectorStoreIndex.from_documents([], memory=memory2)  # 空文档，但记忆还在
```
