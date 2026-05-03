# Semantic Kernel 集成指南

将 NeuralMem 作为 Semantic Kernel 的持久化记忆存储。

## 安装

```bash
pip install neuralmem semantic-kernel
```

## 基本用法

```python
from neuralmem.integrations import SemanticKernelMemory
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# 创建 NeuralMem 记忆
memory = SemanticKernelMemory(
    db_path="./sk_memory.db",
    enable_importance_reinforcement=True
)

# 配置 Kernel
kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="gpt-4", ai_model_id="gpt-4"))
kernel.add_memory(memory)

# 保存信息
await memory.save_information(
    collection="general",
    id="fact_1",
    text="The sky is blue because of Rayleigh scattering"
)

# 检索信息
results = await memory.search(
    collection="general",
    query="Why is the sky blue?",
    limit=3
)
```

## 插件记忆

```python
from semantic_kernel.functions import kernel_function

class MemoryPlugin:
    def __init__(self, memory: SemanticKernelMemory):
        self.memory = memory

    @kernel_function(description="Store a fact")
    async def remember(self, text: str, category: str = "general"):
        await self.memory.save_information(
            collection=category,
            id=f"auto_{hash(text)}",
            text=text
        )
        return "Stored!"

    @kernel_function(description="Recall related facts")
    async def recall(self, query: str, category: str = "general"):
        results = await self.memory.search(
            collection=category,
            query=query,
            limit=5
        )
        return "\n".join([r.text for r in results])

# 注册插件
kernel.add_plugin(MemoryPlugin(memory), plugin_name="memory")
```

## 对话历史

```python
# Semantic Kernel 的 ChatHistory 自动使用 NeuralMem
from semantic_kernel.contents import ChatHistory

history = ChatHistory()
history.add_system_message("You have access to long-term memory.")

# 每次对话自动保存到 NeuralMem
# 后续对话自动检索相关历史
```
