# NeuralMem 集成指南

NeuralMem 提供与主流 AI 框架的无缝集成。

## 支持的框架

| 框架 | 集成类 | 状态 |
|---|---|---|
| LangChain | `NeuralMemLangChainMemory` | ✅ 可用 |
| LlamaIndex | `NeuralMemLlamaIndexMemory` | ✅ 可用 |
| OpenAI | `NeuralMemOpenAICompat` | ✅ 可用 |
| CrewAI | `CrewAIMemory` | ✅ V0.9+ |
| AutoGen | `AutoGenMemory` | ✅ V0.9+ |
| Semantic Kernel | `SemanticKernelMemory` | ✅ V0.9+ |

## LangChain

```python
from neuralmem.integrations import NeuralMemLangChainMemory
from langchain import OpenAI, LLMChain, PromptTemplate

memory = NeuralMemLangChainMemory()
llm = OpenAI()
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="{history}\nHuman: {input}\nAI:"
)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

## LlamaIndex

```python
from neuralmem.integrations import NeuralMemLlamaIndexMemory
from llama_index.core import VectorStoreIndex

memory = NeuralMemLlamaIndexMemory()
index = VectorStoreIndex.from_documents(docs, memory=memory)
```

## CrewAI

```python
from neuralmem.integrations import CrewAIMemory
from crewai import Agent

memory = CrewAIMemory()
agent = Agent(
    role="Researcher",
    goal="Research AI topics",
    memory=memory
)
```

## AutoGen

```python
from neuralmem.integrations import AutoGenMemory
from autogen import ConversableAgent

memory = AutoGenMemory()
agent = ConversableAgent(
    name="assistant",
    llm_config={...},
    memory=memory
)
```

## Semantic Kernel

```python
from neuralmem.integrations import SemanticKernelMemory
from semantic_kernel import Kernel

memory = SemanticKernelMemory()
kernel = Kernel()
kernel.add_memory(memory)
```
