"""Framework integrations for NeuralMem.

LangChain, LlamaIndex, OpenAI compat, CrewAI, AutoGen,
Semantic Kernel.
"""
from neuralmem.integrations.autogen_memory import (
    AutoGenMemory,
)
from neuralmem.integrations.crewai_memory import CrewAIMemory
from neuralmem.integrations.langchain_memory import (
    NeuralMemLangChainChatHistory,
    NeuralMemLangChainMemory,
)
from neuralmem.integrations.llamaindex_memory import (
    NeuralMemLlamaIndexMemory,
)
from neuralmem.integrations.openai_compat import (
    NeuralMemOpenAICompat,
)
from neuralmem.integrations.semantic_kernel_memory import (
    SemanticKernelMemory,
)

__all__ = [
    "NeuralMemLangChainMemory",
    "NeuralMemLangChainChatHistory",
    "NeuralMemLlamaIndexMemory",
    "NeuralMemOpenAICompat",
    "CrewAIMemory",
    "AutoGenMemory",
    "SemanticKernelMemory",
]
