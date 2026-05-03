"""LangChain memory adapters for NeuralMem.

Provides LangChain-compatible memory interfaces that wrap NeuralMem,
allowing seamless integration with LangChain chains and agents.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem

# Try importing langchain types; fall back to plain dicts if unavailable
try:
    from langchain_core.messages import AIMessage, HumanMessage

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


class NeuralMemLangChainMemory:
    """LangChain BaseMemory adapter for NeuralMem.

    Implements the LangChain memory interface for use with
    ConversationChain, LLMChain, etc.

    Usage::

        from neuralmem import NeuralMem
        from neuralmem.integrations import NeuralMemLangChainMemory

        mem = NeuralMem()
        memory = NeuralMemLangChainMemory(mem, user_id="alice")
        chain = ConversationChain(llm=llm, memory=memory)
    """

    def __init__(
        self,
        neural_mem: NeuralMem,
        user_id: str = "default",
        k: int = 10,
    ) -> None:
        self.neural_mem = neural_mem
        self.user_id = user_id
        self.k = k

    @property
    def memory_variables(self) -> list[str]:
        """Return the memory variable keys this memory provides."""
        return ["history"]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load recent memory context for the chain.

        Calls ``neural_mem.recall()`` with the most recent input
        (or a generic query) and formats results as a conversation string.
        """
        # Try to use the input as a query for recall
        query = ""
        for key in ("input", "query", "question", "human_input"):
            if key in inputs and inputs[key]:
                query = str(inputs[key])
                break

        if not query:
            query = "recent conversation history"

        results = self.neural_mem.recall(
            query, user_id=self.user_id, limit=self.k
        )

        # Format as conversation history string
        lines: list[str] = []
        for r in results:
            lines.append(r.memory.content)

        return {"history": "\n".join(lines)}

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """Save the input/output pair to NeuralMem.

        Stores both the human input and AI output as separate memories.
        """
        # Extract input
        input_text = ""
        for key in ("input", "query", "question", "human_input"):
            if key in inputs and inputs[key]:
                input_text = str(inputs[key])
                break

        # Extract output
        output_text = ""
        for key in ("output", "response", "answer", "text"):
            if key in outputs and outputs[key]:
                output_text = str(outputs[key])
                break

        if input_text:
            self.neural_mem.remember(
                f"Human: {input_text}",
                user_id=self.user_id,
                tags=["conversation", "human"],
                infer=False,
            )

        if output_text:
            self.neural_mem.remember(
                f"AI: {output_text}",
                user_id=self.user_id,
                tags=["conversation", "ai"],
                infer=False,
            )

    def clear(self) -> None:
        """Clear all memories for this user."""
        self.neural_mem.forget(user_id=self.user_id)


class NeuralMemLangChainChatHistory:
    """LangChain ChatMessageHistory adapter for NeuralMem.

    Wraps NeuralMem to provide a chat message history interface
    compatible with LangChain's ``ChatMessageHistory`` protocol.

    Usage::

        history = NeuralMemLangChainChatHistory(mem, user_id="alice")
        history.add_user_message("Hello!")
        history.add_ai_message("Hi there!")
        print(history.messages)
    """

    def __init__(
        self,
        neural_mem: NeuralMem,
        user_id: str = "default",
    ) -> None:
        self.neural_mem = neural_mem
        self.user_id = user_id

    @property
    def messages(self) -> list[Any]:
        """Load and return messages from NeuralMem.

        Returns LangChain message objects if langchain is installed,
        otherwise returns plain dicts with 'role' and 'content'.
        """
        results = self.neural_mem.recall(
            "", user_id=self.user_id, limit=100, min_score=0.0
        )

        msgs: list[Any] = []
        for r in results:
            content = r.memory.content
            # Detect role from stored prefix
            if content.startswith("Human: "):
                text = content[len("Human: "):]
                if _HAS_LANGCHAIN:
                    msgs.append(HumanMessage(content=text))
                else:
                    msgs.append({"role": "user", "content": text})
            elif content.startswith("AI: "):
                text = content[len("AI: "):]
                if _HAS_LANGCHAIN:
                    msgs.append(AIMessage(content=text))
                else:
                    msgs.append({"role": "assistant", "content": text})
            else:
                if _HAS_LANGCHAIN:
                    msgs.append(HumanMessage(content=content))
                else:
                    msgs.append({"role": "user", "content": content})

        return msgs

    def add_user_message(self, message: str) -> None:
        """Add a user message to the chat history."""
        self.neural_mem.remember(
            f"Human: {message}",
            user_id=self.user_id,
            tags=["chat", "human"],
            infer=False,
        )

    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the chat history."""
        self.neural_mem.remember(
            f"AI: {message}",
            user_id=self.user_id,
            tags=["chat", "ai"],
            infer=False,
        )

    def clear(self) -> None:
        """Clear all chat history for this user."""
        self.neural_mem.forget(user_id=self.user_id)
