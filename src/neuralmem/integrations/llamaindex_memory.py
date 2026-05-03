"""LlamaIndex memory adapter for NeuralMem.

Provides a LlamaIndex-compatible memory interface that wraps NeuralMem,
allowing use with LlamaIndex query engines and chat engines.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralmem.core.memory import NeuralMem


class NeuralMemLlamaIndexMemory:
    """LlamaIndex BaseMemory adapter for NeuralMem.

    Implements the LlamaIndex memory protocol with get/put/get_all/clear
    methods that delegate to NeuralMem's remember/recall/forget API.

    Usage::

        from neuralmem import NeuralMem
        from neuralmem.integrations import NeuralMemLlamaIndexMemory

        mem = NeuralMem()
        memory = NeuralMemLlamaIndexMemory(mem, user_id="alice")
        memory.put("User prefers dark mode")
        context = memory.get()
    """

    def __init__(
        self,
        neural_mem: NeuralMem,
        user_id: str = "default",
    ) -> None:
        self.neural_mem = neural_mem
        self.user_id = user_id

    def get(self) -> str:
        """Return a formatted string of relevant memory context.

        Recalls the most recent memories for the user and formats
        them as a context string suitable for LlamaIndex prompts.
        """
        results = self.neural_mem.recall(
            "recent memories",
            user_id=self.user_id,
            limit=20,
        )

        if not results:
            return ""

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.memory.content}")

        return "\n".join(lines)

    def put(self, message: str) -> None:
        """Store a memory via NeuralMem.remember().

        Args:
            message: The text content to store as a memory.
        """
        self.neural_mem.remember(
            message,
            user_id=self.user_id,
            infer=False,
        )

    def get_all(self) -> list[str]:
        """Return all memory contents for this user.

        Returns:
            List of memory content strings.
        """
        results = self.neural_mem.recall(
            "",
            user_id=self.user_id,
            limit=100,
            min_score=0.0,
        )
        return [r.memory.content for r in results]

    def clear(self) -> None:
        """Clear all memories for this user."""
        self.neural_mem.forget(user_id=self.user_id)
