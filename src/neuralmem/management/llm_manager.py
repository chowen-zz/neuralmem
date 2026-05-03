"""LLM-driven memory manager — ADD/UPDATE/DELETE/NOOP operations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from neuralmem.management.prompts import DEDUCTION_PROMPT, EXTRACTION_PROMPT

_logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Memory operation types following mem0's pattern."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


@dataclass
class MemoryOperation:
    """Represents a planned memory operation."""

    op_type: OperationType
    content: str = ""
    memory_id: str | None = None
    old_memory_id: str | None = None
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMMemoryManager:
    """Manages memory operations using LLM-driven decision making.

    Uses a two-step process:
    1. Extract memory candidates from conversation
    2. Deduce operations (ADD/UPDATE/DELETE/NOOP) by comparing
       candidates with existing memories.
    """

    def __init__(
        self,
        llm_backend: str = "ollama",
        model: str | None = None,
        **kwargs: object,
    ) -> None:
        self._backend = llm_backend
        self._model = model
        self._kwargs = kwargs
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize LLM client based on backend."""
        if self._backend == "ollama":
            self._model = self._model or "llama3.2:3b"
            self._base_url = self._kwargs.get(
                "base_url", "http://localhost:11434"
            )
        elif self._backend == "openai":
            try:
                import openai

                api_key = self._kwargs.get("api_key")
                self._client = openai.OpenAI(api_key=api_key)
                self._model = self._model or "gpt-4o-mini"
            except ImportError as exc:
                raise ImportError(
                    "Install openai: pip install 'neuralmem[openai]'"
                ) from exc
        elif self._backend == "anthropic":
            try:
                import anthropic

                api_key = self._kwargs.get("api_key")
                self._client = anthropic.Anthropic(api_key=api_key)
                self._model = self._model or "claude-haiku-4-5-20251001"
            except ImportError as exc:
                raise ImportError(
                    "Install anthropic: pip install 'neuralmem[anthropic]'"
                ) from exc
        else:
            raise ValueError(f"Unsupported LLM backend: {self._backend}")

    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM backend and return raw response text."""
        max_retries = 2
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await self._do_call(prompt)
            except Exception as exc:
                last_exc = exc
                _logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
        raise RuntimeError(
            f"LLM call failed after {max_retries + 1} attempts: {last_exc}"
        )

    async def _do_call(self, prompt: str) -> str:
        """Dispatch to the appropriate backend."""
        if self._backend == "ollama":
            return await self._call_ollama(prompt)
        elif self._backend == "openai":
            return await self._call_openai(prompt)
        elif self._backend == "anthropic":
            return await self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        import httpx

        url = f"{self._base_url}/api/generate"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "{}")

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content or "{}"

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        if message.content:
            return message.content[0].text
        return "{}"

    async def _parse_json_response(
        self, text: str
    ) -> list[Any] | dict[str, Any]:
        """Parse JSON from LLM response, handling markdown wrappers."""
        cleaned = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON array or object in the text
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                start = cleaned.find(start_char)
                end = cleaned.rfind(end_char)
                if start != -1 and end > start:
                    try:
                        return json.loads(cleaned[start: end + 1])
                    except json.JSONDecodeError:
                        continue
            _logger.warning(
                "Failed to parse JSON from LLM response: %s",
                text[:200],
            )
            return []

    async def process_conversation(
        self,
        messages: list[dict[str, str]],
        user_id: str | None = None,
        storage: Any = None,
    ) -> list[MemoryOperation]:
        """Process a conversation and return memory operations.

        Steps:
        1. Extract memory candidates from conversation via LLM
        2. Load existing memories for user (if storage provided)
        3. Compare candidates with existing to get operations
        4. Return list of MemoryOperation
        """
        if not messages:
            return []

        # Step 1: Extract memory candidates
        msg_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
        )
        extraction_prompt = EXTRACTION_PROMPT.format(messages=msg_text)
        raw_extraction = await self._call_llm(extraction_prompt)
        candidates = await self._parse_json_response(raw_extraction)

        if not isinstance(candidates, list) or not candidates:
            return []

        # Step 2: Load existing memories
        existing_memories: list[dict[str, str]] = []
        if storage is not None:
            try:
                existing = storage.list_memories(user_id=user_id)
                existing_memories = [
                    {"id": m.id, "content": m.content}
                    for m in existing
                    if m.is_active
                ]
            except Exception as exc:
                _logger.warning("Failed to load existing memories: %s", exc)

        # Step 3: Deduce operations
        candidates_str = json.dumps(
            [c.get("content", str(c)) for c in candidates],
            indent=2,
        )
        existing_str = json.dumps(existing_memories, indent=2)

        deduction_prompt = DEDUCTION_PROMPT.format(
            new_memories=candidates_str,
            existing_memories=existing_str,
        )
        raw_deduction = await self._call_llm(deduction_prompt)
        operations_data = await self._parse_json_response(raw_deduction)

        if not isinstance(operations_data, list):
            return []

        # Step 4: Build MemoryOperation objects
        operations: list[MemoryOperation] = []
        for op in operations_data:
            if not isinstance(op, dict):
                continue
            op_type_str = op.get("op_type", "ADD").upper()
            try:
                op_type = OperationType(op_type_str)
            except ValueError:
                op_type = OperationType.ADD

            confidence = float(op.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            operations.append(
                MemoryOperation(
                    op_type=op_type,
                    content=str(op.get("content", "")),
                    memory_id=op.get("memory_id"),
                    old_memory_id=op.get("old_memory_id"),
                    confidence=confidence,
                    metadata=op.get("metadata", {}),
                )
            )

        return operations
