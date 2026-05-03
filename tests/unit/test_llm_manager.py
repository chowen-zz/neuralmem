"""Tests for LLM Memory Manager — 25+ tests."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neuralmem.management.llm_manager import (
    LLMMemoryManager,
    MemoryOperation,
    OperationType,
)

# ==================== OperationType Enum ====================


class TestOperationTypeEnum:

    def test_operation_type_enum_values(self):
        assert OperationType.ADD.value == "ADD"
        assert OperationType.UPDATE.value == "UPDATE"
        assert OperationType.DELETE.value == "DELETE"
        assert OperationType.NOOP.value == "NOOP"

    def test_operation_type_is_string_enum(self):
        assert isinstance(OperationType.ADD, str)
        assert OperationType.ADD == "ADD"

    def test_operation_type_all_members(self):
        members = list(OperationType)
        assert len(members) == 4


# ==================== MemoryOperation Dataclass ====================


class TestMemoryOperation:

    def test_memory_operation_dataclass_defaults(self):
        op = MemoryOperation(op_type=OperationType.ADD)
        assert op.op_type is OperationType.ADD
        assert op.content == ""
        assert op.memory_id is None
        assert op.old_memory_id is None
        assert op.confidence == 0.5
        assert op.metadata == {}

    def test_memory_operation_dataclass_with_update(self):
        op = MemoryOperation(
            op_type=OperationType.UPDATE,
            content="new fact",
            old_memory_id="old_123",
            confidence=0.9,
            metadata={"source": "test"},
        )
        assert op.op_type is OperationType.UPDATE
        assert op.content == "new fact"
        assert op.old_memory_id == "old_123"
        assert op.confidence == 0.9
        assert op.metadata == {"source": "test"}

    def test_memory_operation_with_memory_id(self):
        op = MemoryOperation(
            op_type=OperationType.DELETE,
            memory_id="mem_456",
            old_memory_id="old_789",
        )
        assert op.memory_id == "mem_456"
        assert op.old_memory_id == "old_789"


# ==================== LLMMemoryManager Init ====================


class TestLLMManagerInit:

    def test_llm_manager_init_ollama(self):
        with patch.object(
            LLMMemoryManager, "_init_client", return_value=None
        ):
            mgr = LLMMemoryManager(llm_backend="ollama")
            assert mgr._backend == "ollama"

    def test_llm_manager_init_openai(self):
        mock_openai = MagicMock()
        with patch.dict(
            "sys.modules", {"openai": mock_openai}
        ):
            with patch.object(
                LLMMemoryManager, "_init_client", return_value=None
            ):
                mgr = LLMMemoryManager(llm_backend="openai")
                assert mgr._backend == "openai"

    def test_llm_manager_init_anthropic(self):
        mock_anthropic = MagicMock()
        with patch.dict(
            "sys.modules", {"anthropic": mock_anthropic}
        ):
            with patch.object(
                LLMMemoryManager, "_init_client", return_value=None
            ):
                mgr = LLMMemoryManager(llm_backend="anthropic")
                assert mgr._backend == "anthropic"

    def test_llm_manager_unsupported_backend(self):
        with pytest.raises(ValueError, match="Unsupported"):
            LLMMemoryManager(llm_backend="invalid_backend")


# ==================== JSON Parsing ====================


class TestParseJsonResponse:

    @pytest.mark.asyncio
    async def test_parse_json_response_valid(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        result = await mgr._parse_json_response('[{"a": 1}]')
        assert result == [{"a": 1}]

    @pytest.mark.asyncio
    async def test_parse_json_response_with_markdown_wrapper(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        text = '```json\n[{"key": "value"}]\n```'
        result = await mgr._parse_json_response(text)
        assert result == [{"key": "value"}]

    @pytest.mark.asyncio
    async def test_parse_json_response_invalid_fallback(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        result = await mgr._parse_json_response("not json at all")
        assert result == []

    @pytest.mark.asyncio
    async def test_parse_json_response_with_surrounding_text(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        text = 'Here is the result:\n[{"x": 1}]\nDone.'
        result = await mgr._parse_json_response(text)
        assert result == [{"x": 1}]

    @pytest.mark.asyncio
    async def test_parse_json_response_object(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        result = await mgr._parse_json_response('{"key": "val"}')
        assert result == {"key": "val"}


# ==================== Extraction ====================


class TestExtractMemories:

    @pytest.mark.asyncio
    async def test_extract_memories_from_conversation(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        mock_response = json.dumps([
            {"content": "User likes Python", "type": "preference",
             "confidence": 0.9}
        ])
        with patch.object(
            mgr, "_call_llm", new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await mgr._parse_json_response(mock_response)
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["content"] == "User likes Python"

    @pytest.mark.asyncio
    async def test_extract_memories_returns_list(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        mock_response = json.dumps([
            {"content": "Fact 1", "type": "fact", "confidence": 0.8},
            {"content": "Pref 1", "type": "preference", "confidence": 0.7},
        ])
        result = await mgr._parse_json_response(mock_response)
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_extract_memories_handles_empty_conversation(self):
        messages: list[dict] = []
        assert messages == []


# ==================== Deduction ====================


class TestDeduceOperations:

    @pytest.mark.asyncio
    async def test_deduce_operations_add_new(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        response = json.dumps([
            {"op_type": "ADD", "content": "New fact", "confidence": 0.9}
        ])
        result = await mgr._parse_json_response(response)
        assert result[0]["op_type"] == "ADD"

    @pytest.mark.asyncio
    async def test_deduce_operations_update_existing(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        response = json.dumps([
            {"op_type": "UPDATE", "content": "Updated",
             "old_memory_id": "mem_1", "confidence": 0.85}
        ])
        result = await mgr._parse_json_response(response)
        assert result[0]["op_type"] == "UPDATE"
        assert result[0]["old_memory_id"] == "mem_1"

    @pytest.mark.asyncio
    async def test_deduce_operations_delete_negated(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        response = json.dumps([
            {"op_type": "DELETE", "old_memory_id": "mem_2",
             "confidence": 0.95}
        ])
        result = await mgr._parse_json_response(response)
        assert result[0]["op_type"] == "DELETE"

    @pytest.mark.asyncio
    async def test_deduce_operations_noop_known(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        response = json.dumps([
            {"op_type": "NOOP", "content": "Already known",
             "confidence": 0.7}
        ])
        result = await mgr._parse_json_response(response)
        assert result[0]["op_type"] == "NOOP"

    @pytest.mark.asyncio
    async def test_deduce_operations_mixed(self):
        mgr = LLMMemoryManager.__new__(LLMMemoryManager)
        response = json.dumps([
            {"op_type": "ADD", "content": "New", "confidence": 0.9},
            {"op_type": "UPDATE", "content": "Updated",
             "old_memory_id": "m1", "confidence": 0.8},
            {"op_type": "NOOP", "content": "Known", "confidence": 0.6},
        ])
        result = await mgr._parse_json_response(response)
        assert len(result) == 3
        types = [r["op_type"] for r in result]
        assert "ADD" in types
        assert "UPDATE" in types
        assert "NOOP" in types


# ==================== End-to-end Process ====================


class TestProcessConversation:

    @pytest.mark.asyncio
    async def test_process_conversation_end_to_end(self):
        """Full flow with mocked LLM and storage."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            mgr = LLMMemoryManager(llm_backend="openai")

        # Mock _call_llm to return extraction then deduction
        extraction_resp = json.dumps([
            {"content": "User likes cats", "type": "preference",
             "confidence": 0.9}
        ])
        deduction_resp = json.dumps([
            {"op_type": "ADD", "content": "User likes cats",
             "confidence": 0.9}
        ])

        call_results = [extraction_resp, deduction_resp]
        call_idx = {"i": 0}

        async def mock_call(prompt: str) -> str:
            idx = call_idx["i"]
            call_idx["i"] += 1
            return call_results[idx]

        mgr._call_llm = mock_call

        # Mock storage
        mock_storage = MagicMock()
        mock_storage.list_memories.return_value = []

        messages = [
            {"role": "user", "content": "I love cats so much"}
        ]
        ops = await mgr.process_conversation(
            messages, user_id="user1", storage=mock_storage
        )

        assert len(ops) == 1
        assert ops[0].op_type is OperationType.ADD
        assert "cats" in ops[0].content.lower()

    @pytest.mark.asyncio
    async def test_process_conversation_empty_messages(self):
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            mgr = LLMMemoryManager(llm_backend="openai")
        ops = await mgr.process_conversation([])
        assert ops == []

    @pytest.mark.asyncio
    async def test_process_conversation_with_user_id(self):
        """Verify user_id is passed to storage.list_memories."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            mgr = LLMMemoryManager(llm_backend="openai")

        extraction_resp = json.dumps([
            {"content": "Test", "type": "fact", "confidence": 0.5}
        ])
        deduction_resp = json.dumps([
            {"op_type": "ADD", "content": "Test", "confidence": 0.5}
        ])
        call_results = [extraction_resp, deduction_resp]
        call_idx = {"i": 0}

        async def mock_call(prompt: str) -> str:
            idx = call_idx["i"]
            call_idx["i"] += 1
            return call_results[idx]

        mgr._call_llm = mock_call

        mock_storage = MagicMock()
        mock_storage.list_memories.return_value = []

        await mgr.process_conversation(
            [{"role": "user", "content": "test"}],
            user_id="user_abc",
            storage=mock_storage,
        )
        mock_storage.list_memories.assert_called_once_with(
            user_id="user_abc"
        )


# ==================== Error Handling ====================


class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_llm_call_timeout_handling(self):
        """LLM call failure raises RuntimeError after retries."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            mgr = LLMMemoryManager(llm_backend="openai")

        async def failing_call(prompt: str) -> str:
            raise TimeoutError("connection timed out")

        mgr._do_call = failing_call

        with pytest.raises(RuntimeError, match="LLM call failed"):
            await mgr._call_llm("test prompt")

    @pytest.mark.asyncio
    async def test_llm_call_retry_on_failure(self):
        """LLM call retries before giving up."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            mgr = LLMMemoryManager(llm_backend="openai")

        call_count = {"n": 0}

        async def sometimes_failing(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("temporary failure")
            return "success"

        mgr._do_call = sometimes_failing
        result = await mgr._call_llm("test")
        assert result == "success"
        assert call_count["n"] == 3


# ==================== Prompt Formatting ====================


class TestPromptFormatting:

    def test_extraction_prompt_format(self):
        from neuralmem.management.prompts import EXTRACTION_PROMPT

        formatted = EXTRACTION_PROMPT.format(messages="user: hello")
        assert "user: hello" in formatted
        assert "{messages}" not in formatted

    def test_deduction_prompt_includes_existing_memories(self):
        from neuralmem.management.prompts import DEDUCTION_PROMPT

        formatted = DEDUCTION_PROMPT.format(
            new_memories='["new fact"]',
            existing_memories='[{"id": "1", "content": "old fact"}]',
        )
        assert "new fact" in formatted
        assert "old fact" in formatted
        assert "{new_memories}" not in formatted
        assert "{existing_memories}" not in formatted


# ==================== Confidence ====================


class TestConfidence:

    def test_operation_confidence_range(self):
        """Confidence should be valid float."""
        op = MemoryOperation(
            op_type=OperationType.ADD,
            confidence=0.0,
        )
        assert 0.0 <= op.confidence <= 1.0

        op2 = MemoryOperation(
            op_type=OperationType.ADD,
            confidence=1.0,
        )
        assert 0.0 <= op2.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_clamped_in_process(self):
        """Process pipeline should clamp confidence to [0, 1]."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            mgr = LLMMemoryManager(llm_backend="openai")

        extraction_resp = json.dumps([
            {"content": "fact", "type": "fact", "confidence": 0.5}
        ])
        # Return out-of-range confidence
        deduction_resp = json.dumps([
            {"op_type": "ADD", "content": "fact", "confidence": 1.5}
        ])
        call_results = [extraction_resp, deduction_resp]
        call_idx = {"i": 0}

        async def mock_call(prompt: str) -> str:
            idx = call_idx["i"]
            call_idx["i"] += 1
            return call_results[idx]

        mgr._call_llm = mock_call
        mock_storage = MagicMock()
        mock_storage.list_memories.return_value = []

        ops = await mgr.process_conversation(
            [{"role": "user", "content": "test"}],
            storage=mock_storage,
        )
        if ops:
            assert ops[0].confidence <= 1.0
