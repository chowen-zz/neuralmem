# NeuralMem LlamaIndex Integration Design

**Date:** 2026-04-30
**Status:** Approved
**Scope:** `integrations/llamaindex/` — `NeuralMemRetriever` + `NeuralMemChatMemory`

---

## 1. Problem Statement

NeuralMem v0.2.0 provides a powerful multi-provider memory engine, but teams using LlamaIndex cannot plug it into RAG pipelines or conversation agents without writing custom glue code. There is no `BaseRetriever` or `BaseMemory` adapter, so NeuralMem cannot participate in LlamaIndex query engines or agent memory systems.

---

## 2. Goals

- Implement `NeuralMemRetriever(BaseRetriever)` for LlamaIndex RAG query engines
- Implement `NeuralMemChatMemory(BaseMemory)` for LlamaIndex conversation agents
- Package as `neuralmem-llamaindex` (wheel), published from the same monorepo alongside `neuralmem-langchain`
- Zero changes to the `neuralmem` core package
- `llama-index-core>=0.10` only — no compatibility shim for older versions
- ≥80% test coverage, no real network calls in CI

## 3. Non-Goals

- LlamaIndex `VectorStore` protocol implementation — NeuralMem has its own storage layer
- Compatibility with `llama-index<0.10` (pre-split unified package)
- Streaming retrieval
- CrewAI, AutoGen adapters — separate sub-projects

---

## 4. Architecture

### 4.1 Directory Structure

```
NeuralMem/
├── integrations/
│   ├── langchain/     (existing)
│   └── llamaindex/
│       ├── pyproject.toml
│       ├── neuralmem_llamaindex/
│       │   ├── __init__.py
│       │   ├── retriever.py      # NeuralMemRetriever + _to_node_with_score
│       │   └── chat_memory.py    # NeuralMemChatMemory + helpers
│       └── tests/
│           ├── conftest.py
│           ├── test_retriever.py
│           └── test_chat_memory.py
└── src/neuralmem/    # unchanged
```

### 4.2 Dependency Graph

```
neuralmem-llamaindex (0.2.0)
  ├── neuralmem >= 0.2.0
  └── llama-index-core >= 0.10.0   # base classes only
```

### 4.3 NeuralMemRetriever

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

class NeuralMemRetriever(BaseRetriever):
    """
    LlamaIndex BaseRetriever backed by NeuralMem.

    Usage:
        mem = NeuralMem()
        retriever = NeuralMemRetriever(mem=mem, user_id="alice", k=5)
        nodes = retriever.retrieve("user preferences")
    """
    mem: Any
    user_id: str = "default"
    k: int = 5
    min_score: float = 0.3

    model_config = {"arbitrary_types_allowed": True}

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        results = self.mem.recall(
            query_bundle.query_str,
            user_id=self.user_id,
            limit=self.k,
            min_score=self.min_score,
        )
        return [_to_node_with_score(r) for r in results]


def _to_node_with_score(result: SearchResult) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=result.memory.content,
            metadata={
                "memory_id": result.memory.id,
                "memory_type": result.memory.memory_type.value,
                "tags": list(result.memory.tags),
                "created_at": result.memory.created_at.isoformat(),
                "user_id": result.memory.user_id,
            },
        ),
        score=result.score,
    )
```

### 4.4 NeuralMemChatMemory

```python
from llama_index.core.memory import BaseMemory
from llama_index.core.llms import ChatMessage, MessageRole

class NeuralMemChatMemory(BaseMemory):
    """
    LlamaIndex BaseMemory backed by NeuralMem.
    Stores chat messages as EPISODIC memories.

    Usage:
        mem = NeuralMem()
        memory = NeuralMemChatMemory(mem=mem, user_id="alice")
        # In an agent:
        agent = ReActAgent.from_tools([...], memory=memory)
    """
    mem: Any
    user_id: str = "default"
    window_size: int = 10

    model_config = {"arbitrary_types_allowed": True}

    def put(self, message: ChatMessage) -> None:
        self.mem.remember(
            message.content,
            user_id=self.user_id,
            memory_type=MemoryType.EPISODIC,
            tags=[message.role.value],
        )

    def get(self, input: str | None = None, **kwargs: Any) -> list[ChatMessage]:
        if input:
            # Semantic retrieval — uses NeuralMem's 4-strategy search
            results = self.mem.recall(input, user_id=self.user_id, limit=self.window_size)
            return [_result_to_chat_message(r) for r in results]
        # Fallback: return most recent N memories in chronological order
        memories = self.mem.storage.list_memories(user_id=self.user_id, limit=self.window_size)
        return [_memory_to_chat_message(m) for m in sorted(memories, key=lambda m: m.created_at)]

    def get_all(self) -> list[ChatMessage]:
        memories = self.mem.storage.list_memories(user_id=self.user_id, limit=10_000)
        return [_memory_to_chat_message(m) for m in sorted(memories, key=lambda m: m.created_at)]

    def reset(self) -> None:
        self.mem.forget(user_id=self.user_id)

    def set(self, messages: list[ChatMessage]) -> None:
        self.reset()
        for msg in messages:
            self.put(msg)
```

**Helper functions:**

```python
def _result_to_chat_message(result: SearchResult) -> ChatMessage:
    """Convert a recall() result to a ChatMessage.
    Role is recovered from tags; defaults to 'user' if not tagged."""
    tags = list(result.memory.tags)
    role = MessageRole.USER
    for r in (MessageRole.ASSISTANT, MessageRole.USER, MessageRole.SYSTEM):
        if r.value in tags:
            role = r
            break
    return ChatMessage(role=role, content=result.memory.content)


def _memory_to_chat_message(memory: Memory) -> ChatMessage:
    """Convert a Memory to a ChatMessage for chronological fallback path."""
    tags = list(memory.tags)
    role = MessageRole.USER
    for r in (MessageRole.ASSISTANT, MessageRole.USER, MessageRole.SYSTEM):
        if r.value in tags:
            role = r
            break
    return ChatMessage(role=role, content=memory.content)
```

### 4.5 Data Flow: NeuralMemRetriever

```
LlamaIndex query_engine.query("user preferences")
  → NeuralMemRetriever._retrieve(QueryBundle(query_str="user preferences"))
  → mem.recall(query_str, user_id, limit=k, min_score)
  → [SearchResult(memory, score)] 
  → [NodeWithScore(TextNode(text=content, metadata={...}), score)]
  → returned to query engine for synthesis
```

### 4.6 Data Flow: NeuralMemChatMemory (semantic path)

```
LlamaIndex agent processes user message "What did I say earlier about Python?"
  → memory.get(input="What did I say earlier about Python?")
  → mem.recall(input, user_id, limit=window_size)
  → [SearchResult] → [ChatMessage(role=user/assistant, content=...)]
  → injected into agent's context window
```

---

## 5. Package Configuration

### `integrations/llamaindex/pyproject.toml`

```toml
[project]
name = "neuralmem-llamaindex"
version = "0.2.0"
description = "LlamaIndex integration for NeuralMem — BaseRetriever and BaseMemory adapters"
requires-python = ">=3.10"
license = "Apache-2.0"
authors = [{ name = "NeuralMem Team" }]
dependencies = [
    "neuralmem>=0.2.0",
    "llama-index-core>=0.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["neuralmem_llamaindex"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--tb=short -q"
```

---

## 6. Testing Strategy

No real network calls. Both `NeuralMem` and `NeuralMem.storage` are mocked.

### Test Cases

**`test_retriever.py` (6 tests)**

| Test | Verifies |
|------|---------|
| `test_returns_nodes` | `recall()` result → `NodeWithScore` list |
| `test_node_text_is_memory_content` | `node.text == memory.content` |
| `test_node_metadata_fields` | all metadata keys present and correct |
| `test_empty_results` | `recall()` returns `[]` → retriever returns `[]` |
| `test_k_forwarded` | `k=3` → `recall(limit=3)` |
| `test_user_id_forwarded` | `user_id="alice"` → `recall(user_id="alice")` |

**`test_chat_memory.py` (8 tests)**

| Test | Verifies |
|------|---------|
| `test_put_calls_remember` | `put(ChatMessage)` → `mem.remember(content, ...)` |
| `test_put_tags_role` | role stored in tags (`"user"` / `"assistant"`) |
| `test_get_with_input_calls_recall` | `get(input="query")` → `mem.recall("query", ...)` |
| `test_get_without_input_uses_storage` | `get(input=None)` → `storage.list_memories(...)` |
| `test_get_returns_chat_messages` | results mapped to `ChatMessage` with correct role |
| `test_reset_calls_forget` | `reset()` → `mem.forget(user_id=...)` |
| `test_set_resets_then_puts` | `set([msg1, msg2])` → `reset()` + 2× `put()` |
| `test_get_all_returns_all_messages` | `get_all()` → `storage.list_memories(limit=10_000)` |

---

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| `recall()` raises `NeuralMemError` | Propagates — no swallowing |
| `remember()` raises `NeuralMemError` | Propagates from `put()` |
| `forget()` raises `NeuralMemError` | Propagates from `reset()` |
| `message.content` is empty string | Stored as-is — caller's responsibility |

---

## 8. Delivery Order

1. Directory structure + `pyproject.toml` + install
2. `conftest.py` fixtures
3. `retriever.py` + `test_retriever.py` (TDD)
4. `chat_memory.py` + `test_chat_memory.py` (TDD)
5. `__init__.py` exports + final verification

---

## 9. Open Questions

- None. All decisions resolved during brainstorming.
