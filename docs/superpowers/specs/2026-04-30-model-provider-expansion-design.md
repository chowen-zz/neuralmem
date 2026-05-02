# NeuralMem — Model Provider Expansion Design (Phase 1)

**Date:** 2026-04-30  
**Status:** Approved  
**Scope:** Core package only (`src/neuralmem/`); framework integrations deferred to Phase 2

---

## 1. Problem Statement

NeuralMem v0.1 supports only three embedding providers (`local`, `openai`, `ollama`) and one LLM extractor (`ollama`). Users on Cohere, Google Gemini, HuggingFace Inference API, Azure OpenAI, or Anthropic Claude must either use the local FastEmbed model or run Ollama locally. This limits adoption for teams already invested in other provider ecosystems.

---

## 2. Goals

- Add 4 new embedding backends: Cohere, Gemini, HuggingFace Inference API, Azure OpenAI
- Add 2 new LLM extractors: OpenAI (gpt-4o-mini), Anthropic (claude-haiku-4-5-20251001)
- Introduce a registry pattern so new providers can be added without touching `NeuralMem.__init__()`
- Keep all new dependencies strictly opt-in via PyPI extras
- Achieve ≥80% unit test coverage on all new modules using mocks (no real API calls in CI)

## 3. Non-Goals (Phase 2)

- Framework integrations: LangChain, LlamaIndex, CrewAI, AutoGen
- `neuralmem-integrations` separate package
- HuggingFace local model support (separate from Inference API)
- Alternative vector stores (Chroma, Qdrant, pgvector)

---

## 4. Architecture

### 4.1 Embedding Registry

New file `src/neuralmem/embedding/registry.py` acts as the single factory for all embedding backends:

```python
def get_embedder(config: NeuralMemConfig) -> EmbeddingBackend:
    match config.embedding_provider:
        case "cohere":       return CohereEmbedding(config)
        case "gemini":       return GeminiEmbedding(config)
        case "huggingface":  return HuggingFaceEmbedding(config)
        case "azure_openai": return AzureOpenAIEmbedding(config)
        case "openai":       return OpenAIEmbedding(config)
        case _:              return LocalEmbedding(config)  # default
```

`NeuralMem.__init__()` changes from `self.embedding = LocalEmbedding(self.config)` to `self.embedding = get_embedder(self.config)`. No other change to the core engine.

### 4.2 Extractor Registry

New file `src/neuralmem/extraction/extractor_registry.py`:

```python
def get_extractor(config: NeuralMemConfig) -> MemoryExtractor | LLMExtractor:
    match config.llm_extractor:
        case "openai":    return OpenAIExtractor(config)
        case "anthropic": return AnthropicExtractor(config)
        case "ollama":    return LLMExtractor(config)   # existing
        case _:           return MemoryExtractor(config)  # rule-based default
# Note: "gemini" extractor deferred to Phase 2
```

`NeuralMem.__init__()` replaces the current `if config.enable_llm_extraction` branch with `get_extractor(self.config)`.

### 4.3 New Embedding Backends

All inherit `EmbeddingBackend` (ABC in `embedding/base.py`). Lazy import of the SDK — `ImportError` is caught at instantiation time with a clear message pointing to the correct `pip install` extra.

| File | Provider | SDK | Default model | Dimension |
|------|----------|-----|---------------|-----------|
| `embedding/cohere.py` | Cohere | `cohere>=5.0` | `embed-multilingual-v3.0` | 1024 |
| `embedding/gemini.py` | Google Gemini | `google-generativeai>=0.8` | `text-embedding-004` | 768 |
| `embedding/huggingface.py` | HF Inference API | `httpx` (already a dep) | `BAAI/bge-m3` | 1024 |
| `embedding/azure_openai.py` | Azure OpenAI | `openai>=1.0` (existing extra) | `text-embedding-3-small` | 1536 |

Each backend encodes the dimension as a class-level constant derived from the configured model name, with a fallback default. On first `encode()` call, if dimension does not match the sqlite-vec column, a clear `ConfigurationError` is raised.

### 4.4 New LLM Extractors

Both inherit the same base structure as the existing `LLMExtractor` (Ollama). They share:
- The same `_EXTRACT_PROMPT` template
- The same JSON-parsing + markdown-stripping logic
- The same fallback to `MemoryExtractor` on any exception

| File | Provider | SDK | Default model |
|------|----------|-----|---------------|
| `extraction/openai_extractor.py` | OpenAI | `openai>=1.0` | `gpt-4o-mini` |
| `extraction/anthropic_extractor.py` | Anthropic | `anthropic>=0.30` | `claude-haiku-4-5-20251001` |

Each extractor exposes a single overridden method `_call_llm(prompt: str) -> str` that returns the raw LLM response string. Parsing is handled in the shared base.

---

## 5. Configuration Changes

`NeuralMemConfig` new fields (all optional, default `None` or sensible default):

```python
# Embedding provider selection
embedding_provider: str = "local"  # existing, now routes through registry

# Cohere
cohere_api_key: str | None = None
cohere_embedding_model: str = "embed-multilingual-v3.0"

# Gemini
gemini_api_key: str | None = None
gemini_embedding_model: str = "text-embedding-004"

# HuggingFace Inference API
hf_api_key: str | None = None
hf_model: str = "BAAI/bge-m3"
hf_inference_url: str = "https://api-inference.huggingface.co"

# Azure OpenAI
azure_endpoint: str | None = None
azure_api_key: str | None = None
azure_deployment: str = "text-embedding-3-small"
azure_api_version: str = "2024-02-01"

# LLM Extractor selection (replaces enable_llm_extraction bool)
llm_extractor: str = "none"  # none | ollama | openai | anthropic | gemini

# OpenAI extractor
openai_extractor_model: str = "gpt-4o-mini"

# Anthropic extractor
anthropic_api_key: str | None = None
anthropic_model: str = "claude-haiku-4-5-20251001"
```

`from_env()` maps `NEURALMEM_LLM_EXTRACTOR`, `NEURALMEM_COHERE_API_KEY`, etc.

**Backward compatibility:** `enable_llm_extraction=True` continues to work (maps to `llm_extractor="ollama"` internally via a `@model_validator`).

---

## 6. pyproject.toml Changes

```toml
[project.optional-dependencies]
cohere    = ["cohere>=5.0"]
gemini    = ["google-generativeai>=0.8"]
anthropic = ["anthropic>=0.30"]
# azure_openai reuses existing openai extra
# huggingface reuses existing httpx (already in dev deps)
all = ["neuralmem[server,openai,ollama,reranker,cohere,gemini,anthropic]"]
```

---

## 7. Testing Strategy

No real API calls in CI. All external SDKs are mocked via `unittest.mock.patch`.

### New test files

```
tests/unit/
├── test_embedding_cohere.py        # mock cohere.ClientV2.embed()
├── test_embedding_gemini.py        # mock genai.embed_content()
├── test_embedding_huggingface.py   # mock httpx.post()
├── test_embedding_azure.py         # mock openai.AzureOpenAI.embeddings.create()
├── test_extractor_openai.py        # mock openai.chat.completions.create()
├── test_extractor_anthropic.py     # mock anthropic.Anthropic.messages.create()
└── test_provider_registry.py       # registry routing, unknown provider fallback

tests/contract/
└── test_all_embedders_protocol.py  # parametrized: all backends satisfy EmbedderProtocol
```

### conftest.py additions

```python
@pytest.fixture
def mock_cohere_client(): ...    # returns MagicMock with .embed() returning fixed vectors
@pytest.fixture  
def mock_gemini_client(): ...
@pytest.fixture
def mock_hf_response(): ...      # httpx Response mock
```

### Coverage target

≥80% line coverage on all new `embedding/` and `extraction/` files. `registry.py` files must reach 100% (all branches exercised).

---

## 8. Error Handling

| Scenario | Behavior |
|----------|----------|
| SDK not installed | `ImportError` caught at `__init__`, raises `NeuralMemError("Install neuralmem[cohere] to use CohereEmbedding")` |
| API key missing | `ConfigurationError` at `__init__` before any network call |
| API call fails | Log warning, raise `EmbeddingError` (new exception subclass) |
| LLM extractor fails | Log warning, fallback to `MemoryExtractor` (rule-based) |
| Dimension mismatch on first use | `ConfigurationError` with message showing configured vs expected dim |

---

## 9. Delivery Order

1. `NeuralMemConfig` — add all new fields + backward-compat validator
2. `embedding/registry.py` — factory only, tests first
3. Four new embedding backends (can be parallelized: Cohere + Gemini || HuggingFace + Azure)
4. `extraction/extractor_registry.py` + two new extractors
5. Wire registries into `NeuralMem.__init__()`
6. Update `pyproject.toml` extras
7. Update `CLAUDE.md` and `README_en.md` with new provider table

---

## 10. Open Questions

- None. All decisions resolved during brainstorming session.
