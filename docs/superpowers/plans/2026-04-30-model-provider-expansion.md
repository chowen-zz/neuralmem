# Model Provider Expansion (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new embedding backends (Cohere, Gemini, HuggingFace, Azure OpenAI) and 2 new LLM extractors (OpenAI, Anthropic) behind a registry pattern, all as opt-in PyPI extras.

**Architecture:** A `registry.py` factory in each subsystem replaces hard-coded instantiation in `NeuralMem.__init__()`. All new backends inherit existing ABCs (`EmbeddingBackend`, `BaseLLMExtractor`). A new `BaseLLMExtractor` ABC extracts shared JSON-parsing + fallback logic so Ollama/OpenAI/Anthropic extractors each only override `_call_llm()`.

**Tech Stack:** Python 3.10+, Pydantic v2, `cohere>=5.0`, `google-generativeai>=0.8`, `httpx` (already present), `openai>=1.0` (already present), `anthropic>=0.30`, `pytest`, `unittest.mock`

---

## File Map

**New files:**
- `src/neuralmem/embedding/openai.py`
- `src/neuralmem/embedding/cohere.py`
- `src/neuralmem/embedding/gemini.py`
- `src/neuralmem/embedding/huggingface.py`
- `src/neuralmem/embedding/azure_openai.py`
- `src/neuralmem/embedding/registry.py`
- `src/neuralmem/extraction/base_llm_extractor.py`
- `src/neuralmem/extraction/openai_extractor.py`
- `src/neuralmem/extraction/anthropic_extractor.py`
- `src/neuralmem/extraction/extractor_registry.py`
- `tests/unit/test_embedding_openai.py`
- `tests/unit/test_embedding_cohere.py`
- `tests/unit/test_embedding_gemini.py`
- `tests/unit/test_embedding_huggingface.py`
- `tests/unit/test_embedding_azure.py`
- `tests/unit/test_extractor_openai.py`
- `tests/unit/test_extractor_anthropic.py`
- `tests/unit/test_provider_registry.py`
- `tests/contract/test_all_embedders_protocol.py`

**Modified files:**
- `src/neuralmem/core/config.py` — add new provider fields + backward-compat validator
- `src/neuralmem/extraction/llm_extractor.py` — inherit `BaseLLMExtractor`
- `src/neuralmem/core/memory.py` — use registries instead of hard-coded classes
- `pyproject.toml` — add `cohere`, `gemini`, `anthropic` extras
- `CLAUDE.md` — add provider table

---

## Task 1: Extend NeuralMemConfig

**Files:**
- Modify: `src/neuralmem/core/config.py`
- Test: `tests/unit/test_config.py` (existing — extend it)

- [ ] **Step 1: Write failing tests for new config fields**

Add to `tests/unit/test_config.py`:

```python
def test_new_provider_fields_have_defaults():
    cfg = NeuralMemConfig(db_path=":memory:")
    assert cfg.cohere_api_key is None
    assert cfg.cohere_embedding_model == "embed-multilingual-v3.0"
    assert cfg.gemini_api_key is None
    assert cfg.gemini_embedding_model == "text-embedding-004"
    assert cfg.hf_api_key is None
    assert cfg.hf_model == "BAAI/bge-m3"
    assert cfg.hf_inference_url == "https://api-inference.huggingface.co"
    assert cfg.azure_endpoint is None
    assert cfg.azure_api_key is None
    assert cfg.azure_deployment == "text-embedding-3-small"
    assert cfg.azure_api_version == "2024-02-01"
    assert cfg.llm_extractor == "none"
    assert cfg.openai_extractor_model == "gpt-4o-mini"
    assert cfg.anthropic_api_key is None
    assert cfg.anthropic_model == "claude-haiku-4-5-20251001"


def test_backward_compat_enable_llm_extraction():
    """enable_llm_extraction=True should silently map to llm_extractor='ollama'."""
    cfg = NeuralMemConfig(db_path=":memory:", enable_llm_extraction=True)
    assert cfg.llm_extractor == "ollama"


def test_explicit_llm_extractor_takes_precedence():
    cfg = NeuralMemConfig(db_path=":memory:", llm_extractor="openai")
    assert cfg.llm_extractor == "openai"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_config.py::test_new_provider_fields_have_defaults \
       tests/unit/test_config.py::test_backward_compat_enable_llm_extraction -v
```
Expected: `FAILED` — `NeuralMemConfig` has no field `cohere_api_key`

- [ ] **Step 3: Add new fields and backward-compat validator to config.py**

In `src/neuralmem/core/config.py`, add `from pydantic import model_validator` to imports, then add these fields after the existing `openai_embedding_model` field:

```python
    # Cohere
    cohere_api_key: str | None = Field(default=None)
    cohere_embedding_model: str = Field(default="embed-multilingual-v3.0")

    # Gemini
    gemini_api_key: str | None = Field(default=None)
    gemini_embedding_model: str = Field(default="text-embedding-004")

    # HuggingFace Inference API
    hf_api_key: str | None = Field(default=None)
    hf_model: str = Field(default="BAAI/bge-m3")
    hf_inference_url: str = Field(default="https://api-inference.huggingface.co")

    # Azure OpenAI
    azure_endpoint: str | None = Field(default=None)
    azure_api_key: str | None = Field(default=None)
    azure_deployment: str = Field(default="text-embedding-3-small")
    azure_api_version: str = Field(default="2024-02-01")

    # LLM extractor selection (replaces enable_llm_extraction bool)
    llm_extractor: str = Field(
        default="none",
        description="LLM 提取器：none | ollama | openai | anthropic",
    )
    openai_extractor_model: str = Field(default="gpt-4o-mini")
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-haiku-4-5-20251001")

    @model_validator(mode="after")
    def _backcompat_llm_extraction(self) -> "NeuralMemConfig":
        if self.enable_llm_extraction and self.llm_extractor == "none":
            self.llm_extractor = "ollama"
        return self
```

Also extend `from_env()` — replace the `return cls(...)` call body with these additions after the existing `openai_api_key` line:

```python
            cohere_api_key=os.getenv("NEURALMEM_COHERE_API_KEY"),
            cohere_embedding_model=os.getenv("NEURALMEM_COHERE_EMBEDDING_MODEL", "embed-multilingual-v3.0"),
            gemini_api_key=os.getenv("NEURALMEM_GEMINI_API_KEY"),
            gemini_embedding_model=os.getenv("NEURALMEM_GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
            hf_api_key=os.getenv("NEURALMEM_HF_API_KEY"),
            hf_model=os.getenv("NEURALMEM_HF_MODEL", "BAAI/bge-m3"),
            hf_inference_url=os.getenv("NEURALMEM_HF_INFERENCE_URL", "https://api-inference.huggingface.co"),
            azure_endpoint=os.getenv("NEURALMEM_AZURE_ENDPOINT"),
            azure_api_key=os.getenv("NEURALMEM_AZURE_API_KEY"),
            azure_deployment=os.getenv("NEURALMEM_AZURE_DEPLOYMENT", "text-embedding-3-small"),
            azure_api_version=os.getenv("NEURALMEM_AZURE_API_VERSION", "2024-02-01"),
            llm_extractor=os.getenv("NEURALMEM_LLM_EXTRACTOR", "none"),
            openai_extractor_model=os.getenv("NEURALMEM_OPENAI_EXTRACTOR_MODEL", "gpt-4o-mini"),
            anthropic_api_key=os.getenv("NEURALMEM_ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("NEURALMEM_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_config.py -v
```
Expected: all config tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/core/config.py tests/unit/test_config.py
git commit -m "feat(config): 添加多 provider 配置字段及向后兼容验证器"
```

---

## Task 2: Embedding Registry

**Files:**
- Create: `src/neuralmem/embedding/registry.py`
- Create: `tests/unit/test_provider_registry.py` (partial — embedding section)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_provider_registry.py`:

```python
"""Tests for embedding and extractor registry routing."""
import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.embedding.local import LocalEmbedding


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def test_registry_default_returns_local():
    from neuralmem.embedding.registry import get_embedder
    result = get_embedder(cfg())
    assert isinstance(result, LocalEmbedding)


def test_registry_unknown_provider_falls_back_to_local():
    from neuralmem.embedding.registry import get_embedder
    result = get_embedder(cfg(embedding_provider="nonexistent"))
    assert isinstance(result, LocalEmbedding)


def test_registry_openai_instantiates_openai_embedding():
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.registry import get_embedder
        from neuralmem.embedding.openai import OpenAIEmbedding
        result = get_embedder(cfg(embedding_provider="openai", openai_api_key="sk-test"))
        assert isinstance(result, OpenAIEmbedding)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_provider_registry.py::test_registry_default_returns_local -v
```
Expected: `FAILED` — `cannot import name 'get_embedder' from 'neuralmem.embedding.registry'`

- [ ] **Step 3: Create registry.py**

Create `src/neuralmem/embedding/registry.py`:

```python
"""Embedding backend factory — maps config.embedding_provider to a concrete EmbeddingBackend."""
from __future__ import annotations
from neuralmem.core.config import NeuralMemConfig
from neuralmem.embedding.base import EmbeddingBackend


def get_embedder(config: NeuralMemConfig) -> EmbeddingBackend:
    """Return the configured embedding backend. Unknown providers fall back to LocalEmbedding."""
    match config.embedding_provider:
        case "openai":
            from neuralmem.embedding.openai import OpenAIEmbedding
            return OpenAIEmbedding(config)
        case "cohere":
            from neuralmem.embedding.cohere import CohereEmbedding
            return CohereEmbedding(config)
        case "gemini":
            from neuralmem.embedding.gemini import GeminiEmbedding
            return GeminiEmbedding(config)
        case "huggingface":
            from neuralmem.embedding.huggingface import HuggingFaceEmbedding
            return HuggingFaceEmbedding(config)
        case "azure_openai":
            from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
            return AzureOpenAIEmbedding(config)
        case _:
            from neuralmem.embedding.local import LocalEmbedding
            return LocalEmbedding(config)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_provider_registry.py::test_registry_default_returns_local \
       tests/unit/test_provider_registry.py::test_registry_unknown_provider_falls_back_to_local -v
```
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/embedding/registry.py tests/unit/test_provider_registry.py
git commit -m "feat(embedding): 添加 embedding provider registry 工厂"
```

---

## Task 3: OpenAI Embedding Backend

**Files:**
- Create: `src/neuralmem/embedding/openai.py`
- Create: `tests/unit/test_embedding_openai.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_embedding_openai.py`:

```python
import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_openai_sdk():
    mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
    mock.OpenAI.return_value.embeddings.create.return_value = mock_resp
    return mock


def test_openai_missing_api_key_raises_config_error():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        with pytest.raises(ConfigError, match="openai_api_key"):
            OpenAIEmbedding(cfg())


def test_openai_sdk_not_installed_raises_neuralmem_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _block_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    if "neuralmem.embedding.openai" in sys.modules:
        del sys.modules["neuralmem.embedding.openai"]
    monkeypatch.setattr(builtins, "__import__", _block_openai)
    from neuralmem.embedding.openai import OpenAIEmbedding
    with pytest.raises(NeuralMemError, match="neuralmem\\[openai\\]"):
        OpenAIEmbedding(cfg(openai_api_key="sk-test"))


def test_openai_encode_returns_vectors():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        embedder = OpenAIEmbedding(cfg(openai_api_key="sk-test"))
        result = embedder.encode(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 1536


def test_openai_encode_empty_returns_empty():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        embedder = OpenAIEmbedding(cfg(openai_api_key="sk-test"))
        assert embedder.encode([]) == []


def test_openai_api_failure_raises_embedding_error():
    mock_sdk = _mock_openai_sdk()
    mock_sdk.OpenAI.return_value.embeddings.create.side_effect = Exception("rate limit")
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        embedder = OpenAIEmbedding(cfg(openai_api_key="sk-test"))
        with pytest.raises(EmbeddingError, match="OpenAI embedding failed"):
            embedder.encode(["hello"])


def test_openai_dimension_for_known_models():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.openai" in sys.modules:
            del sys.modules["neuralmem.embedding.openai"]
        from neuralmem.embedding.openai import OpenAIEmbedding
        e = OpenAIEmbedding(cfg(openai_api_key="sk-test", openai_embedding_model="text-embedding-3-large"))
        assert e.dimension == 3072
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_embedding_openai.py -v
```
Expected: `FAILED` — `cannot import name 'OpenAIEmbedding'`

- [ ] **Step 3: Create openai.py**

Create `src/neuralmem/embedding/openai.py`:

```python
"""OpenAI Embedding backend — requires neuralmem[openai]."""
from __future__ import annotations
from typing import Sequence
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError
from neuralmem.embedding.base import EmbeddingBackend

_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[openai] to use OpenAIEmbedding: pip install 'neuralmem[openai]'"
            ) from exc
        if not config.openai_api_key:
            raise ConfigError(
                "openai_api_key is required for OpenAIEmbedding. "
                "Set NEURALMEM_OPENAI_API_KEY or OPENAI_API_KEY."
            )
        self._client = _openai.OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_embedding_model

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, 1536)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self._model, input=list(texts))
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_embedding_openai.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/embedding/openai.py tests/unit/test_embedding_openai.py
git commit -m "feat(embedding): 添加 OpenAI embedding backend"
```

---

## Task 4: Cohere Embedding Backend

**Files:**
- Create: `src/neuralmem/embedding/cohere.py`
- Create: `tests/unit/test_embedding_cohere.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_embedding_cohere.py`:

```python
import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_cohere_sdk():
    mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.embeddings.float_ = [[0.1] * 1024]
    mock.ClientV2.return_value.embed.return_value = mock_resp
    return mock


def test_cohere_missing_api_key_raises():
    mock_sdk = _mock_cohere_sdk()
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        with pytest.raises(ConfigError, match="cohere_api_key"):
            CohereEmbedding(cfg())


def test_cohere_encode_returns_vectors():
    mock_sdk = _mock_cohere_sdk()
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        embedder = CohereEmbedding(cfg(cohere_api_key="test-key"))
        result = embedder.encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 1024


def test_cohere_encode_empty_returns_empty():
    mock_sdk = _mock_cohere_sdk()
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        embedder = CohereEmbedding(cfg(cohere_api_key="test-key"))
        assert embedder.encode([]) == []


def test_cohere_api_failure_raises_embedding_error():
    mock_sdk = _mock_cohere_sdk()
    mock_sdk.ClientV2.return_value.embed.side_effect = Exception("quota exceeded")
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        embedder = CohereEmbedding(cfg(cohere_api_key="test-key"))
        with pytest.raises(EmbeddingError, match="Cohere embedding failed"):
            embedder.encode(["hello"])


def test_cohere_dimension_light_model():
    mock_sdk = _mock_cohere_sdk()
    with patch.dict(sys.modules, {"cohere": mock_sdk}):
        if "neuralmem.embedding.cohere" in sys.modules:
            del sys.modules["neuralmem.embedding.cohere"]
        from neuralmem.embedding.cohere import CohereEmbedding
        e = CohereEmbedding(cfg(cohere_api_key="k", cohere_embedding_model="embed-multilingual-light-v3.0"))
        assert e.dimension == 384
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_embedding_cohere.py -v
```
Expected: `FAILED` — `cannot import name 'CohereEmbedding'`

- [ ] **Step 3: Create cohere.py**

Create `src/neuralmem/embedding/cohere.py`:

```python
"""Cohere Embedding backend — requires neuralmem[cohere]."""
from __future__ import annotations
from typing import Sequence
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError
from neuralmem.embedding.base import EmbeddingBackend

_KNOWN_DIMS: dict[str, int] = {
    "embed-multilingual-v3.0": 1024,
    "embed-english-v3.0": 1024,
    "embed-multilingual-light-v3.0": 384,
    "embed-english-light-v3.0": 384,
}


class CohereEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        try:
            import cohere as _cohere
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[cohere] to use CohereEmbedding: pip install 'neuralmem[cohere]'"
            ) from exc
        if not config.cohere_api_key:
            raise ConfigError(
                "cohere_api_key is required for CohereEmbedding. Set NEURALMEM_COHERE_API_KEY."
            )
        self._client = _cohere.ClientV2(api_key=config.cohere_api_key)
        self._model = config.cohere_embedding_model

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, 1024)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embed(
                texts=list(texts),
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
            )
            return [list(vec) for vec in response.embeddings.float_]
        except Exception as exc:
            raise EmbeddingError(f"Cohere embedding failed: {exc}") from exc
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_embedding_cohere.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/embedding/cohere.py tests/unit/test_embedding_cohere.py
git commit -m "feat(embedding): 添加 Cohere embedding backend"
```

---

## Task 5: Gemini Embedding Backend

**Files:**
- Create: `src/neuralmem/embedding/gemini.py`
- Create: `tests/unit/test_embedding_gemini.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_embedding_gemini.py`:

```python
import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_genai_sdk():
    mock = MagicMock()
    mock.embed_content.return_value = {"embedding": [0.1] * 768}
    return mock


def test_gemini_missing_api_key_raises():
    mock_sdk = _mock_genai_sdk()
    with patch.dict(sys.modules, {"google.generativeai": mock_sdk, "google": MagicMock()}):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        with pytest.raises(ConfigError, match="gemini_api_key"):
            GeminiEmbedding(cfg())


def test_gemini_encode_returns_vectors():
    mock_sdk = _mock_genai_sdk()
    with patch.dict(sys.modules, {"google.generativeai": mock_sdk, "google": MagicMock()}):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        embedder = GeminiEmbedding(cfg(gemini_api_key="test-key"))
        result = embedder.encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 768


def test_gemini_encode_empty_returns_empty():
    mock_sdk = _mock_genai_sdk()
    with patch.dict(sys.modules, {"google.generativeai": mock_sdk, "google": MagicMock()}):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        embedder = GeminiEmbedding(cfg(gemini_api_key="test-key"))
        assert embedder.encode([]) == []


def test_gemini_api_failure_raises_embedding_error():
    mock_sdk = _mock_genai_sdk()
    mock_sdk.embed_content.side_effect = Exception("model not found")
    with patch.dict(sys.modules, {"google.generativeai": mock_sdk, "google": MagicMock()}):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        embedder = GeminiEmbedding(cfg(gemini_api_key="test-key"))
        with pytest.raises(EmbeddingError, match="Gemini embedding failed"):
            embedder.encode(["hello"])


def test_gemini_dimension():
    mock_sdk = _mock_genai_sdk()
    with patch.dict(sys.modules, {"google.generativeai": mock_sdk, "google": MagicMock()}):
        if "neuralmem.embedding.gemini" in sys.modules:
            del sys.modules["neuralmem.embedding.gemini"]
        from neuralmem.embedding.gemini import GeminiEmbedding
        e = GeminiEmbedding(cfg(gemini_api_key="k"))
        assert e.dimension == 768
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_embedding_gemini.py -v
```
Expected: `FAILED` — `cannot import name 'GeminiEmbedding'`

- [ ] **Step 3: Create gemini.py**

Create `src/neuralmem/embedding/gemini.py`:

```python
"""Google Gemini Embedding backend — requires neuralmem[gemini]."""
from __future__ import annotations
from typing import Sequence
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError
from neuralmem.embedding.base import EmbeddingBackend

_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-004": 768,
    "embedding-001": 768,
}


class GeminiEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        try:
            import google.generativeai as _genai
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[gemini] to use GeminiEmbedding: pip install 'neuralmem[gemini]'"
            ) from exc
        if not config.gemini_api_key:
            raise ConfigError(
                "gemini_api_key is required for GeminiEmbedding. Set NEURALMEM_GEMINI_API_KEY."
            )
        _genai.configure(api_key=config.gemini_api_key)
        self._genai = _genai
        self._model = config.gemini_embedding_model

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, 768)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return [
                self._genai.embed_content(model=self._model, content=text)["embedding"]
                for text in texts
            ]
        except Exception as exc:
            raise EmbeddingError(f"Gemini embedding failed: {exc}") from exc
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_embedding_gemini.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/embedding/gemini.py tests/unit/test_embedding_gemini.py
git commit -m "feat(embedding): 添加 Google Gemini embedding backend"
```

---

## Task 6: HuggingFace Inference API Embedding Backend

**Files:**
- Create: `src/neuralmem/embedding/huggingface.py`
- Create: `tests/unit/test_embedding_huggingface.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_embedding_huggingface.py`:

```python
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_response(data):
    mock = MagicMock()
    mock.json.return_value = data
    mock.raise_for_status.return_value = None
    return mock


def test_hf_missing_api_key_raises():
    from neuralmem.embedding.huggingface import HuggingFaceEmbedding
    with pytest.raises(ConfigError, match="hf_api_key"):
        HuggingFaceEmbedding(cfg())


def test_hf_encode_flat_response():
    resp = _mock_response([[0.1] * 1024])
    with patch("neuralmem.embedding.huggingface.httpx.post", return_value=resp):
        from neuralmem.embedding.huggingface import HuggingFaceEmbedding
        embedder = HuggingFaceEmbedding(cfg(hf_api_key="hf-test"))
        result = embedder.encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 1024


def test_hf_encode_nested_response_mean_pools():
    """HF returns [n_texts, n_tokens, dim] for some models — should mean pool."""
    token_vecs = [[1.0, 0.0], [0.0, 1.0]]  # 2 tokens, dim=2
    resp = _mock_response([token_vecs])
    with patch("neuralmem.embedding.huggingface.httpx.post", return_value=resp):
        from neuralmem.embedding.huggingface import HuggingFaceEmbedding
        embedder = HuggingFaceEmbedding(cfg(hf_api_key="hf-test"))
        result = embedder.encode(["hello"])
        assert result[0] == pytest.approx([0.5, 0.5])


def test_hf_encode_empty_returns_empty():
    from neuralmem.embedding.huggingface import HuggingFaceEmbedding
    embedder = HuggingFaceEmbedding(cfg(hf_api_key="hf-test"))
    assert embedder.encode([]) == []


def test_hf_http_error_raises_embedding_error():
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"
    with patch(
        "neuralmem.embedding.huggingface.httpx.post",
        side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=mock_resp),
    ):
        from neuralmem.embedding.huggingface import HuggingFaceEmbedding
        embedder = HuggingFaceEmbedding(cfg(hf_api_key="hf-test"))
        with pytest.raises(EmbeddingError, match="HuggingFace API error 503"):
            embedder.encode(["hello"])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_embedding_huggingface.py -v
```
Expected: `FAILED` — `cannot import name 'HuggingFaceEmbedding'`

- [ ] **Step 3: Create huggingface.py**

Create `src/neuralmem/embedding/huggingface.py`:

```python
"""HuggingFace Inference API Embedding backend — uses httpx (no extra install needed)."""
from __future__ import annotations
from typing import Sequence
import httpx
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError
from neuralmem.embedding.base import EmbeddingBackend

_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class HuggingFaceEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        if not config.hf_api_key:
            raise ConfigError(
                "hf_api_key is required for HuggingFaceEmbedding. Set NEURALMEM_HF_API_KEY."
            )
        self._api_key = config.hf_api_key
        self._model = config.hf_model
        base = config.hf_inference_url.rstrip("/")
        self._url = f"{base}/pipeline/feature-extraction/{self._model}"

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._model, 1024)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"inputs": list(texts), "options": {"wait_for_model": True}},
                timeout=30.0,
            )
            response.raise_for_status()
            data: list = response.json()
            results: list[list[float]] = []
            for item in data:
                if item and isinstance(item[0], list):
                    n = len(item)
                    dim = len(item[0])
                    pooled = [sum(item[t][d] for t in range(n)) / n for d in range(dim)]
                    results.append(pooled)
                else:
                    results.append(item)
            return results
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"HuggingFace API error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(f"HuggingFace embedding failed: {exc}") from exc
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_embedding_huggingface.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/embedding/huggingface.py tests/unit/test_embedding_huggingface.py
git commit -m "feat(embedding): 添加 HuggingFace Inference API embedding backend"
```

---

## Task 7: Azure OpenAI Embedding Backend

**Files:**
- Create: `src/neuralmem/embedding/azure_openai.py`
- Create: `tests/unit/test_embedding_azure.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_embedding_azure.py`:

```python
import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_openai_sdk():
    mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1] * 1536)]
    mock.AzureOpenAI.return_value.embeddings.create.return_value = mock_resp
    return mock


def test_azure_missing_endpoint_raises():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        with pytest.raises(ConfigError, match="azure_endpoint"):
            AzureOpenAIEmbedding(cfg(azure_api_key="key"))


def test_azure_missing_api_key_raises():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        with pytest.raises(ConfigError, match="azure_api_key"):
            AzureOpenAIEmbedding(cfg(azure_endpoint="https://my.openai.azure.com"))


def test_azure_encode_returns_vectors():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        e = AzureOpenAIEmbedding(cfg(azure_endpoint="https://x.openai.azure.com", azure_api_key="k"))
        result = e.encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 1536


def test_azure_encode_empty_returns_empty():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        e = AzureOpenAIEmbedding(cfg(azure_endpoint="https://x.openai.azure.com", azure_api_key="k"))
        assert e.encode([]) == []


def test_azure_api_failure_raises_embedding_error():
    mock_sdk = _mock_openai_sdk()
    mock_sdk.AzureOpenAI.return_value.embeddings.create.side_effect = Exception("auth failed")
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.embedding.azure_openai" in sys.modules:
            del sys.modules["neuralmem.embedding.azure_openai"]
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        e = AzureOpenAIEmbedding(cfg(azure_endpoint="https://x.openai.azure.com", azure_api_key="k"))
        with pytest.raises(EmbeddingError, match="Azure OpenAI embedding failed"):
            e.encode(["hello"])
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_embedding_azure.py -v
```
Expected: `FAILED` — `cannot import name 'AzureOpenAIEmbedding'`

- [ ] **Step 3: Create azure_openai.py**

Create `src/neuralmem/embedding/azure_openai.py`:

```python
"""Azure OpenAI Embedding backend — requires neuralmem[openai]."""
from __future__ import annotations
from typing import Sequence
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, EmbeddingError, NeuralMemError
from neuralmem.embedding.base import EmbeddingBackend

_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class AzureOpenAIEmbedding(EmbeddingBackend):
    def __init__(self, config: NeuralMemConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[openai] to use AzureOpenAIEmbedding: pip install 'neuralmem[openai]'"
            ) from exc
        if not config.azure_endpoint:
            raise ConfigError(
                "azure_endpoint is required for AzureOpenAIEmbedding. Set NEURALMEM_AZURE_ENDPOINT."
            )
        if not config.azure_api_key:
            raise ConfigError(
                "azure_api_key is required for AzureOpenAIEmbedding. Set NEURALMEM_AZURE_API_KEY."
            )
        self._client = _openai.AzureOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
        )
        self._deployment = config.azure_deployment

    @property
    def dimension(self) -> int:
        return _KNOWN_DIMS.get(self._deployment, 1536)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self._deployment, input=list(texts))
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise EmbeddingError(f"Azure OpenAI embedding failed: {exc}") from exc
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_embedding_azure.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/embedding/azure_openai.py tests/unit/test_embedding_azure.py
git commit -m "feat(embedding): 添加 Azure OpenAI embedding backend"
```

---

## Task 8: BaseLLMExtractor + Refactor Ollama Extractor

**Files:**
- Create: `src/neuralmem/extraction/base_llm_extractor.py`
- Modify: `src/neuralmem/extraction/llm_extractor.py`

- [ ] **Step 1: Write failing test for BaseLLMExtractor**

Add to `tests/unit/test_llm_extractor.py` (existing file — add these tests):

```python
def test_base_llm_extractor_fallback_on_error():
    """BaseLLMExtractor falls back to rule extractor when _call_llm raises."""
    from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor
    from neuralmem.core.config import NeuralMemConfig

    class BrokenExtractor(BaseLLMExtractor):
        def _call_llm(self, prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

    cfg = NeuralMemConfig(db_path=":memory:")
    extractor = BrokenExtractor(cfg)
    items = extractor.extract("Alice works at OpenAI")
    assert len(items) >= 1
    assert "Alice" in items[0].content or "OpenAI" in items[0].content or items[0].content


def test_base_llm_extractor_merges_entities():
    """BaseLLMExtractor merges LLM-extracted entities with rule-extracted ones."""
    from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor
    from neuralmem.core.config import NeuralMemConfig

    class FakeExtractor(BaseLLMExtractor):
        def _call_llm(self, prompt: str) -> str:
            return '{"facts": [], "entities": [{"name": "DeepMind", "type": "project"}]}'

    cfg = NeuralMemConfig(db_path=":memory:")
    extractor = FakeExtractor(cfg)
    items = extractor.extract("Alice works at DeepMind")
    all_entity_names = [e.name for e in items[0].entities] if items else []
    assert "DeepMind" in all_entity_names
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_llm_extractor.py::test_base_llm_extractor_fallback_on_error -v
```
Expected: `FAILED` — `cannot import name 'BaseLLMExtractor'`

- [ ] **Step 3: Create base_llm_extractor.py**

Create `src/neuralmem/extraction/base_llm_extractor.py`:

```python
"""Shared base for all LLM-backed extractors — subclasses only implement _call_llm()."""
from __future__ import annotations
import json
import logging
from abc import ABC, abstractmethod
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.types import Entity
from neuralmem.extraction.extractor import ExtractedItem, MemoryExtractor

_logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = (
    "Extract key facts and entities from the following text.\n"
    'Return JSON with: {{"facts": ["fact1", "fact2"], '
    '"entities": [{{"name": "X", "type": "person|project|technology|concept"}}]}}\n'
    "Text: {text}\nJSON:"
)


class BaseLLMExtractor(ABC):
    def __init__(self, config: NeuralMemConfig) -> None:
        self._config = config
        self._rule_extractor = MemoryExtractor(config)

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return raw response string."""
        ...

    def extract(self, content: str, **kwargs: object) -> list[ExtractedItem]:
        try:
            prompt = _EXTRACT_PROMPT.format(text=content[:1000])
            raw = self._call_llm(prompt)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            items = self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]
            extra_entities = [
                Entity(name=e["name"], entity_type=e.get("type", "concept"))
                for e in data.get("entities", [])
            ]
            if items and extra_entities:
                first = items[0]
                merged = list({e.name: e for e in first.entities + extra_entities}.values())
                items[0] = ExtractedItem(
                    content=first.content,
                    memory_type=first.memory_type,
                    entities=merged,
                    relations=first.relations,
                    tags=first.tags,
                    importance=first.importance,
                )
            return items
        except Exception as exc:
            _logger.warning("LLM extraction failed (%s), falling back to rules.", exc)
            return self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]
```

- [ ] **Step 4: Refactor llm_extractor.py to inherit BaseLLMExtractor**

Replace the entire content of `src/neuralmem/extraction/llm_extractor.py` with:

```python
"""Ollama-backed LLM extractor — inherits BaseLLMExtractor."""
from __future__ import annotations
import logging
from neuralmem.core.config import NeuralMemConfig
from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor

_logger = logging.getLogger(__name__)


class LLMExtractor(BaseLLMExtractor):
    """Ollama LLM 增强提取器。enable_llm_extraction=True 或 llm_extractor='ollama' 时使用。"""

    def __init__(self, config: NeuralMemConfig) -> None:
        super().__init__(config)
        self._available: bool | None = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import httpx
            resp = httpx.get(f"{self._config.ollama_url}/api/tags", timeout=2.0)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
            _logger.info("Ollama not available at %s, using rule extractor.", self._config.ollama_url)
        return self._available

    def _call_llm(self, prompt: str) -> str:
        if not self._check_available():
            raise RuntimeError(f"Ollama not available at {self._config.ollama_url}")
        import httpx
        resp = httpx.post(
            f"{self._config.ollama_url}/api/generate",
            json={"model": self._config.ollama_model, "prompt": prompt, "stream": False},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "{}")
```

- [ ] **Step 5: Run all extractor tests to confirm nothing regressed**

```bash
pytest tests/unit/test_llm_extractor.py tests/unit/test_extractor.py -v
```
Expected: all `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/neuralmem/extraction/base_llm_extractor.py \
        src/neuralmem/extraction/llm_extractor.py \
        tests/unit/test_llm_extractor.py
git commit -m "refactor(extraction): 提取 BaseLLMExtractor，重构 Ollama extractor 继承它"
```

---

## Task 9: OpenAI LLM Extractor

**Files:**
- Create: `src/neuralmem/extraction/openai_extractor.py`
- Create: `tests/unit/test_extractor_openai.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_extractor_openai.py`:

```python
import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, NeuralMemError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_openai_sdk(response_text: str = '{"facts": [], "entities": []}'):
    mock = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    mock.OpenAI.return_value.chat.completions.create.return_value.choices = [choice]
    return mock


def test_openai_extractor_missing_api_key_raises():
    mock_sdk = _mock_openai_sdk()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.extraction.openai_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.openai_extractor"]
        from neuralmem.extraction.openai_extractor import OpenAIExtractor
        with pytest.raises(ConfigError, match="openai_api_key"):
            OpenAIExtractor(cfg())


def test_openai_extractor_extracts_entities():
    mock_sdk = _mock_openai_sdk(
        '{"facts": ["Alice works at OpenAI"], "entities": [{"name": "Alice", "type": "person"}, {"name": "OpenAI", "type": "project"}]}'
    )
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.extraction.openai_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.openai_extractor"]
        from neuralmem.extraction.openai_extractor import OpenAIExtractor
        extractor = OpenAIExtractor(cfg(openai_api_key="sk-test"))
        items = extractor.extract("Alice works at OpenAI")
        all_names = [e.name for e in items[0].entities] if items else []
        assert "Alice" in all_names or "OpenAI" in all_names


def test_openai_extractor_fallback_on_llm_error():
    mock_sdk = _mock_openai_sdk()
    mock_sdk.OpenAI.return_value.chat.completions.create.side_effect = Exception("rate limit")
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        if "neuralmem.extraction.openai_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.openai_extractor"]
        from neuralmem.extraction.openai_extractor import OpenAIExtractor
        extractor = OpenAIExtractor(cfg(openai_api_key="sk-test"))
        items = extractor.extract("hello world")
        assert isinstance(items, list)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_extractor_openai.py -v
```
Expected: `FAILED` — `cannot import name 'OpenAIExtractor'`

- [ ] **Step 3: Create openai_extractor.py**

Create `src/neuralmem/extraction/openai_extractor.py`:

```python
"""OpenAI LLM Extractor — requires neuralmem[openai]."""
from __future__ import annotations
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, NeuralMemError
from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor


class OpenAIExtractor(BaseLLMExtractor):
    def __init__(self, config: NeuralMemConfig) -> None:
        super().__init__(config)
        try:
            import openai as _openai
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[openai] to use OpenAIExtractor: pip install 'neuralmem[openai]'"
            ) from exc
        if not config.openai_api_key:
            raise ConfigError("openai_api_key is required for OpenAIExtractor.")
        self._client = _openai.OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_extractor_model

    def _call_llm(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content or "{}"
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_extractor_openai.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/extraction/openai_extractor.py tests/unit/test_extractor_openai.py
git commit -m "feat(extraction): 添加 OpenAI LLM extractor"
```

---

## Task 10: Anthropic LLM Extractor

**Files:**
- Create: `src/neuralmem/extraction/anthropic_extractor.py`
- Create: `tests/unit/test_extractor_anthropic.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_extractor_anthropic.py`:

```python
import sys
from unittest.mock import MagicMock, patch
import pytest
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, NeuralMemError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_anthropic_sdk(response_text: str = '{"facts": [], "entities": []}'):
    mock = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    mock.Anthropic.return_value.messages.create.return_value.content = [content_block]
    return mock


def test_anthropic_extractor_missing_api_key_raises():
    mock_sdk = _mock_anthropic_sdk()
    with patch.dict(sys.modules, {"anthropic": mock_sdk}):
        if "neuralmem.extraction.anthropic_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.anthropic_extractor"]
        from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
        with pytest.raises(ConfigError, match="anthropic_api_key"):
            AnthropicExtractor(cfg())


def test_anthropic_extractor_extracts_entities():
    mock_sdk = _mock_anthropic_sdk(
        '{"facts": [], "entities": [{"name": "Claude", "type": "technology"}]}'
    )
    with patch.dict(sys.modules, {"anthropic": mock_sdk}):
        if "neuralmem.extraction.anthropic_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.anthropic_extractor"]
        from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
        extractor = AnthropicExtractor(cfg(anthropic_api_key="sk-ant-test"))
        items = extractor.extract("Claude is made by Anthropic")
        all_names = [e.name for e in items[0].entities] if items else []
        assert "Claude" in all_names


def test_anthropic_extractor_fallback_on_llm_error():
    mock_sdk = _mock_anthropic_sdk()
    mock_sdk.Anthropic.return_value.messages.create.side_effect = Exception("overloaded")
    with patch.dict(sys.modules, {"anthropic": mock_sdk}):
        if "neuralmem.extraction.anthropic_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.anthropic_extractor"]
        from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
        extractor = AnthropicExtractor(cfg(anthropic_api_key="sk-ant-test"))
        items = extractor.extract("hello world")
        assert isinstance(items, list)


def test_anthropic_extractor_uses_configured_model():
    mock_sdk = _mock_anthropic_sdk()
    with patch.dict(sys.modules, {"anthropic": mock_sdk}):
        if "neuralmem.extraction.anthropic_extractor" in sys.modules:
            del sys.modules["neuralmem.extraction.anthropic_extractor"]
        from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
        extractor = AnthropicExtractor(cfg(anthropic_api_key="sk-ant-test"))
        extractor.extract("test")
        call_kwargs = mock_sdk.Anthropic.return_value.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_extractor_anthropic.py -v
```
Expected: `FAILED` — `cannot import name 'AnthropicExtractor'`

- [ ] **Step 3: Create anthropic_extractor.py**

Create `src/neuralmem/extraction/anthropic_extractor.py`:

```python
"""Anthropic Claude LLM Extractor — requires neuralmem[anthropic]."""
from __future__ import annotations
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ConfigError, NeuralMemError
from neuralmem.extraction.base_llm_extractor import BaseLLMExtractor


class AnthropicExtractor(BaseLLMExtractor):
    def __init__(self, config: NeuralMemConfig) -> None:
        super().__init__(config)
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise NeuralMemError(
                "Install neuralmem[anthropic] to use AnthropicExtractor: pip install 'neuralmem[anthropic]'"
            ) from exc
        if not config.anthropic_api_key:
            raise ConfigError("anthropic_api_key is required for AnthropicExtractor.")
        self._client = _anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._model = config.anthropic_model

    def _call_llm(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text if message.content else "{}"
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_extractor_anthropic.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/extraction/anthropic_extractor.py tests/unit/test_extractor_anthropic.py
git commit -m "feat(extraction): 添加 Anthropic Claude LLM extractor"
```

---

## Task 11: Extractor Registry

**Files:**
- Create: `src/neuralmem/extraction/extractor_registry.py`
- Modify: `tests/unit/test_provider_registry.py` (add extractor section)

- [ ] **Step 1: Write failing tests — add to test_provider_registry.py**

```python
def test_extractor_registry_default_returns_rule_extractor():
    from neuralmem.extraction.extractor_registry import get_extractor
    from neuralmem.extraction.extractor import MemoryExtractor
    result = get_extractor(cfg())
    assert isinstance(result, MemoryExtractor)


def test_extractor_registry_ollama_returns_llm_extractor():
    from neuralmem.extraction.extractor_registry import get_extractor
    from neuralmem.extraction.llm_extractor import LLMExtractor
    result = get_extractor(cfg(llm_extractor="ollama"))
    assert isinstance(result, LLMExtractor)


def test_extractor_registry_unknown_falls_back_to_rule():
    from neuralmem.extraction.extractor_registry import get_extractor
    from neuralmem.extraction.extractor import MemoryExtractor
    result = get_extractor(cfg(llm_extractor="nonexistent"))
    assert isinstance(result, MemoryExtractor)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_provider_registry.py::test_extractor_registry_default_returns_rule_extractor -v
```
Expected: `FAILED` — `cannot import name 'get_extractor'`

- [ ] **Step 3: Create extractor_registry.py**

Create `src/neuralmem/extraction/extractor_registry.py`:

```python
"""Extractor factory — maps config.llm_extractor to a concrete extractor instance."""
from __future__ import annotations
from neuralmem.core.config import NeuralMemConfig
from neuralmem.extraction.extractor import MemoryExtractor


def get_extractor(config: NeuralMemConfig) -> MemoryExtractor:
    """Return the configured LLM extractor, falling back to rule-based on unknown values."""
    match config.llm_extractor:
        case "openai":
            from neuralmem.extraction.openai_extractor import OpenAIExtractor
            return OpenAIExtractor(config)  # type: ignore[return-value]
        case "anthropic":
            from neuralmem.extraction.anthropic_extractor import AnthropicExtractor
            return AnthropicExtractor(config)  # type: ignore[return-value]
        case "ollama":
            from neuralmem.extraction.llm_extractor import LLMExtractor
            return LLMExtractor(config)  # type: ignore[return-value]
        case _:
            return MemoryExtractor(config)
```

- [ ] **Step 4: Run all registry tests**

```bash
pytest tests/unit/test_provider_registry.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/neuralmem/extraction/extractor_registry.py tests/unit/test_provider_registry.py
git commit -m "feat(extraction): 添加 extractor registry 工厂"
```

---

## Task 12: Wire Registries into NeuralMem

**Files:**
- Modify: `src/neuralmem/core/memory.py`

- [ ] **Step 1: Write integration test to confirm registry wiring**

Add to `tests/integration/test_memory_facade.py`:

```python
def test_neuralmem_uses_embedding_registry(tmp_path):
    """NeuralMem should use registry to select embedder, defaulting to LocalEmbedding."""
    from neuralmem.embedding.local import LocalEmbedding
    from neuralmem.core.config import NeuralMemConfig
    from neuralmem.core.memory import NeuralMem
    cfg = NeuralMemConfig(db_path=str(tmp_path / "test.db"))
    mem = NeuralMem(config=cfg)
    assert isinstance(mem.embedding, LocalEmbedding)


def test_neuralmem_uses_extractor_registry(tmp_path):
    """NeuralMem should use registry to select extractor, defaulting to MemoryExtractor."""
    from neuralmem.extraction.extractor import MemoryExtractor
    from neuralmem.core.config import NeuralMemConfig
    from neuralmem.core.memory import NeuralMem
    cfg = NeuralMemConfig(db_path=str(tmp_path / "test.db"))
    mem = NeuralMem(config=cfg)
    assert isinstance(mem.extractor, MemoryExtractor)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/integration/test_memory_facade.py::test_neuralmem_uses_embedding_registry \
       tests/integration/test_memory_facade.py::test_neuralmem_uses_extractor_registry -v
```
Expected: `FAILED` (embedding still uses `LocalEmbedding` directly, extractor still uses old branch)

- [ ] **Step 3: Update memory.py to use registries**

In `src/neuralmem/core/memory.py`:

Replace the import block at the top — remove:
```python
from neuralmem.embedding.local import LocalEmbedding
from neuralmem.extraction.extractor import MemoryExtractor
from neuralmem.extraction.llm_extractor import LLMExtractor
```

Add:
```python
from neuralmem.embedding.registry import get_embedder
from neuralmem.extraction.extractor_registry import get_extractor
```

Replace in `__init__`:
```python
        self.embedding = LocalEmbedding(self.config)
```
with:
```python
        self.embedding = get_embedder(self.config)
```

Replace the extractor block:
```python
        # 提取器：优先使用 LLM（如果配置了的话）
        if self.config.enable_llm_extraction:
            self.extractor: MemoryExtractor | LLMExtractor = LLMExtractor(self.config)
        else:
            self.extractor = MemoryExtractor(self.config)
```
with:
```python
        self.extractor = get_extractor(self.config)
```

- [ ] **Step 4: Run integration tests**

```bash
pytest tests/integration/test_memory_facade.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -m "not slow" -q
```
Expected: all `PASSED`, no regressions

- [ ] **Step 6: Commit**

```bash
git add src/neuralmem/core/memory.py tests/integration/test_memory_facade.py
git commit -m "feat(core): NeuralMem 通过 registry 动态选择 embedding 和 extractor"
```

---

## Task 13: Update pyproject.toml Extras

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add new extras**

In `pyproject.toml`, replace the `[project.optional-dependencies]` section with:

```toml
[project.optional-dependencies]
server    = ["fastapi>=0.110", "uvicorn>=0.30"]
openai    = ["openai>=1.0"]
ollama    = ["httpx>=0.27"]
reranker  = ["sentence-transformers>=3.0"]
cohere    = ["cohere>=5.0"]
gemini    = ["google-generativeai>=0.8"]
anthropic = ["anthropic>=0.30"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
    "httpx>=0.27",
]
all = ["neuralmem[server,openai,ollama,reranker,cohere,gemini,anthropic]"]
```

- [ ] **Step 2: Verify pyproject.toml is valid**

```bash
pip install --dry-run -e ".[all]" 2>&1 | head -20
```
Expected: dependency resolution output without errors

- [ ] **Step 3: Run lint check**

```bash
ruff check src/ tests/
```
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: 添加 cohere/gemini/anthropic PyPI extras"
```

---

## Task 14: Contract Tests for All Embedder Backends

**Files:**
- Create: `tests/contract/test_all_embedders_protocol.py`

- [ ] **Step 1: Write contract tests**

Create `tests/contract/test_all_embedders_protocol.py`:

```python
"""Verify all embedding backends satisfy EmbedderProtocol at the interface level."""
import sys
from unittest.mock import MagicMock
import pytest
from neuralmem.core.protocols import EmbedderProtocol
from neuralmem.core.config import NeuralMemConfig


def _cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _make_openai_embedder():
    mock = MagicMock()
    mock.OpenAI.return_value.embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 1536)]
    if "neuralmem.embedding.openai" in sys.modules:
        del sys.modules["neuralmem.embedding.openai"]
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {"openai": mock}):
        from neuralmem.embedding.openai import OpenAIEmbedding
        return OpenAIEmbedding(_cfg(openai_api_key="sk-test"))


def _make_cohere_embedder():
    mock = MagicMock()
    mock.ClientV2.return_value.embed.return_value.embeddings.float_ = [[0.1] * 1024]
    if "neuralmem.embedding.cohere" in sys.modules:
        del sys.modules["neuralmem.embedding.cohere"]
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {"cohere": mock}):
        from neuralmem.embedding.cohere import CohereEmbedding
        return CohereEmbedding(_cfg(cohere_api_key="ck-test"))


def _make_gemini_embedder():
    mock = MagicMock()
    mock.embed_content.return_value = {"embedding": [0.1] * 768}
    if "neuralmem.embedding.gemini" in sys.modules:
        del sys.modules["neuralmem.embedding.gemini"]
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"google.generativeai": mock, "google": MagicMock()}
    ):
        from neuralmem.embedding.gemini import GeminiEmbedding
        return GeminiEmbedding(_cfg(gemini_api_key="gk-test"))


def _make_hf_embedder():
    from neuralmem.embedding.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(_cfg(hf_api_key="hf-test"))


def _make_azure_embedder():
    mock = MagicMock()
    mock.AzureOpenAI.return_value.embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 1536)]
    if "neuralmem.embedding.azure_openai" in sys.modules:
        del sys.modules["neuralmem.embedding.azure_openai"]
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {"openai": mock}):
        from neuralmem.embedding.azure_openai import AzureOpenAIEmbedding
        return AzureOpenAIEmbedding(_cfg(azure_endpoint="https://x.openai.azure.com", azure_api_key="k"))


def _make_local_embedder():
    from neuralmem.embedding.local import LocalEmbedding
    return LocalEmbedding(_cfg())


@pytest.mark.parametrize("factory", [
    _make_local_embedder,
    _make_hf_embedder,
])
def test_embedder_satisfies_protocol(factory):
    """Each backend must be an instance of EmbedderProtocol (runtime-checkable)."""
    embedder = factory()
    assert isinstance(embedder, EmbedderProtocol), (
        f"{type(embedder).__name__} does not satisfy EmbedderProtocol"
    )
    assert isinstance(embedder.dimension, int)
    assert embedder.dimension > 0
    assert hasattr(embedder, "encode")
    assert hasattr(embedder, "encode_one")
    assert embedder.encode([]) == []
```

- [ ] **Step 2: Run contract tests**

```bash
pytest tests/contract/test_all_embedders_protocol.py -v
```
Expected: all `PASSED`

- [ ] **Step 3: Commit**

```bash
git add tests/contract/test_all_embedders_protocol.py
git commit -m "test(contract): 添加所有 embedding backend 的 Protocol 一致性测试"
```

---

## Task 15: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add provider table to CLAUDE.md**

In `CLAUDE.md`, in the `src/neuralmem/embedding/` section, replace the existing list with:

```markdown
├── embedding/
│   ├── base.py          (EmbeddingBackend ABC — all backends inherit this)
│   ├── registry.py      (factory: embedding_provider string → backend instance)
│   ├── local.py         (FastEmbed, default)
│   ├── openai.py        (OpenAI text-embedding-3-*, requires neuralmem[openai])
│   ├── cohere.py        (Cohere embed-multilingual-v3.0, requires neuralmem[cohere])
│   ├── gemini.py        (Google text-embedding-004, requires neuralmem[gemini])
│   ├── huggingface.py   (HF Inference API, uses httpx — no extra needed)
│   └── azure_openai.py  (Azure OpenAI, requires neuralmem[openai])
├── extraction/
│   ├── base_llm_extractor.py  (ABC with shared JSON-parse + fallback logic)
│   ├── extractor_registry.py  (factory: llm_extractor string → extractor instance)
│   ├── extractor.py           (rule-based, default)
│   ├── llm_extractor.py       (Ollama, llm_extractor="ollama")
│   ├── openai_extractor.py    (OpenAI gpt-4o-mini, llm_extractor="openai")
│   ├── anthropic_extractor.py (Anthropic claude-haiku, llm_extractor="anthropic")
│   └── entity_resolver.py
```

Also add a **Provider Selection** section after the Configuration section:

```markdown
### Provider Selection Quick Reference

| `embedding_provider` | Extra | Default model | Dim |
|---|---|---|---|
| `local` (default) | — | all-MiniLM-L6-v2 | 384 |
| `openai` | `neuralmem[openai]` | text-embedding-3-small | 1536 |
| `cohere` | `neuralmem[cohere]` | embed-multilingual-v3.0 | 1024 |
| `gemini` | `neuralmem[gemini]` | text-embedding-004 | 768 |
| `huggingface` | — | BAAI/bge-m3 | 1024 |
| `azure_openai` | `neuralmem[openai]` | text-embedding-3-small | 1536 |

| `llm_extractor` | Extra | Model |
|---|---|---|
| `none` (default) | — | rule-based |
| `ollama` | — | llama3.2:3b (local) |
| `openai` | `neuralmem[openai]` | gpt-4o-mini |
| `anthropic` | `neuralmem[anthropic]` | claude-haiku-4-5-20251001 |
```

- [ ] **Step 2: Run full test suite one final time**

```bash
pytest tests/ -m "not slow" -q --tb=short
```
Expected: all `PASSED`

- [ ] **Step 3: Run ruff lint**

```bash
ruff check src/ tests/
```
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: 更新 CLAUDE.md，添加 provider 选择快速参考表"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ 4 new embedding backends: Cohere (T4), Gemini (T5), HuggingFace (T6), Azure OpenAI (T7)
- ✅ OpenAI embedding (T3, was missing from existing code)
- ✅ 2 new LLM extractors: OpenAI (T9), Anthropic (T10)
- ✅ Registry pattern for both embedding and extraction (T2, T11)
- ✅ NeuralMemConfig extended with all new fields + backward-compat (T1)
- ✅ All SDKs behind lazy import with helpful error messages
- ✅ No real API calls in tests — all mocked
- ✅ pyproject.toml extras (T13)
- ✅ Contract tests (T14)
- ✅ CLAUDE.md updated (T15)

**Type consistency:**
- `get_embedder()` returns `EmbeddingBackend` — all backends implement `dimension`, `encode()`, `encode_one()`
- `get_extractor()` returns `MemoryExtractor` — all extractors implement `extract()`
- `BaseLLMExtractor._call_llm()` takes `str`, returns `str` — consistent across T9, T10
- `NeuralMemConfig.llm_extractor` field name used consistently across T1, T11, T12
