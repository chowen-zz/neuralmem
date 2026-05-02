"""Tests for the Ollama embedding backend with mocked httpx responses."""
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import EmbeddingError


def cfg(**kwargs):
    return NeuralMemConfig(db_path=":memory:", **kwargs)


def _mock_response(data):
    mock = MagicMock()
    mock.json.return_value = data
    mock.raise_for_status.return_value = None
    return mock


class TestOllamaEmbeddingBackend:

    def test_dimension_known_model_nomic(self):
        from neuralmem.embedding.ollama import OllamaEmbeddingBackend

        embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
        assert embedder.dimension == 768

    def test_dimension_known_model_mxbai(self):
        from neuralmem.embedding.ollama import OllamaEmbeddingBackend

        embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="mxbai-embed-large"))
        assert embedder.dimension == 1024

    def test_dimension_unknown_model_defaults_768(self):
        from neuralmem.embedding.ollama import OllamaEmbeddingBackend

        embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="custom-model"))
        assert embedder.dimension == 768

    def test_encode_single_text(self):
        resp = _mock_response({"model": "nomic-embed-text", "embeddings": [[0.1] * 768]})
        with patch("neuralmem.embedding.ollama.httpx.post", return_value=resp) as mock_post:
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
            result = embedder.encode(["hello world"])

            assert len(result) == 1
            assert len(result[0]) == 768
            mock_post.assert_called_once_with(
                "http://localhost:11434/api/embed",
                json={"model": "nomic-embed-text", "input": ["hello world"]},
                timeout=60.0,
            )

    def test_encode_batch_texts(self):
        embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        resp = _mock_response({"model": "nomic-embed-text", "embeddings": embeddings})
        with patch("neuralmem.embedding.ollama.httpx.post", return_value=resp):
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
            result = embedder.encode(["text one", "text two", "text three"])

            assert len(result) == 3
            assert result[0] == [0.1] * 768
            assert result[1] == [0.2] * 768
            assert result[2] == [0.3] * 768

    def test_encode_empty_returns_empty(self):
        from neuralmem.embedding.ollama import OllamaEmbeddingBackend

        embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
        assert embedder.encode([]) == []

    def test_encode_one(self):
        resp = _mock_response({"model": "nomic-embed-text", "embeddings": [[0.5] * 768]})
        with patch("neuralmem.embedding.ollama.httpx.post", return_value=resp):
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
            result = embedder.encode_one("single text")

            assert len(result) == 768
            assert result == [0.5] * 768

    def test_custom_base_url(self):
        resp = _mock_response({"model": "nomic-embed-text", "embeddings": [[0.1] * 768]})
        with patch("neuralmem.embedding.ollama.httpx.post", return_value=resp) as mock_post:
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(
                cfg(
                    ollama_embedding_model="nomic-embed-text",
                    ollama_url="http://gpu-server:9999",
                )
            )
            embedder.encode(["test"])

            mock_post.assert_called_once_with(
                "http://gpu-server:9999/api/embed",
                json={"model": "nomic-embed-text", "input": ["test"]},
                timeout=60.0,
            )

    def test_http_error_raises_embedding_error(self):
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch(
            "neuralmem.embedding.ollama.httpx.post",
            side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp),
        ):
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
            with pytest.raises(EmbeddingError, match="Ollama API error 500"):
                embedder.encode(["hello"])

    def test_connection_error_raises_embedding_error(self):
        import httpx

        with patch(
            "neuralmem.embedding.ollama.httpx.post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
            with pytest.raises(EmbeddingError, match="Cannot connect to Ollama"):
                embedder.encode(["hello"])

    def test_unexpected_response_raises_embedding_error(self):
        resp = _mock_response({"model": "nomic-embed-text", "embeddings": "not-a-list"})
        with patch("neuralmem.embedding.ollama.httpx.post", return_value=resp):
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
            with pytest.raises(EmbeddingError, match="Unexpected Ollama response"):
                embedder.encode(["hello"])

    def test_mismatched_embedding_count_raises(self):
        resp = _mock_response({"model": "nomic-embed-text", "embeddings": [[0.1] * 768]})
        with patch("neuralmem.embedding.ollama.httpx.post", return_value=resp):
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
            with pytest.raises(EmbeddingError, match="Ollama returned 1 embeddings for 2 texts"):
                embedder.encode(["text1", "text2"])

    def test_invalid_base_url_raises_config_error(self):
        from neuralmem.core.exceptions import ConfigError
        from neuralmem.embedding.ollama import OllamaEmbeddingBackend

        with pytest.raises(ConfigError, match="ollama_url must start with"):
            OllamaEmbeddingBackend(
                cfg(ollama_url="not-a-url", ollama_embedding_model="nomic-embed-text")
            )

    def test_strips_trailing_slash_from_url(self):
        resp = _mock_response({"model": "nomic-embed-text", "embeddings": [[0.1] * 768]})
        with patch("neuralmem.embedding.ollama.httpx.post", return_value=resp) as mock_post:
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(
                cfg(ollama_url="http://localhost:11434/", ollama_embedding_model="nomic-embed-text")
            )
            embedder.encode(["test"])

            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/embed"

    def test_returns_lists_not_tuples(self):
        resp = _mock_response({"model": "nomic-embed-text", "embeddings": [(0.1, 0.2, 0.3)]})
        with patch("neuralmem.embedding.ollama.httpx.post", return_value=resp):
            from neuralmem.embedding.ollama import OllamaEmbeddingBackend

            embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
            result = embedder.encode(["hi"])

            assert isinstance(result[0], list)
            assert result[0] == [0.1, 0.2, 0.3]

    def test_implements_embedder_protocol(self):
        """Verify OllamaEmbeddingBackend satisfies EmbedderProtocol."""
        from neuralmem.core.protocols import EmbedderProtocol
        from neuralmem.embedding.ollama import OllamaEmbeddingBackend

        embedder = OllamaEmbeddingBackend(cfg(ollama_embedding_model="nomic-embed-text"))
        assert isinstance(embedder, EmbedderProtocol)
