"""FastEmbed 本地 ONNX Embedding 实现"""
from __future__ import annotations
import logging
import threading
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import EmbeddingError
from neuralmem.embedding.base import EmbeddingBackend

if TYPE_CHECKING:
    from fastembed import TextEmbedding

_logger = logging.getLogger(__name__)

_KNOWN_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-small-zh-v1.5": 512,
    "nomic-ai/nomic-embed-text-v1": 768,
}


class LocalEmbedding(EmbeddingBackend):
    """基于 FastEmbed 的本地 ONNX Embedding 后端。

    特性：
    - 懒加载：首次调用 encode 时才下载/初始化模型
    - 线程安全：双重检查锁定保护懒加载
    - 错误包装：所有 fastembed 异常包装为 EmbeddingError
    """

    def __init__(self, config: NeuralMemConfig) -> None:
        self._config = config
        self._model_name = config.embedding_model
        self._cache_dir = Path("~/.cache/neuralmem/models").expanduser()
        self._model: "TextEmbedding | None" = None
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        """返回向量维度；初始化前通过模型名推断，未知模型默认 384"""
        return _KNOWN_DIMS.get(self._model_name, 384)

    def _get_model(self) -> "TextEmbedding":
        """懒加载模型（双重检查锁定）"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from fastembed import TextEmbedding

                        self._cache_dir.mkdir(parents=True, exist_ok=True)
                        _logger.info("Loading embedding model: %s", self._model_name)
                        self._model = TextEmbedding(
                            model_name=self._model_name,
                            cache_dir=str(self._cache_dir),
                        )
                    except Exception as e:
                        raise EmbeddingError(
                            f"Failed to load model {self._model_name}: {e}"
                        ) from e
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文本为浮点向量列表"""
        if not texts:
            return []
        try:
            model = self._get_model()
            embeddings = list(model.embed(list(texts)))
            return [emb.tolist() for emb in embeddings]
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Encoding failed: {e}") from e

    def encode_one(self, text: str) -> list[float]:
        """编码单条文本为浮点向量"""
        return self.encode([text])[0]
