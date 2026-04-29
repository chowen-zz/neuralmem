"""CrossEncoderReranker — sentence-transformers cross-encoder 实现"""
from __future__ import annotations
import logging
from neuralmem.core.types import Memory

_logger = logging.getLogger(__name__)
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Cross-Encoder 重排序。
    懒加载 sentence-transformers；未安装时自动降级（保持原顺序）。

    启用: pip install neuralmem[reranker]
    配置: NeuralMemConfig(enable_reranker=True)
    """

    def __init__(self) -> None:
        self._model: object = None  # None=未检测; False=不可用; model=可用

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(_MODEL_NAME)
                _logger.info("CrossEncoder loaded: %s", _MODEL_NAME)
            except ImportError:
                _logger.info(
                    "sentence-transformers not installed. "
                    "Run 'pip install neuralmem[reranker]' to enable reranking."
                )
                self._model = False
        return self._model if self._model is not False else None

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Memory, float]],
    ) -> list[tuple[str, float]]:
        """
        对候选记忆重排序。

        Args:
            query: 原始查询字符串
            candidates: [(Memory, rrf_score), ...] 按 rrf_score 降序排列

        Returns:
            [(memory_id, new_score), ...] 按 cross-encoder 分数降序
        """
        if not candidates:
            return []

        model = self._get_model()
        if model is None:
            return [(m.id, score) for m, score in candidates]

        if len(candidates) <= 1:
            return [(m.id, score) for m, score in candidates]

        pairs = [(query, m.content) for m, _ in candidates]
        scores = model.predict(pairs).tolist()  # type: ignore[attr-defined]

        ranked = sorted(
            zip([m.id for m, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked
