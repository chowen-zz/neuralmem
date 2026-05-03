"""AsyncEmbedder — 异步嵌入后端包装器，将同步 EmbeddingBackend 包装为 async API。

使用 asyncio.to_thread 将阻塞的编码操作（尤其是本地模型推理）卸载到线程池中执行。
提供批量编码的并发优化，支持 encode_batch 使用 asyncio.gather 并行处理。
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from neuralmem.embedding.base import EmbeddingBackend

_logger = logging.getLogger(__name__)


class AsyncEmbedder:
    """异步 Embedding 包装器 — 将同步 EmbeddingBackend 包装为 async 接口。

    所有底层编码操作通过 asyncio.to_thread 在线程池中执行，
    避免阻塞 asyncio 事件循环。提供批量并发编码优化。
    """

    def __init__(self, embedder: EmbeddingBackend) -> None:
        self._embedder = embedder

    @property
    def dimension(self) -> int:
        """向量维度（同步属性，直接返回）。"""
        return self._embedder.dimension

    async def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """异步批量编码文本为向量列表。"""
        if not texts:
            return []
        return await asyncio.to_thread(self._embedder.encode, texts)

    async def encode_one(self, text: str) -> list[float]:
        """异步编码单条文本为向量。"""
        return await asyncio.to_thread(self._embedder.encode_one, text)

    async def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_concurrency: int = 4,
    ) -> list[list[float]]:
        """分块并发批量编码 — 将 texts 分块后使用 asyncio.gather 并行处理。

        Args:
            texts: 待编码文本列表。
            batch_size: 每个子批次的大小。
            max_concurrency: 最大并发批次数（通过 semaphore 控制）。

        Returns:
            与输入顺序一致的向量列表。
        """
        if not texts:
            return []

        text_list = list(texts)
        if len(text_list) <= batch_size:
            return await self.encode(text_list)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _encode_chunk(chunk: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self.encode(chunk)

        chunks = [
            text_list[i : i + batch_size]
            for i in range(0, len(text_list), batch_size)
        ]

        results = await asyncio.gather(*[_encode_chunk(c) for c in chunks])

        # Flatten results maintaining order
        vectors: list[list[float]] = []
        for chunk_result in results:
            vectors.extend(chunk_result)
        return vectors

    @property
    def underlying(self) -> EmbeddingBackend:
        """返回底层的同步 EmbeddingBackend 实例。"""
        return self._embedder
