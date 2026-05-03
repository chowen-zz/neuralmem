"""NeuralMem V1.1 上下文重写模块 — 定期扫描记忆库，将相关记忆合并为更高层次的摘要，并发现记忆之间的隐性连接。"""
from __future__ import annotations

from neuralmem.rewrite.base import ContextRewriter, RewriteResult
from neuralmem.rewrite.summarizer import MemorySummarizer
from neuralmem.rewrite.connector import ConnectionFinder
from neuralmem.rewrite.updater import SummaryUpdater

__all__ = [
    "ContextRewriter",
    "RewriteResult",
    "MemorySummarizer",
    "ConnectionFinder",
    "SummaryUpdater",
]
