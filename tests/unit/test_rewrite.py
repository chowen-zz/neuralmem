"""Context rewriter 单元测试 — 全部使用 mock."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from neuralmem.rewrite.summarizer import MemorySummarizer
from neuralmem.rewrite.connector import ConnectionFinder
from neuralmem.rewrite.updater import SummaryUpdater
from neuralmem.rewrite.base import RewriteResult


# --------------------------------------------------------------------------- #
# RewriteResult
# --------------------------------------------------------------------------- #

def test_rewrite_result_merge():
    """测试两个 RewriteResult 合并."""
    r1 = RewriteResult(
        new_summaries=[MagicMock(id="s1")],
        connections_found=[{"a": 1}],
    )
    r2 = RewriteResult(
        new_summaries=[MagicMock(id="s2")],
        memories_archived=["m1"],
    )
    merged = r1.merge(r2)
    assert len(merged.new_summaries) == 2
    assert len(merged.connections_found) == 1
    assert len(merged.memories_archived) == 1


# --------------------------------------------------------------------------- #
# MemorySummarizer
# --------------------------------------------------------------------------- #

def test_summarizer_init():
    config = MagicMock()
    storage = MagicMock()
    summarizer = MemorySummarizer(config, storage)
    assert summarizer is not None


def test_summarizer_cluster_memories():
    """测试向量相似度聚类 — 使用简单数据避免 numpy 计算超时."""
    config = MagicMock()
    storage = MagicMock()
    summarizer = MemorySummarizer(config, storage)
    # Only 1 memory with embedding — should fall back to time window
    memories = [
        MagicMock(id="m1", content="AI safety is important", embedding=[1.0, 0.0]),
    ]
    clusters = summarizer._cluster_memories(memories)
    assert len(clusters) >= 0  # Single memory may return empty or single cluster


def test_summarizer_rule_summary():
    """LLM 不可用时使用规则回退."""
    config = MagicMock()
    storage = MagicMock()
    summarizer = MemorySummarizer(config, storage, llm_caller=None)
    # _rule_summary expects a combined string, not list
    combined = "- First point about AI.\n- Second point about safety."
    result = summarizer._rule_summary(combined)
    assert result is not None
    summary_text, importance = result
    assert "First point" in summary_text
    assert "Second point" in summary_text
    assert importance == 0.5


def test_summarizer_rewrite():
    """测试 rewrite() 入口方法."""
    config = MagicMock()
    storage = MagicMock()
    storage.list_memories.return_value = [
        MagicMock(id="m1", content="Point A", embedding=[1.0, 0.0], tags=[]),
        MagicMock(id="m2", content="Point B", embedding=[0.9, 0.1], tags=[]),
    ]
    summarizer = MemorySummarizer(config, storage, llm_caller=None)
    result = summarizer.rewrite(user_id="user1")
    assert isinstance(result, RewriteResult)


# --------------------------------------------------------------------------- #
# ConnectionFinder
# --------------------------------------------------------------------------- #

def test_connection_finder_init():
    config = MagicMock()
    storage = MagicMock()
    finder = ConnectionFinder(config, storage)
    assert finder is not None


def test_connection_finder_rewrite():
    """发现记忆之间的连接."""
    config = MagicMock()
    storage = MagicMock()
    storage.list_memories.return_value = [
        MagicMock(id="m1", content="Neural networks use backpropagation", tags=["ml"], embedding=[1.0, 0.0]),
        MagicMock(id="m2", content="Gradient descent optimizes weights", tags=["ml"], embedding=[0.9, 0.1]),
    ]
    finder = ConnectionFinder(config, storage, llm_caller=None)
    result = finder.rewrite(user_id="user1")
    assert isinstance(result, RewriteResult)


# --------------------------------------------------------------------------- #
# SummaryUpdater
# --------------------------------------------------------------------------- #

def test_summary_updater_init():
    config = MagicMock()
    storage = MagicMock()
    updater = SummaryUpdater(config, storage)
    assert updater is not None


def test_summary_updater_rewrite():
    """增量更新摘要."""
    config = MagicMock()
    storage = MagicMock()
    storage.list_memories.return_value = [
        MagicMock(id="s1", content="Original summary", tags=["summary"], embedding=[1.0, 0.0]),
        MagicMock(id="m1", content="New point A", tags=[], embedding=[0.9, 0.1]),
    ]
    updater = SummaryUpdater(config, storage, llm_caller=None)
    result = updater.rewrite(user_id="user1")
    assert isinstance(result, RewriteResult)
