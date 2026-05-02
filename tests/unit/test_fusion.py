"""RRF 融合单元测试"""
from __future__ import annotations

import pytest

from neuralmem.retrieval.fusion import RankedItem, RRFMerger


def test_rrf_single_strategy():
    merger = RRFMerger(k=60)
    results = merger.merge({
        "semantic": [
            RankedItem("a", 0.9, "semantic"),
            RankedItem("b", 0.7, "semantic"),
        ]
    })
    assert results[0][0] == "a"
    assert results[0][1] == pytest.approx(1.0)  # 归一化后最高分为 1.0


def test_rrf_two_strategies_agree():
    merger = RRFMerger(k=60)
    results = merger.merge({
        "semantic": [RankedItem("a", 0.9, "s"), RankedItem("b", 0.5, "s")],
        "keyword": [RankedItem("a", 0.8, "k"), RankedItem("c", 0.6, "k")],
    })
    # "a" 在两个策略中都排第一，应该得分最高
    assert results[0][0] == "a"


def test_rrf_empty_input():
    merger = RRFMerger()
    assert merger.merge({}) == []


def test_rrf_scores_normalized():
    merger = RRFMerger(k=60)
    results = merger.merge({
        "s": [RankedItem("a", 1.0, "s"), RankedItem("b", 0.5, "s")],
    })
    scores = [s for _, s in results]
    assert max(scores) == pytest.approx(1.0)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_rrf_k_affects_spread():
    # 不同 k 值不影响相对顺序（只影响分数差异）
    m1 = RRFMerger(k=1)
    m2 = RRFMerger(k=100)
    items = {"s": [RankedItem("a", 0.9, "s"), RankedItem("b", 0.1, "s")]}
    r1 = m1.merge(items)
    r2 = m2.merge(items)
    assert r1[0][0] == r2[0][0] == "a"


def test_rrf_deduplicates_across_strategies():
    merger = RRFMerger(k=60)
    # 同一个 id 出现在两个策略中，不应该重复出现在结果里
    results = merger.merge({
        "semantic": [RankedItem("a", 0.9, "s"), RankedItem("b", 0.5, "s")],
        "keyword": [RankedItem("a", 0.8, "k"), RankedItem("b", 0.6, "k")],
    })
    ids = [r[0] for r in results]
    assert len(ids) == len(set(ids))  # 无重复
