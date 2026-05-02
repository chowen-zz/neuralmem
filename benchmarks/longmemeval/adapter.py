"""NeuralMem LongMemEval 适配器"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """单条基准测试结果"""
    question_id: str
    question: str
    expected_answer: str
    retrieved_memories: list[str] = field(default_factory=list)
    top_score: float = 0.0
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    mrr: float = 0.0


@dataclass
class BenchmarkSummary:
    """基准测试汇总结果"""
    total: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    avg_score: float = 0.0

    def __str__(self) -> str:
        return (
            f"LongMemEval Results (n={self.total})\n"
            f"  Recall@1: {self.recall_at_1:.1%}\n"
            f"  Recall@3: {self.recall_at_3:.1%}\n"
            f"  Recall@5: {self.recall_at_5:.1%}\n"
            f"  MRR:      {self.mrr:.3f}\n"
            f"  Avg Score:{self.avg_score:.3f}"
        )


class LongMemEvalAdapter:
    """
    将 NeuralMem 接入 LongMemEval 基准测试框架。

    用法:
        adapter = LongMemEvalAdapter(db_path="./bench.db")
        adapter.load_context(context_texts)  # 加载背景文档
        summary = adapter.evaluate(qa_pairs)  # 运行问答评测
    """

    def __init__(self, db_path: str = "./benchmarks/results/bench.db") -> None:
        from neuralmem import NeuralMem
        self.mem = NeuralMem(db_path=db_path)
        self._loaded = False

    def load_context(
        self,
        texts: list[str],
        user_id: str = "benchmark",
        batch_size: int = 50,
    ) -> int:
        """将背景文档存入记忆库"""
        stored = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                if text.strip():
                    memories = self.mem.remember(text, user_id=user_id)
                    stored += len(memories)
            _logger.info("Loaded %d / %d texts", min(i + batch_size, len(texts)), len(texts))
        self._loaded = True
        return stored

    def evaluate(
        self,
        qa_pairs: list[dict[str, str]],
        user_id: str = "benchmark",
        top_k: int = 5,
    ) -> BenchmarkSummary:
        """
        运行问答评测。

        qa_pairs 格式: [{"question": "...", "answer": "..."}]
        """
        results: list[BenchmarkResult] = []

        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            expected = qa.get("answer", qa.get("expected", "")).lower()

            retrieved = self.mem.recall(question, user_id=user_id, limit=top_k)
            contents = [r.memory.content.lower() for r in retrieved]
            scores = [r.score for r in retrieved]

            # 判断命中：expected 是否出现在检索结果内容中
            hits = [expected in c or c in expected for c in contents]

            result = BenchmarkResult(
                question_id=str(i),
                question=question,
                expected_answer=expected,
                retrieved_memories=contents,
                top_score=scores[0] if scores else 0.0,
                hit_at_1=any(hits[:1]),
                hit_at_3=any(hits[:3]),
                hit_at_5=any(hits[:5]),
                mrr=next(
                    (1.0 / (j + 1) for j, h in enumerate(hits) if h),
                    0.0,
                ),
            )
            results.append(result)

            if (i + 1) % 10 == 0:
                _logger.info("Evaluated %d / %d questions", i + 1, len(qa_pairs))

        return self._summarize(results)

    def _summarize(self, results: list[BenchmarkResult]) -> BenchmarkSummary:
        n = len(results)
        if n == 0:
            return BenchmarkSummary()

        return BenchmarkSummary(
            total=n,
            recall_at_1=sum(r.hit_at_1 for r in results) / n,
            recall_at_3=sum(r.hit_at_3 for r in results) / n,
            recall_at_5=sum(r.hit_at_5 for r in results) / n,
            mrr=sum(r.mrr for r in results) / n,
            avg_score=sum(r.top_score for r in results) / n,
        )
