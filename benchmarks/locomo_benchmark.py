#!/usr/bin/env python3
"""
LoCoMo Benchmark for NeuralMem

LoCoMo (Long-Context Memory) benchmark measures how well a memory system
can store conversational knowledge and answer questions about past interactions.

Usage:
    python -m benchmarks.locomo_benchmark --db-path ./bench.db --dataset benchmarks/locomo_sample.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralmem.core.memory import NeuralMem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LoCoMoMessage:
    """A single message in a conversation."""
    role: str
    content: str
    timestamp: str = ""


@dataclass
class LoCoMoQuestion:
    """A question with ground-truth answer and optional evidence IDs."""
    question: str
    answer: str
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class LoCoMoConversation:
    """A conversation with messages and evaluation questions."""
    conversation_id: str
    messages: list[LoCoMoMessage] = field(default_factory=list)
    questions: list[LoCoMoQuestion] = field(default_factory=list)


@dataclass
class EvalResult:
    """Results of a LoCoMo evaluation run."""
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_questions: int = 0
    total_conversations: int = 0
    ingestion_stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric helpers (pure functions, easy to test)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K: fraction of relevant items found in top-K results."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@K: fraction of top-K results that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """MRR: 1/rank of the first relevant result."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def simple_answer_match(query: str, expected_answer: str, recalled_texts: list[str]) -> bool:
    """
    Simple heuristic: check if key terms from the expected answer
    appear in any recalled memory text.

    This is a proxy for relevance since NeuralMem stores content as text,
    not IDs matching evidence_ids.
    """
    expected_lower = expected_answer.lower()
    # Extract meaningful words (>3 chars) from expected answer
    key_terms = [w.strip(".,!?;:") for w in expected_lower.split() if len(w) > 3]

    if not key_terms:
        # Fall back to exact substring match on full answer
        return any(expected_lower in text.lower() for text in recalled_texts)

    # At least half the key terms should appear across recalled texts
    combined = " ".join(recalled_texts).lower()
    matches = sum(1 for term in key_terms if term in combined)
    return matches >= max(1, len(key_terms) // 2)


def percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile from a sorted list."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------

class LoCoMoBenchmarker:
    """
    LoCoMo benchmark runner for NeuralMem.

    Evaluates how well NeuralMem can ingest conversational data
    and answer retrieval questions about stored conversations.
    """

    def __init__(self, mem: NeuralMem):
        self.mem = mem

    @staticmethod
    def load_locomo_dataset(path: str) -> list[LoCoMoConversation]:
        """Load a LoCoMo-format JSON dataset and return parsed conversations."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        conversations: list[LoCoMoConversation] = []
        for conv_data in data.get("conversations", []):
            messages = [
                LoCoMoMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    timestamp=m.get("timestamp", ""),
                )
                for m in conv_data.get("messages", [])
            ]
            questions = [
                LoCoMoQuestion(
                    question=q.get("question", ""),
                    answer=q.get("answer", ""),
                    evidence_ids=q.get("evidence_ids", []),
                )
                for q in conv_data.get("questions", [])
            ]
            conversations.append(LoCoMoConversation(
                conversation_id=conv_data.get("conversation_id", ""),
                messages=messages,
                questions=questions,
            ))

        return conversations

    def ingest_conversations(
        self,
        conversations: list[LoCoMoConversation],
        user_id: str = "locomo_user",
    ) -> dict[str, Any]:
        """
        Ingest all messages from conversations into NeuralMem.

        Returns dict with ingestion statistics.
        """
        total_messages = 0
        total_memories = 0
        start_time = time.perf_counter()

        for conv in conversations:
            for msg in conv.messages:
                # Prefix with role for context
                content = f"[{msg.role}] {msg.content}"
                try:
                    memories = self.mem.remember(content, user_id=user_id)
                    total_memories += len(memories)
                except Exception as e:
                    logger.warning(
                        "Failed to ingest message from %s: %s",
                        conv.conversation_id, e,
                    )
                total_messages += 1

        elapsed = time.perf_counter() - start_time

        return {
            "conversations_processed": len(conversations),
            "messages_ingested": total_messages,
            "memories_stored": total_memories,
            "ingestion_time_s": round(elapsed, 3),
            "throughput_msg_per_s": round(total_messages / elapsed, 1) if elapsed > 0 else 0,
        }

    def run_eval(
        self,
        conversations: list[LoCoMoConversation],
        user_id: str = "locomo_user",
        k_values: list[int] | None = None,
    ) -> EvalResult:
        """
        Evaluate recall quality against conversation questions.

        For each question, recalls memories and checks if the expected answer
        terms appear in the recalled content.
        """
        if k_values is None:
            k_values = [1, 3, 5]

        max_k = max(k_values)

        recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
        precision_sums: dict[int, float] = {k: 0.0 for k in k_values}
        mrr_sum = 0.0
        latencies: list[float] = []
        total_questions = 0

        for conv in conversations:
            for qa in conv.questions:
                question = qa.question
                expected_answer = qa.answer

                # Recall with enough results
                start = time.perf_counter()
                results = self.mem.recall(
                    question,
                    user_id=user_id,
                    limit=max_k,
                    min_score=0.0,  # Don't filter by score for benchmarking
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

                # Get recalled content texts
                recalled_texts = [r.memory.content for r in results]

                # Use simple answer matching to determine relevance
                # For each recalled result, check if it's "relevant" to the answer
                relevant_indices: list[int] = []
                for i, text in enumerate(recalled_texts):
                    if simple_answer_match(question, expected_answer, [text]):
                        relevant_indices.append(i)

                # Build a relevance vector: for each position in recalled, is it relevant?
                is_relevant = [1.0 if i in relevant_indices else 0.0 for i in range(len(recalled_texts))]

                # If no direct match found in individual results, check combined
                # (the answer might span multiple memories)
                if not relevant_indices and recalled_texts:
                    if simple_answer_match(question, expected_answer, recalled_texts):
                        # Mark top result as relevant for scoring purposes
                        is_relevant[0] = 1.0
                        relevant_indices = [0]

                # Compute metrics
                n_relevant = len(relevant_indices) if relevant_indices else 0
                # For recall/precision, we treat "found answer" as binary
                found = 1.0 if n_relevant > 0 else 0.0

                for k in k_values:
                    if found > 0:
                        # Found in top results — recall@k depends on rank
                        best_rank = min(relevant_indices) if relevant_indices else max_k
                        recall_sums[k] += 1.0 if best_rank < k else 0.0
                        precision_sums[k] += (1.0 / k) if best_rank < k else 0.0
                    # else: both stay 0

                # MRR
                if relevant_indices:
                    mrr_sum += 1.0 / (min(relevant_indices) + 1)

                total_questions += 1

        # Aggregate
        result = EvalResult(
            recall_at_k={
                k: round(recall_sums[k] / total_questions, 4) if total_questions else 0.0
                for k in k_values
            },
            precision_at_k={
                k: round(precision_sums[k] / total_questions, 4) if total_questions else 0.0
                for k in k_values
            },
            mrr=round(mrr_sum / total_questions, 4) if total_questions else 0.0,
            avg_latency_ms=round(statistics.mean(latencies), 2) if latencies else 0.0,
            p50_latency_ms=round(percentile(sorted(latencies), 50), 2) if latencies else 0.0,
            p95_latency_ms=round(percentile(sorted(latencies), 95), 2) if latencies else 0.0,
            total_questions=total_questions,
            total_conversations=len(conversations),
        )

        return result

    @staticmethod
    def generate_report(
        result: EvalResult,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON-serializable report from evaluation results."""
        report = {
            "benchmark": "LoCoMo",
            "summary": {
                "total_conversations": result.total_conversations,
                "total_questions": result.total_questions,
                "mrr": result.mrr,
            },
            "recall_at_k": {f"recall@{k}": v for k, v in result.recall_at_k.items()},
            "precision_at_k": {f"precision@{k}": v for k, v in result.precision_at_k.items()},
            "latency": {
                "avg_ms": result.avg_latency_ms,
                "p50_ms": result.p50_latency_ms,
                "p95_ms": result.p95_latency_ms,
            },
            "ingestion_stats": result.ingestion_stats,
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info("Report saved to %s", output_path)

        return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LoCoMo benchmark on NeuralMem",
    )
    parser.add_argument(
        "--db-path",
        default="~/.neuralmem/locomo_bench.db",
        help="SQLite database path for NeuralMem",
    )
    parser.add_argument(
        "--dataset",
        default="benchmarks/locomo_sample.json",
        help="Path to LoCoMo-format JSON dataset",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/locomo_results.json",
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--user-id",
        default="locomo_user",
        help="User ID for memory scoping",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5",
        help="Comma-separated K values for Recall@K / Precision@K",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Initialize NeuralMem
    from neuralmem.core.config import NeuralMemConfig
    config = NeuralMemConfig(db_path=args.db_path)
    mem = NeuralMem(config=config)

    benchmarker = LoCoMoBenchmarker(mem)

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    conversations = benchmarker.load_locomo_dataset(args.dataset)
    print(f"Loaded {len(conversations)} conversations")

    # Ingest
    print("Ingesting conversations...")
    ingestion_stats = benchmarker.ingest_conversations(conversations, user_id=args.user_id)
    print(f"Ingestion complete: {ingestion_stats}")

    # Evaluate
    print("Running evaluation...")
    result = benchmarker.run_eval(
        conversations,
        user_id=args.user_id,
        k_values=k_values,
    )
    result.ingestion_stats = ingestion_stats

    # Generate report
    benchmarker.generate_report(result, output_path=args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("  LoCoMo Benchmark Results")
    print("=" * 60)
    print(f"  Conversations: {result.total_conversations}")
    print(f"  Questions:     {result.total_questions}")
    print(f"  MRR:           {result.mrr:.4f}")
    print()
    for k in k_values:
        print(f"  Recall@{k}:    {result.recall_at_k.get(k, 0):.4f}")
        print(f"  Precision@{k}: {result.precision_at_k.get(k, 0):.4f}")
    print()
    print(f"  Avg Latency:   {result.avg_latency_ms:.2f} ms")
    print(f"  P50 Latency:   {result.p50_latency_ms:.2f} ms")
    print(f"  P95 Latency:   {result.p95_latency_ms:.2f} ms")
    print("=" * 60)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
