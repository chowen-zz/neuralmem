"""运行 LongMemEval 基准测试"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)

# 示例数据（当无法加载真实数据集时使用）
_SAMPLE_CONTEXT = [
    "Alice is a software engineer who specializes in Python and machine learning.",
    "Bob is a frontend developer who works with React and TypeScript.",
    "The project NeuralMem uses SQLite for storage and FastEmbed for embeddings.",
    "Alice prefers working with PyTorch for deep learning tasks.",
    "Bob's favorite tool is Vite for frontend build tooling.",
    "The team uses PostgreSQL in production for the main database.",
    "Alice joined the company in 2023 and leads the AI infrastructure team.",
    "Bob has been working on the dashboard feature for the last two weeks.",
]

_SAMPLE_QA = [
    {"question": "What programming language does Alice specialize in?", "answer": "Python"},
    {"question": "What does Bob work with for frontend?", "answer": "React"},
    {"question": "What storage does NeuralMem use?", "answer": "SQLite"},
    {"question": "What ML framework does Alice prefer?", "answer": "PyTorch"},
    {"question": "What is Bob's favorite build tool?", "answer": "Vite"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark for NeuralMem")
    parser.add_argument("--data", type=str, help="Path to benchmark data JSON file")
    parser.add_argument("--sample", type=int, default=0, help="Use N sample QA pairs (0 = all)")
    parser.add_argument("--db", type=str, default="./benchmarks/results/bench.db")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    # 加载数据
    if args.data:
        data = json.loads(Path(args.data).read_text())
        context = data.get("context", [])
        qa_pairs = data.get("qa_pairs", [])
    else:
        _logger.info("Using sample data (pass --data to use real LongMemEval dataset)")
        context = _SAMPLE_CONTEXT
        qa_pairs = _SAMPLE_QA

    if args.sample > 0:
        qa_pairs = qa_pairs[: args.sample]

    _logger.info("Context size: %d texts, QA pairs: %d", len(context), len(qa_pairs))

    # 运行评测
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from benchmarks.longmemeval.adapter import LongMemEvalAdapter

    adapter = LongMemEvalAdapter(db_path=args.db)

    _logger.info("Loading context into NeuralMem...")
    stored = adapter.load_context(context)
    _logger.info("Stored %d memories", stored)

    _logger.info("Running evaluation...")
    summary = adapter.evaluate(qa_pairs)

    print("\n" + "=" * 50)
    print(summary)
    print("=" * 50)

    if args.output:
        import dataclasses
        Path(args.output).write_text(json.dumps(dataclasses.asdict(summary), indent=2))
        _logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
