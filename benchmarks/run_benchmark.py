#!/usr/bin/env python3
"""
NeuralMem Performance Benchmark

Measures:
  1. remember() throughput (memories/second) for batch sizes 1, 10, 50, 100
  2. recall() latency percentiles (p50, p95, p99) with 100 and 1000 stored memories
  3. Batch vs sequential remember comparison

Uses a deterministic 4-dim mock embedder for reproducibility (no model download).

Run: python benchmarks/run_benchmark.py
"""
from __future__ import annotations

import hashlib
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.memory import NeuralMem


# ---------------------------------------------------------------------------
# Deterministic mock embedder (same logic as tests/conftest.py)
# ---------------------------------------------------------------------------

class MockEmbedder:
    """4-dim deterministic embedder — no model download needed."""

    dimension = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            v = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(4)]
            norm = sum(x ** 2 for x in v) ** 0.5 or 1.0
            results.append([x / norm for x in v])
        return results

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_memory_content(i: int) -> str:
    """Generate unique memory content for index i."""
    return f"Memory item {i}: user preference number {i} about topic {i % 20}"


def percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile from a sorted list."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a plain text table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "| " + " | ".join(c.ljust(w) for c, w in zip(row, col_widths)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_remember_throughput(mem: NeuralMem, batch_sizes: list[int]) -> None:
    """Benchmark remember() throughput for various batch sizes."""
    print("\n" + "=" * 60)
    print("  1. remember() Throughput")
    print("=" * 60)

    headers = ["Batch Size", "Total Time (s)", "Throughput (mem/s)", "Memories Stored"]
    rows = []

    for n in batch_sizes:
        # Fresh DB for each batch size
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "bench.db")
            cfg = NeuralMemConfig(db_path=db_path)
            m = NeuralMem(config=cfg, embedder=MockEmbedder())

            contents = [make_memory_content(i) for i in range(n)]
            start = time.perf_counter()
            total_stored = 0
            for content in contents:
                stored = m.remember(content)
                total_stored += len(stored)
            elapsed = time.perf_counter() - start

            throughput = total_stored / elapsed if elapsed > 0 else 0
            rows.append([
                str(n),
                f"{elapsed:.4f}",
                f"{throughput:.1f}",
                str(total_stored),
            ])

    print(format_table(headers, rows))


def bench_recall_latency(mem: NeuralMem, num_memories: int, num_queries: int = 50) -> None:
    """Benchmark recall() latency percentiles at a given memory count."""
    queries = [
        "user preferences",
        "technology stack",
        "deployment tools",
        "programming language",
        "development workflow",
        "favorite editor",
        "database choice",
        "testing framework",
        "frontend backend",
        "cloud provider",
    ]

    latencies: list[float] = []
    for i in range(num_queries):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        mem.recall(query)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    latencies.sort()
    return {
        "count": len(latencies),
        "mean_ms": statistics.mean(latencies) * 1000,
        "p50_ms": percentile(latencies, 50) * 1000,
        "p95_ms": percentile(latencies, 95) * 1000,
        "p99_ms": percentile(latencies, 99) * 1000,
        "min_ms": latencies[0] * 1000,
        "max_ms": latencies[-1] * 1000,
    }


def bench_recall() -> None:
    """Benchmark recall() latency with different memory store sizes."""
    print("\n" + "=" * 60)
    print("  2. recall() Latency")
    print("=" * 60)

    headers = ["Stored Memories", "Queries", "Mean (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)"]
    rows = []

    for num_mem in [100, 1000]:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "bench.db")
            cfg = NeuralMemConfig(db_path=db_path)
            m = NeuralMem(config=cfg, embedder=MockEmbedder())

            # Pre-populate
            print(f"  Populating {num_mem} memories...", end=" ", flush=True)
            for i in range(num_mem):
                m.remember(make_memory_content(i))
            print("done")

            stats = bench_recall_latency(m, num_mem)
            rows.append([
                str(num_mem),
                str(stats["count"]),
                f"{stats['mean_ms']:.2f}",
                f"{stats['p50_ms']:.2f}",
                f"{stats['p95_ms']:.2f}",
                f"{stats['p99_ms']:.2f}",
            ])

    print(format_table(headers, rows))


def bench_batch_vs_sequential() -> None:
    """Compare batch remember_batch() vs sequential remember() calls."""
    print("\n" + "=" * 60)
    print("  3. Batch vs Sequential remember()")
    print("=" * 60)

    batch_sizes = [10, 50, 100]
    headers = ["Size", "Sequential (s)", "Batch (s)", "Speedup"]
    rows = []

    for n in batch_sizes:
        contents = [make_memory_content(i) for i in range(n)]

        # Sequential: individual remember() calls
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "seq.db")
            cfg = NeuralMemConfig(db_path=db_path)
            m_seq = NeuralMem(config=cfg, embedder=MockEmbedder())

            start = time.perf_counter()
            for content in contents:
                m_seq.remember(content)
            seq_time = time.perf_counter() - start

        # Batch: remember_batch() call
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "batch.db")
            cfg = NeuralMemConfig(db_path=db_path)
            m_batch = NeuralMem(config=cfg, embedder=MockEmbedder())

            start = time.perf_counter()
            m_batch.remember_batch(contents)
            batch_time = time.perf_counter() - start

        speedup = seq_time / batch_time if batch_time > 0 else float("inf")
        rows.append([
            str(n),
            f"{seq_time:.4f}",
            f"{batch_time:.4f}",
            f"{speedup:.2f}x",
        ])

    print(format_table(headers, rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  NeuralMem Performance Benchmark")
    print("  (using deterministic 4-dim mock embedder)")
    print("=" * 60)

    bench_remember_throughput(None, batch_sizes=[1, 10, 50, 100])
    bench_recall()
    bench_batch_vs_sequential()

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
