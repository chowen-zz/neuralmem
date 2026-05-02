#!/usr/bin/env python3
"""
NeuralMem vs Mem0 — 5-Round Multi-Dimension Competitive Benchmark
=================================================================

Dimensions tested:
  D1: Write Throughput (remember/add)
  D2: Read Latency + Quality (recall/search precision/recall)
  D3: Memory Efficiency (storage size, dedup)
  D4: Concurrency (parallel read/write)
  D5: Retrieval Quality (precision@k, recall@k on labeled data)
  D6: Feature Completeness (capability matrix)

Runs 5 rounds, aggregates mean/std for each metric.
"""

import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Test Data ──────────────────────────────────────────────────────────────

SAMPLE_MEMORIES = [
    "用户偏好使用 Python 编程语言",
    "项目使用 PostgreSQL 作为主数据库",
    "部署环境是 AWS us-east-1 区域",
    "代码风格遵循 PEP 8 规范",
    "测试框架使用 pytest",
    "前端使用 React + TypeScript",
    "API 设计遵循 RESTful 原则",
    "日志级别设为 INFO",
    "缓存使用 Redis，TTL 为 3600 秒",
    "CI/CD 使用 GitHub Actions",
    "数据库连接池大小为 20",
    "使用 Docker Compose 管理本地开发环境",
    "项目使用 uv 作为包管理器",
    "代码审查需要至少两人批准",
    "生产环境使用 Kubernetes 部署",
    "监控使用 Prometheus + Grafana",
    "错误追踪使用 Sentry",
    "CDN 使用 CloudFront",
    "搜索功能使用 Elasticsearch",
    "消息队列使用 RabbitMQ",
    "文件存储使用 AWS S3",
    "认证使用 JWT + OAuth2",
    "时区统一使用 UTC",
    "文档使用 MkDocs 生成",
    "性能测试使用 Locust",
    "代码覆盖率要求 > 80%",
    "数据库迁移使用 Alembic",
    "API 版本使用 URL 路径版本号",
    "日志格式使用 JSON 结构化",
    "使用 Prettier 格式化前端代码",
    "Python 版本要求 3.10+",
    "使用 Ruff 进行代码检查",
    "项目使用 monorepo 结构",
    "API 限流设为 100 req/min",
    "使用 WebSocket 实现实时通知",
    "图片处理使用 Pillow",
    "机器学习使用 PyTorch",
    "NLP 处理使用 spaCy",
    "任务调度使用 Celery",
    "配置管理使用 pydantic-settings",
]

SEARCH_QUERIES = [
    ("数据库配置是什么", "项目使用 PostgreSQL 作为主数据库"),
    ("部署在哪里", "部署环境是 AWS us-east-1 区域"),
    ("测试用什么框架", "测试框架使用 pytest"),
    ("前端技术栈", "前端使用 React + TypeScript"),
    ("缓存方案", "缓存使用 Redis，TTL 为 3600 秒"),
    ("Python 代码风格", "代码风格遵循 PEP 8 规范"),
    ("CI/CD 工具", "CI/CD 使用 GitHub Actions"),
    ("监控方案", "监控使用 Prometheus + Grafana"),
    ("认证方式", "认证使用 JWT + OAuth2"),
    ("消息队列", "消息队列使用 RabbitMQ"),
    ("日志配置", "日志级别设为 INFO"),
    ("文件存储", "文件存储使用 AWS S3"),
    ("搜索引擎", "搜索功能使用 Elasticsearch"),
    ("容器化方案", "使用 Docker Compose 管理本地开发环境"),
    ("K8s 部署", "生产环境使用 Kubernetes 部署"),
]

EXTENDED_MEMORIES = []
for i in range(200):
    templates = [
        f"服务实例 {i} 的端口号为 {8000 + i}",
        f"用户 {i} 的邮箱是 user{i}@example.com",
        f"任务 {i} 的截止日期是 2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        f"配置项 config_{i} 的值为 {random.randint(1, 1000)}",
        f"日志条目 {i}: 系统运行正常，CPU 使用率 {random.randint(10, 90)}%",
        f"API 端点 /api/v1/resource_{i} 返回状态码 200",
        f"数据库表 table_{i} 有 {random.randint(100, 10000)} 行记录",
        f"缓存键 cache_{i} 的命中率为 {random.randint(50, 99)}%",
    ]
    EXTENDED_MEMORIES.append(templates[i % len(templates)])


def generate_memories(count: int) -> list[str]:
    """Generate `count` memories by cycling through samples + extended."""
    base = SAMPLE_MEMORIES + EXTENDED_MEMORIES
    result = []
    for i in range(count):
        result.append(base[i % len(base)])
    return result


# ── Benchmark Result Types ─────────────────────────────────────────────────

@dataclass
class RoundResult:
    dimension: str
    metric: str
    neuralmem_value: float
    mem0_value: float
    unit: str
    higher_is_better: bool = True


@dataclass
class BenchmarkResults:
    rounds: list[list[RoundResult]] = field(default_factory=list)

    def add_round(self, results: list[RoundResult]):
        self.rounds.append(results)

    def aggregate(self) -> dict[str, dict[str, dict[str, float]]]:
        """Aggregate across rounds: mean and std for each (dimension, metric)."""
        agg: dict[str, dict[str, dict[str, list[float]]]] = {}
        for round_results in self.rounds:
            for r in round_results:
                agg.setdefault(r.dimension, {}).setdefault(r.metric, {"nm": [], "m0": []})
                agg[r.dimension][r.metric]["nm"].append(r.neuralmem_value)
                agg[r.dimension][r.metric]["m0"].append(r.mem0_value)

        result = {}
        for dim, metrics in agg.items():
            result[dim] = {}
            for metric, vals in metrics.items():
                nm_mean = statistics.mean(vals["nm"])
                nm_std = statistics.stdev(vals["nm"]) if len(vals["nm"]) > 1 else 0
                m0_mean = statistics.mean(vals["m0"])
                m0_std = statistics.stdev(vals["m0"]) if len(vals["m0"]) > 1 else 0
                result[dim][metric] = {
                    "nm_mean": nm_mean, "nm_std": nm_std,
                    "m0_mean": m0_mean, "m0_std": m0_std,
                }
        return result


# ── NeuralMem Wrapper ─────────────────────────────────────────────────────

class NeuralMemBenchmark:
    def __init__(self, db_path: str):
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from neuralmem.core.config import NeuralMemConfig
        from neuralmem.core.memory import NeuralMem

        self.config = NeuralMemConfig(
            db_path=db_path,
            embedding_provider="local",
            llm_extractor="none",
        )
        self.mem = NeuralMem(config=self.config)

    def remember(self, text: str) -> float:
        start = time.perf_counter()
        self.mem.remember(text)
        return time.perf_counter() - start

    def recall(self, query: str, limit: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        results = self.mem.recall(query, limit=limit)
        elapsed = time.perf_counter() - start
        return [r.memory.content for r in results], elapsed

    def get_all(self) -> list[str]:
        memories = self.mem.storage.list_memories()
        return [m.content for m in memories]

    def close(self):
        pass


# ── Mem0 Wrapper ──────────────────────────────────────────────────────────

class Mem0Benchmark:
    def __init__(self, db_path: str):
        from mem0 import Memory
        from mem0.configs.base import MemoryConfig
        from mem0.embeddings.configs import EmbedderConfig
        from mem0.llms.configs import LlmConfig
        from mem0.vector_stores.configs import VectorStoreConfig

        config = MemoryConfig(
            embedder=EmbedderConfig(
                provider="huggingface",
                config={"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384},
            ),
            llm=LlmConfig(
                provider="openai",
                config={"model": "gpt-4o-mini", "api_key": "sk-fake-for-benchmark"},
            ),
            vector_store=VectorStoreConfig(
                provider="qdrant",
                config={"path": db_path, "collection_name": "benchmark", "embedding_model_dims": 384},
            ),
            version="v1.1",
        )
        self.mem = Memory(config=config)
        self.user_id = "benchmark_user"

    def remember(self, text: str) -> float:
        start = time.perf_counter()
        self.mem.add(text, user_id=self.user_id, infer=False)
        return time.perf_counter() - start

    def recall(self, query: str, limit: int = 5) -> tuple[list[str], float]:
        start = time.perf_counter()
        results = self.mem.search(query, filters={"user_id": self.user_id}, top_k=limit)
        elapsed = time.perf_counter() - start
        contents = [r["memory"] for r in results.get("results", [])]
        return contents, elapsed

    def get_all(self) -> list[str]:
        results = self.mem.get_all(filters={"user_id": self.user_id}, top_k=10000)
        return [r["memory"] for r in results.get("results", [])]

    def close(self):
        self.mem.reset()


# ── Benchmark Dimensions ──────────────────────────────────────────────────

def benchmark_write_throughput(nm: NeuralMemBenchmark, m0: Mem0Benchmark, count: int = 100) -> list[RoundResult]:
    """D1: Write throughput — time to add N memories."""
    memories = generate_memories(count)

    # NeuralMem
    nm_times = []
    for m in memories:
        nm_times.append(nm.remember(m))
    nm_total = sum(nm_times)
    nm_throughput = count / nm_total if nm_total > 0 else 0

    # Mem0
    m0_times = []
    for m in memories:
        m0_times.append(m0.remember(m))
    m0_total = sum(m0_times)
    m0_throughput = count / m0_total if m0_total > 0 else 0

    return [
        RoundResult("D1_Write", "throughput_mem_per_sec", nm_throughput, m0_throughput, "mem/s", True),
        RoundResult("D1_Write", "total_time_sec", nm_total, m0_total, "s", False),
        RoundResult("D1_Write", "p50_latency_ms",
                     statistics.median(nm_times) * 1000, statistics.median(m0_times) * 1000, "ms", False),
        RoundResult("D1_Write", "p95_latency_ms",
                     sorted(nm_times)[int(len(nm_times) * 0.95)] * 1000,
                     sorted(m0_times)[int(len(m0_times) * 0.95)] * 1000, "ms", False),
    ]


def benchmark_read_latency(nm: NeuralMemBenchmark, m0: Mem0Benchmark) -> list[RoundResult]:
    """D2: Read latency — search latency on known queries."""
    nm_latencies = []
    m0_latencies = []

    for query, _ in SEARCH_QUERIES:
        _, nm_lat = nm.recall(query)
        nm_latencies.append(nm_lat)

        _, m0_lat = m0.recall(query)
        m0_latencies.append(m0_lat)

    return [
        RoundResult("D2_Read", "p50_latency_ms",
                     statistics.median(nm_latencies) * 1000, statistics.median(m0_latencies) * 1000, "ms", False),
        RoundResult("D2_Read", "p95_latency_ms",
                     sorted(nm_latencies)[int(len(nm_latencies) * 0.95)] * 1000,
                     sorted(m0_latencies)[int(len(m0_latencies) * 0.95)] * 1000, "ms", False),
        RoundResult("D2_Read", "mean_latency_ms",
                     statistics.mean(nm_latencies) * 1000, statistics.mean(m0_latencies) * 1000, "ms", False),
    ]


def benchmark_retrieval_quality(nm: NeuralMemBenchmark, m0: Mem0Benchmark) -> list[RoundResult]:
    """D5: Retrieval quality — precision@3 and recall@5 on labeled queries."""
    nm_prec3_scores = []
    nm_recall5_scores = []
    m0_prec3_scores = []
    m0_recall5_scores = []

    for query, expected in SEARCH_QUERIES[:10]:
        # NeuralMem
        nm_results, _ = nm.recall(query, limit=5)
        if nm_results:
            nm_top3 = nm_results[:3]
            nm_prec3 = sum(1 for r in nm_top3 if expected in r or r in expected) / 3
            nm_in_top5 = any(expected in r or r in expected for r in nm_results)
        else:
            nm_prec3 = 0
            nm_in_top5 = False

        # Mem0
        m0_results, _ = m0.recall(query, limit=5)
        if m0_results:
            m0_top3 = m0_results[:3]
            m0_prec3 = sum(1 for r in m0_top3 if expected in r or r in expected) / 3
            m0_in_top5 = any(expected in r or r in expected for r in m0_results)
        else:
            m0_prec3 = 0
            m0_in_top5 = False

        nm_prec3_scores.append(nm_prec3)
        nm_recall5_scores.append(1.0 if nm_in_top5 else 0.0)
        m0_prec3_scores.append(m0_prec3)
        m0_recall5_scores.append(1.0 if m0_in_top5 else 0.0)

    return [
        RoundResult("D5_Quality", "precision_at_3",
                     statistics.mean(nm_prec3_scores), statistics.mean(m0_prec3_scores), "ratio", True),
        RoundResult("D5_Quality", "recall_at_5",
                     statistics.mean(nm_recall5_scores), statistics.mean(m0_recall5_scores), "ratio", True),
    ]


def benchmark_memory_efficiency(nm: NeuralMemBenchmark, m0: Mem0Benchmark, db_dir: str) -> list[RoundResult]:
    """D3: Memory efficiency — DB file size and record count."""
    # NeuralMem DB size
    nm_db = Path(db_dir) / "neuralmem" / "memory.db"
    nm_size = nm_db.stat().st_size if nm_db.exists() else 0

    # Mem0 DB size (qdrant storage)
    m0_db = Path(db_dir) / "mem0"
    m0_size = sum(f.stat().st_size for f in m0_db.rglob("*") if f.is_file()) if m0_db.exists() else 0

    # Record counts
    nm_count = len(nm.get_all())
    m0_count = len(m0.get_all())

    return [
        RoundResult("D3_Efficiency", "storage_size_mb", nm_size / (1024*1024), m0_size / (1024*1024), "MB", False),
        RoundResult("D3_Efficiency", "records_stored", nm_count, m0_count, "count", True),
        RoundResult("D3_Efficiency", "bytes_per_record",
                     (nm_size / nm_count if nm_count > 0 else 0),
                     (m0_size / m0_count if m0_count > 0 else 0), "bytes", False),
    ]


def benchmark_concurrency(nm: NeuralMemBenchmark, m0: Mem0Benchmark) -> list[RoundResult]:
    """D4: Concurrent read performance (4 threads)."""
    queries = [q for q, _ in SEARCH_QUERIES[:8]]

    # NeuralMem concurrent reads
    nm_results = []
    nm_lock = threading.Lock()

    def nm_worker(q):
        _, lat = nm.recall(q)
        with nm_lock:
            nm_results.append(lat)

    start = time.perf_counter()
    threads = [threading.Thread(target=nm_worker, args=(q,)) for q in queries]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    nm_concurrent_time = time.perf_counter() - start

    # Mem0 concurrent reads
    m0_results = []
    m0_lock = threading.Lock()

    def m0_worker(q):
        _, lat = m0.recall(q)
        with m0_lock:
            m0_results.append(lat)

    start = time.perf_counter()
    threads = [threading.Thread(target=m0_worker, args=(q,)) for q in queries]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    m0_concurrent_time = time.perf_counter() - start

    nm_qps = len(queries) / nm_concurrent_time if nm_concurrent_time > 0 else 0
    m0_qps = len(queries) / m0_concurrent_time if m0_concurrent_time > 0 else 0

    return [
        RoundResult("D4_Concurrent", "queries_per_sec", nm_qps, m0_qps, "q/s", True),
        RoundResult("D4_Concurrent", "total_time_sec", nm_concurrent_time, m0_concurrent_time, "s", False),
        RoundResult("D4_Concurrent", "p50_thread_latency_ms",
                     statistics.median(nm_results) * 1000 if nm_results else 0,
                     statistics.median(m0_results) * 1000 if m0_results else 0, "ms", False),
    ]


# ── Feature Matrix ────────────────────────────────────────────────────────

FEATURE_MATRIX = [
    ("向量语义检索", True, True),
    ("BM25 关键词检索", True, True),
    ("图谱关系检索", True, False),
    ("时间衰减检索", True, False),
    ("RRF 多策略融合", True, False),
    ("Cross-Encoder 重排", True, True),
    ("实体提取", True, True),
    ("知识图谱", True, False),
    ("记忆衰减/生命周期", True, False),
    ("记忆合并/去重", True, True),
    ("批量操作", True, True),
    ("记忆导出", True, False),
    ("冲突解决", True, False),
    ("带解释检索", True, False),
    ("MCP 原生支持", True, False),
    ("本地优先/零依赖", True, False),
    ("多后端嵌入模型", True, True),
    ("多 LLM 提取器", True, True),
    ("CLI 工具", True, True),
    ("TypeScript/npm SDK", True, True),
]


def benchmark_features() -> list[RoundResult]:
    """D6: Feature completeness — count available features."""
    nm_count = sum(1 for _, nm_has, _ in FEATURE_MATRIX if nm_has)
    m0_count = sum(1 for _, _, m0_has in FEATURE_MATRIX if m0_has)
    return [
        RoundResult("D6_Features", "feature_count", nm_count, m0_count, "count", True),
        RoundResult("D6_Features", "feature_pct",
                     nm_count / len(FEATURE_MATRIX) * 100, m0_count / len(FEATURE_MATRIX) * 100, "%", True),
    ]


# ── Main Runner ───────────────────────────────────────────────────────────

def run_round(round_num: int, base_dir: str, data_count: int = 100) -> list[RoundResult]:
    """Run one complete benchmark round."""
    round_dir = os.path.join(base_dir, f"round_{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    nm_db = os.path.join(round_dir, "neuralmem")
    m0_db = os.path.join(round_dir, "mem0")
    os.makedirs(nm_db, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ROUND {round_num} — {data_count} memories")
    print(f"{'='*60}")

    # Initialize
    print("  [init] NeuralMem...", end=" ", flush=True)
    nm = NeuralMemBenchmark(os.path.join(nm_db, "memory.db"))
    print("OK")

    print("  [init] Mem0...", end=" ", flush=True)
    m0 = Mem0Benchmark(os.path.join(m0_db, "qdrant"))
    print("OK")

    # D1: Write
    print(f"  [D1] Write throughput ({data_count} memories)...", end=" ", flush=True)
    d1 = benchmark_write_throughput(nm, m0, data_count)
    print("OK")

    # D2: Read latency
    print("  [D2] Read latency...", end=" ", flush=True)
    d2 = benchmark_read_latency(nm, m0)
    print("OK")

    # D3: Memory efficiency
    print("  [D3] Memory efficiency...", end=" ", flush=True)
    d3 = benchmark_memory_efficiency(nm, m0, round_dir)
    print("OK")

    # D4: Concurrency
    print("  [D4] Concurrency...", end=" ", flush=True)
    d4 = benchmark_concurrency(nm, m0)
    print("OK")

    # D5: Retrieval quality
    print("  [D5] Retrieval quality...", end=" ", flush=True)
    d5 = benchmark_retrieval_quality(nm, m0)
    print("OK")

    # D6: Features (static, only run once but we include for completeness)
    d6 = benchmark_features()

    # Cleanup
    try:
        m0.close()
    except Exception:
        pass

    results = d1 + d2 + d3 + d4 + d5 + d6
    return results


def print_report(bench: BenchmarkResults):
    """Print aggregated benchmark report."""
    agg = bench.aggregate()

    print("\n" + "=" * 80)
    print("  NEURALMEM vs MEM0 — 5-ROUND BENCHMARK RESULTS")
    print("=" * 80)

    for dim, metrics in agg.items():
        print(f"\n  {dim}")
        print(f"  {'─' * 70}")
        print(f"  {'Metric':<30} {'NeuralMem':>14} {'Mem0':>14} {'Winner':>10}")
        print(f"  {'─' * 70}")

        for metric, vals in metrics.items():
            nm_str = f"{vals['nm_mean']:.2f} ±{vals['nm_std']:.2f}"
            m0_str = f"{vals['m0_mean']:.2f} ±{vals['m0_std']:.2f}"

            # Determine winner (check if higher is better by looking at the RoundResult)
            higher_better = True
            for round_results in bench.rounds:
                for r in round_results:
                    if r.dimension == dim and r.metric == metric:
                        higher_better = r.higher_is_better
                        break

            if higher_better:
                winner = "NeuralMem" if vals["nm_mean"] > vals["m0_mean"] else "Mem0"
            else:
                winner = "NeuralMem" if vals["nm_mean"] < vals["m0_mean"] else "Mem0"

            # Tie threshold
            diff_pct = abs(vals["nm_mean"] - vals["m0_mean"]) / max(abs(vals["nm_mean"]), abs(vals["m0_mean"]), 0.001) * 100
            if diff_pct < 5:
                winner = "Tie"

            print(f"  {metric:<30} {nm_str:>14} {m0_str:>14} {winner:>10}")

    # Score summary
    print(f"\n  {'=' * 70}")
    print("  SCORE SUMMARY")
    print(f"  {'=' * 70}")

    nm_wins = 0
    m0_wins = 0
    ties = 0

    for dim, metrics in agg.items():
        for metric, vals in metrics.items():
            for round_results in bench.rounds:
                for r in round_results:
                    if r.dimension == dim and r.metric == metric:
                        higher_better = r.higher_is_better
                        break

            if higher_better:
                if vals["nm_mean"] > vals["m0_mean"]:
                    nm_wins += 1
                else:
                    m0_wins += 1
            else:
                if vals["nm_mean"] < vals["m0_mean"]:
                    nm_wins += 1
                else:
                    m0_wins += 1

    print(f"  NeuralMem wins: {nm_wins}")
    print(f"  Mem0 wins:      {m0_wins}")
    print(f"  Ties:           {ties}")
    print(f"  Total metrics:  {nm_wins + m0_wins + ties}")

    return agg


def main():
    """Main entry point — run 5 rounds."""
    base_dir = tempfile.mkdtemp(prefix="nm_vs_m0_benchmark_")
    print(f"Benchmark directory: {base_dir}")

    bench = BenchmarkResults()

    data_counts = [50, 100, 100, 100, 150]  # Different sizes per round

    for round_num in range(1, 6):
        try:
            results = run_round(round_num, base_dir, data_counts[round_num - 1])
            bench.add_round(results)
        except Exception as e:
            print(f"\n  ERROR in round {round_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print and save report
    agg = print_report(bench)

    # Save JSON results
    results_path = Path(__file__).parent / "results" / "mem0_comparison.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rounds": len(bench.rounds),
        "aggregated": {},
        "feature_matrix": FEATURE_MATRIX,
    }

    for dim, metrics in agg.items():
        output["aggregated"][dim] = {}
        for metric, vals in metrics.items():
            output["aggregated"][dim][metric] = vals

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {results_path}")

    # Cleanup temp dir
    try:
        shutil.rmtree(base_dir)
    except Exception:
        pass

    return output


if __name__ == "__main__":
    main()
