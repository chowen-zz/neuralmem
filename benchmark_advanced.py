#!/usr/bin/env python3
"""
NeuralMem 业界级压测套件
=========================
对标: VectorDBBench (Milvus/Zilliz), ANN-Benchmarks, Qdrant Benchmarks

测试维度:
  1. 吞吐量基准 — remember/recall QPS + P50/P95/P99 延迟
  2. Recall@K 精度 — 向量搜索准确性 vs ground truth
  3. 大规模稳定性 — 5000+ 记忆的写入/检索/存储完整性
  4. 并发混合负载 — 多线程读写 + 过滤 + 实体图遍历
  5. WAL checkpoint 压力 — 大量小事务触发 checkpoint
  6. 内存压力 — 大量记录下 RAM 使用增长
  7. 实体图密度 — 大量实体/关系下的图遍历性能
  8. 降级与恢复 — 存储异常后的恢复能力

指标 (对标 VectorDBBench):
  - QPS (queries per second)
  - Latency P50 / P95 / P99 (ms)
  - Recall@K accuracy
  - Throughput (vectors/sec)
  - Memory growth (MB)
  - Data integrity (%)
"""
from __future__ import annotations

import gc
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.memory import NeuralMem
from neuralmem.core.types import MemoryType

# ── Constants ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)


# ── Helpers ────────────────────────────────────────────────────────────────


class BenchResult:
    """Structured benchmark result with metrics."""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.passed = False
        self.error: str | None = None
        self.metrics: dict[str, float | str | int] = {}
        self.duration: float = 0.0

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        m = ", ".join(f"{k}={v}" for k, v in self.metrics.items())
        return f"[{status}] {self.name} ({self.duration:.2f}s) — {m}"


def percentile(data: list[float], p: float) -> float:
    """Calculate P-th percentile of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def measure_latencies(fn, iterations: int) -> list[float]:
    """Run fn() N times and return list of latencies in ms."""
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def make_mem(tmpdir: str, label: str = "") -> NeuralMem:
    db = os.path.join(tmpdir, f"bench_{label}_{int(time.time()*1000)}.db")
    cfg = NeuralMemConfig(db_path=db)
    return NeuralMem(config=cfg)


def get_process_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import resource
        # On macOS/Linux, ru_maxrss is in bytes on Linux, KB on macOS
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return ru / 1024 / 1024  # KB -> MB
        return ru / 1024 / 1024  # bytes -> MB
    except Exception:
        return 0.0


def run_bench(fn, name: str, category: str) -> BenchResult:
    """Run a benchmark function and capture result."""
    result = BenchResult(name, category)
    t0 = time.monotonic()
    try:
        metrics = fn()
        result.passed = True
        if metrics:
            result.metrics = metrics
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.metrics["error_detail"] = traceback.format_exc()
    result.duration = time.monotonic() - t0
    return result


# ── Benchmark Categories ───────────────────────────────────────────────────

# ==========================================================================
# Category 1: THROUGHPUT BENCHMARK (对标 VectorDBBench)
# 衡量 remember/recall 的 QPS 和延迟分位数
# ==========================================================================


def bench_remember_throughput():
    """Bulk insert throughput: vectors/sec + latency distribution."""
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_rt_")
    try:
        mem = make_mem(tmpdir, "rt")
        n = 1000
        latencies = []
        for i in range(n):
            content = f"吞吐测试记忆 {i}: 主题{i % 100}，标签标签{i % 30}，内容描述。"
            t0 = time.perf_counter()
            mem.remember(content)
            latencies.append((time.perf_counter() - t0) * 1000)

        total_time = sum(latencies) / 1000  # seconds
        qps = n / total_time if total_time > 0 else 0

        return {
            "records": n,
            "total_sec": round(total_time, 2),
            "qps": round(qps, 1),
            "latency_p50_ms": round(percentile(latencies, 50), 2),
            "latency_p95_ms": round(percentile(latencies, 95), 2),
            "latency_p99_ms": round(percentile(latencies, 99), 2),
            "latency_max_ms": round(max(latencies), 2),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_recall_throughput():
    """Recall throughput: QPS + latency distribution with pre-populated data."""
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_rc_")
    try:
        mem = make_mem(tmpdir, "rc")
        # Pre-populate
        topics = ["机器学习", "深度学习", "自然语言处理", "计算机视觉",
                   "强化学习", "知识图谱", "推荐系统", "搜索引擎",
                   "数据挖掘", "分布式系统"]
        for i in range(500):
            topic = topics[i % len(topics)]
            mem.remember(f"关于{topic}的知识点 {i}: 这是一条详细的描述信息。")

        # Measure recall latencies
        queries = [f"{topics[i % len(topics)]}相关知识" for i in range(200)]
        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            mem.recall(q, limit=10)
            latencies.append((time.perf_counter() - t0) * 1000)

        total_time = sum(latencies) / 1000
        qps = len(queries) / total_time if total_time > 0 else 0

        return {
            "queries": len(queries),
            "db_size": 500,
            "qps": round(qps, 1),
            "latency_p50_ms": round(percentile(latencies, 50), 2),
            "latency_p95_ms": round(percentile(latencies, 95), 2),
            "latency_p99_ms": round(percentile(latencies, 99), 2),
            "latency_max_ms": round(max(latencies), 2),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_filtered_recall_latency():
    """Filtered recall (user_id, tags, memory_type) latency comparison."""
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_fr_")
    try:
        mem = make_mem(tmpdir, "fr")
        users = ["alice", "bob", "charlie", "diana"]
        types = [
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL,
        ]
        for i in range(300):
            user = users[i % len(users)]
            mtype = types[i % len(types)]
            mem.remember(
                f"用户{user}的记忆 {i}: 主题{i % 20}",
                memory_type=mtype,
                user_id=user,
                tags=[f"tag{i % 10}", f"cat{i % 5}"],
            )

        # Unfiltered recall
        lat_unfiltered = measure_latencies(
            lambda: mem.recall("记忆主题", limit=10), 100
        )
        # User-filtered recall
        lat_user = measure_latencies(
            lambda: mem.recall("记忆主题", limit=10, user_id="alice"),
            100,
        )
        # Tag-filtered recall
        lat_tag = measure_latencies(
            lambda: mem.recall("记忆主题", limit=10, tags=["tag3"]),
            100,
        )

        return {
            "unfiltered_p50_ms": round(percentile(lat_unfiltered, 50), 2),
            "unfiltered_p99_ms": round(percentile(lat_unfiltered, 99), 2),
            "user_filtered_p50_ms": round(percentile(lat_user, 50), 2),
            "user_filtered_p99_ms": round(percentile(lat_user, 99), 2),
            "tag_filtered_p50_ms": round(percentile(lat_tag, 50), 2),
            "tag_filtered_p99_ms": round(percentile(lat_tag, 99), 2),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Category 2: RECALL ACCURACY (对标 ANN-Benchmarks)
# 衡量向量搜索的 recall@K 精度
# ==========================================================================


def bench_recall_accuracy():
    """
    Measure recall@K accuracy by comparing with brute-force ground truth.
    Uses deterministic mock embeddings to enable exact comparison.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_acc_")
    try:
        mem = make_mem(tmpdir, "acc")
        # Inject deterministic embedder
        dim = 32

        class DeterministicEmbedder:
            """Deterministic mock embedder for reproducible tests."""
            dimension = dim

            def encode(self, texts):
                return [self._embed(t) for t in texts]

            def encode_one(self, text):
                return self._embed(text)

            def _embed(self, text: str) -> list[float]:
                # Deterministic: hash-based embedding
                h = hash(text) % (2**31)
                rng_local = random.Random(h)
                vec = [rng_local.uniform(-1, 1) for _ in range(dim)]
                # Normalize
                mag = sum(x**2 for x in vec) ** 0.5
                return [x / mag for x in vec] if mag > 0 else vec

        mem._embedding_provider = DeterministicEmbedder()

        # Store known items with predictable embeddings
        stored_texts = []
        for i in range(200):
            text = f"知识条目 {i}: 关于主题{i % 20}的详细描述"
            mem.remember(text)
            stored_texts.append(text)

        # For each query, compute ground truth (brute-force cosine similarity)
        # and compare with NeuralMem's recall results
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = sum(x**2 for x in a) ** 0.5
            mag_b = sum(x**2 for x in b) ** 0.5
            return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0

        def brute_force_top_k(query_text, k=5):
            q_emb = DeterministicEmbedder()._embed(query_text)
            sims = []
            for text in stored_texts:
                e = DeterministicEmbedder()._embed(text)
                sims.append((text, cosine_sim(q_emb, e)))
            sims.sort(key=lambda x: -x[1])
            return [s[0] for s in sims[:k]]

        # Test recall accuracy
        test_queries = [f"关于主题{i}的知识" for i in range(20)]
        recalls_at_5 = []
        recalls_at_10 = []

        for q in test_queries:
            # Ground truth
            gt_5 = set(brute_force_top_k(q, 5))
            gt_10 = set(brute_force_top_k(q, 10))

            # NeuralMem results
            results = mem.recall(q, limit=10, min_score=0.0)
            result_texts = {r.memory.content for r in results}
            nm_5 = set(list(result_texts)[:5])
            nm_10 = result_texts

            # Recall@K = |intersection| / |ground_truth|
            r5 = len(gt_5 & nm_5) / len(gt_5) if gt_5 else 1.0
            r10 = len(gt_10 & nm_10) / len(gt_10) if gt_10 else 1.0
            recalls_at_5.append(r5)
            recalls_at_10.append(r10)

        avg_r5 = statistics.mean(recalls_at_5) if recalls_at_5 else 0
        avg_r10 = statistics.mean(recalls_at_10) if recalls_at_10 else 0
        min_r5 = min(recalls_at_5) if recalls_at_5 else 0

        return {
            "recall_at_5": round(avg_r5, 4),
            "recall_at_10": round(avg_r10, 4),
            "recall_at_5_min": round(min_r5, 4),
            "test_queries": len(test_queries),
            "db_size": len(stored_texts),
            # Industry benchmark: recall@10 >= 0.95 is good
            "recall_grade": "A" if avg_r10 >= 0.95 else "B" if avg_r10 >= 0.85 else "C",
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Category 3: LARGE-SCALE STRESS (对标 Milvus large dataset tests)
# 5000+ 记忆的写入/检索/完整性
# ==========================================================================


def bench_large_scale_5k():
    """5000 records: insert throughput, recall latency, data integrity."""
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_5k_")
    try:
        mem = make_mem(tmpdir, "5k")
        n = 5000
        mem_before = get_process_memory_mb()

        # Bulk insert
        insert_start = time.perf_counter()
        for i in range(n):
            topic = f"主题{i % 100}"
            mem.remember(
                f"大规模测试 {i}: {topic}相关内容，标签{i % 50}",
                tags=[f"tag{i % 50}", f"cat{i % 10}"],
                importance=0.3 + (i % 7) * 0.1,
            )
        insert_time = time.perf_counter() - insert_start

        mem_after = get_process_memory_mb()
        all_mems = mem.storage.list_memories()

        # Recall latency on large dataset
        latencies = measure_latencies(
            lambda: mem.recall("主题50相关内容", limit=10), 50
        )

        # Data integrity: verify no corruption
        integrity_ok = len(all_mems) > 0
        for m in all_mems[:100]:
            if not m.content or not m.id:
                integrity_ok = False
                break

        return {
            "inserted": n,
            "stored": len(all_mems),
            "insert_qps": round(n / insert_time, 1),
            "insert_total_sec": round(insert_time, 2),
            "recall_p50_ms": round(percentile(latencies, 50), 2),
            "recall_p95_ms": round(percentile(latencies, 95), 2),
            "recall_p99_ms": round(percentile(latencies, 99), 2),
            "memory_growth_mb": round(mem_after - mem_before, 1),
            "integrity_ok": integrity_ok,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Category 4: CONCURRENT MIXED WORKLOAD (对标 Qdrant benchmarks)
# 多线程读写 + 过滤 + 实体图遍历
# ==========================================================================


def bench_concurrent_mixed_workload():
    """
    Mixed workload: 4 writer threads + 4 reader threads + 2 filter threads.
    Measures error rate, throughput, and latency under contention.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_cm_")
    try:
        mem = make_mem(tmpdir, "cm")
        errors = []
        lock = threading.Lock()
        write_latencies = []
        read_latencies = []
        stop_event = threading.Event()

        def writer(tid: int):
            rng = random.Random(tid * 1000)
            while not stop_event.is_set():
                try:
                    content = f"并发写入 tid={tid} 项目{rng.randint(0, 999)}"
                    t0 = time.perf_counter()
                    mem.remember(content, tags=[f"w{tid}", f"t{rng.randint(0, 9)}"])
                    lat = (time.perf_counter() - t0) * 1000
                    with lock:
                        write_latencies.append(lat)
                except Exception as e:
                    with lock:
                        errors.append(f"W{tid}: {e}")

        def reader(tid: int):
            rng = random.Random(tid * 2000)
            queries = ["机器学习", "深度学习", "自然语言", "计算机", "数据"]
            while not stop_event.is_set():
                try:
                    q = queries[rng.randint(0, len(queries) - 1)]
                    t0 = time.perf_counter()
                    mem.recall(q, limit=5)
                    lat = (time.perf_counter() - t0) * 1000
                    with lock:
                        read_latencies.append(lat)
                except Exception as e:
                    with lock:
                        errors.append(f"R{tid}: {e}")

        def filter_reader(tid: int):
            rng = random.Random(tid * 3000)
            users = ["alice", "bob", "charlie"]
            while not stop_event.is_set():
                try:
                    user = users[rng.randint(0, len(users) - 1)]
                    t0 = time.perf_counter()
                    mem.recall("知识", limit=5, user_id=user)
                    lat = (time.perf_counter() - t0) * 1000
                    with lock:
                        read_latencies.append(lat)
                except Exception as e:
                    with lock:
                        errors.append(f"F{tid}: {e}")

        # Pre-populate with some data
        for i in range(100):
            mem.remember(f"预填充数据 {i}: 机器学习和深度学习相关内容", user_id="alice")

        # Run mixed workload for 5 seconds
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = []
            for tid in range(4):
                futures.append(pool.submit(writer, tid))
            for tid in range(4):
                futures.append(pool.submit(reader, tid))
            for tid in range(2):
                futures.append(pool.submit(filter_reader, tid))

            time.sleep(5)
            stop_event.set()
            for f in futures:
                f.result(timeout=10)

        all_mems = mem.storage.list_memories()

        return {
            "total_errors": len(errors),
            "write_ops": len(write_latencies),
            "read_ops": len(read_latencies),
            "write_qps": round(len(write_latencies) / 5, 1),
            "read_qps": round(len(read_latencies) / 5, 1),
            "write_p50_ms": round(percentile(write_latencies, 50), 2) if write_latencies else 0,
            "write_p99_ms": round(percentile(write_latencies, 99), 2) if write_latencies else 0,
            "read_p50_ms": round(percentile(read_latencies, 50), 2) if read_latencies else 0,
            "read_p99_ms": round(percentile(read_latencies, 99), 2) if read_latencies else 0,
            "stored_memories": len(all_mems),
            "errors_sample": errors[:5] if errors else [],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_concurrent_high_contention():
    """
    Extreme contention: 16 threads all writing to same DB.
    Tests SQLite WAL serialization + lock contention handling.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_hc_")
    try:
        mem = make_mem(tmpdir, "hc")
        errors = []
        lock = threading.Lock()
        latencies = []

        def heavy_writer(tid: int):
            try:
                for i in range(50):
                    content = f"高争用 tid={tid} 序号={i} " + "x" * 200
                    t0 = time.perf_counter()
                    mem.remember(content)
                    lat = (time.perf_counter() - t0) * 1000
                    with lock:
                        latencies.append(lat)
            except Exception as e:
                with lock:
                    errors.append(f"T{tid}: {e}")

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(heavy_writer, tid) for tid in range(16)]
            for f in futures:
                f.result(timeout=120)
        total_time = time.perf_counter() - t0

        all_mems = mem.storage.list_memories()

        return {
            "threads": 16,
            "total_ops": len(latencies),
            "total_errors": len(errors),
            "total_sec": round(total_time, 2),
            "global_qps": round(len(latencies) / total_time, 1) if total_time > 0 else 0,
            "latency_p50_ms": round(percentile(latencies, 50), 2) if latencies else 0,
            "latency_p95_ms": round(percentile(latencies, 95), 2) if latencies else 0,
            "latency_p99_ms": round(percentile(latencies, 99), 2) if latencies else 0,
            "stored": len(all_mems),
            "error_sample": errors[:5],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Category 5: WAL CHECKPOINT STRESS (SQLite-specific)
# 大量小事务触发 WAL auto-checkpoint
# ==========================================================================


def bench_wal_checkpoint_stress():
    """
    Force WAL checkpoint by doing many small transactions.
    Verify data integrity after checkpoint.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_wal_")
    try:
        mem = make_mem(tmpdir, "wal")
        n = 2000

        # Many small writes (each triggers WAL append)
        t0 = time.perf_counter()
        for i in range(n):
            mem.remember(f"WAL压力测试 {i}")
        write_time = time.perf_counter() - t0

        # Force checkpoint via ANALYZE + WAL checkpoint pragma
        try:
            mem.storage._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass  # Some SQLite builds may not support this

        # Verify all data survived checkpoint
        all_mems = mem.storage.list_memories()

        # Read performance after checkpoint
        latencies = measure_latencies(
            lambda: mem.recall("WAL压力", limit=10), 100
        )

        return {
            "written": n,
            "survived": len(all_mems),
            "integrity_pct": round(len(all_mems) / n * 100, 1) if n > 0 else 0,
            "write_sec": round(write_time, 2),
            "post_checkpoint_p50_ms": round(percentile(latencies, 50), 2),
            "post_checkpoint_p99_ms": round(percentile(latencies, 99), 2),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Category 6: MEMORY PRESSURE (Resource exhaustion)
# 大量记录下 RAM 增长和泄漏检测
# ==========================================================================


def bench_memory_pressure():
    """
    Monitor RAM growth during 3000 remember + recall cycles.
    Detect memory leaks by comparing baseline vs final RSS.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_mem_")
    try:
        gc.collect()
        baseline_mb = get_process_memory_mb()

        mem = make_mem(tmpdir, "mem")
        mem_after_init = get_process_memory_mb()

        # Write phase
        for i in range(3000):
            mem.remember(f"内存压力测试 {i}: 这是一条较长的记忆内容用于消耗内存")

        mem_after_write = get_process_memory_mb()

        # Read phase (many recalls)
        for i in range(500):
            mem.recall(f"压力测试 {i % 50}", limit=10)

        mem_after_read = get_process_memory_mb()

        # Cleanup phase: delete and recall
        mem.forget("内存压力测试")
        gc.collect()
        mem_after_cleanup = get_process_memory_mb()

        all_mems = mem.storage.list_memories()

        return {
            "baseline_mb": round(baseline_mb, 1),
            "after_init_mb": round(mem_after_init, 1),
            "after_3k_write_mb": round(mem_after_write, 1),
            "after_500_read_mb": round(mem_after_read, 1),
            "after_cleanup_mb": round(mem_after_cleanup, 1),
            "write_growth_mb": round(mem_after_write - mem_after_init, 1),
            "read_growth_mb": round(mem_after_read - mem_after_write, 1),
            "cleanup_recovery_mb": round(mem_after_write - mem_after_cleanup, 1),
            "remaining_memories": len(all_mems),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Category 7: ENTITY GRAPH DENSITY (NeuralMem-specific)
# 大量实体/关系下的图遍历性能
# ==========================================================================


def bench_entity_graph_density():
    """
    Create a dense entity graph and measure traversal performance.
    Tests: entity resolution, graph traversal, relation queries.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_graph_")
    try:
        mem = make_mem(tmpdir, "graph")

        # Create many memories with overlapping entities
        entities = [f"实体{i}" for i in range(50)]
        t0 = time.perf_counter()
        for i in range(500):
            e1 = entities[i % len(entities)]
            e2 = entities[(i + 1) % len(entities)]
            mem.remember(f"{e1}和{e2}之间存在关系，事件编号{i}")
        insert_time = time.perf_counter() - t0

        # Measure graph traversal latency
        latencies = measure_latencies(
            lambda: mem.recall("实体0和实体1的关系", limit=10), 100
        )

        # Check graph stats
        try:
            kg = mem._knowledge_graph
            node_count = len(kg.graph.nodes) if hasattr(kg, 'graph') else 0
            edge_count = len(kg.graph.edges) if hasattr(kg, 'graph') else 0
        except Exception:
            node_count = -1
            edge_count = -1

        return {
            "memories": 500,
            "entities": len(entities),
            "graph_nodes": node_count,
            "graph_edges": edge_count,
            "insert_sec": round(insert_time, 2),
            "traversal_p50_ms": round(percentile(latencies, 50), 2),
            "traversal_p95_ms": round(percentile(latencies, 95), 2),
            "traversal_p99_ms": round(percentile(latencies, 99), 2),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Category 8: DEGRADATION & RECOVERY (Resilience)
# 异常注入后的恢复能力
# ==========================================================================


def bench_rapid_write_delete_cycle():
    """
    Rapid create-delete-create cycle to test WAL consistency.
    Simulates memory churn in long-running agent sessions.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_churn_")
    try:
        mem = make_mem(tmpdir, "churn")
        errors = []

        for cycle in range(10):
            # Write batch
            for i in range(100):
                try:
                    mem.remember(f"周期{cycle} 记忆{i}")
                except Exception as e:
                    errors.append(f"W_c{cycle}_i{i}: {e}")

            # Delete half
            try:
                mem.forget(f"周期{cycle}")
            except Exception as e:
                errors.append(f"D_c{cycle}: {e}")

            # Verify remaining
            try:
                mem.recall(f"周期{cycle}", limit=50)
            except Exception as e:
                errors.append(f"R_c{cycle}: {e}")

        all_mems = mem.storage.list_memories()

        return {
            "cycles": 10,
            "total_errors": len(errors),
            "remaining_memories": len(all_mems),
            "errors_sample": errors[:5],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_session_layer_stress():
    """
    Stress test session-based memory isolation.
    Multiple concurrent sessions writing/reading simultaneously.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_sess_")
    try:
        mem = make_mem(tmpdir, "sess")
        errors = []
        lock = threading.Lock()

        def session_worker(sid: int):
            session_id = f"session_{sid}"
            try:
                for i in range(50):
                    mem.remember(
                        f"会话{sid} 记忆{i}",
                        session_id=session_id,
                        user_id=f"user_{sid % 5}",
                    )
                results = mem.recall(
                    f"会话{sid} 记忆", limit=10, user_id=f"user_{sid % 5}"
                )
                # Verify session isolation
                for r in results:
                    if r.memory.user_id and r.memory.user_id != f"user_{sid % 5}":
                        with lock:
                            errors.append(f"S{sid}: User isolation leak")
                        break
            except Exception as e:
                with lock:
                    errors.append(f"S{sid}: {e}")

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(session_worker, i) for i in range(8)]
            for f in futures:
                f.result(timeout=30)
        total_time = time.perf_counter() - t0

        all_mems = mem.storage.list_memories()

        return {
            "sessions": 8,
            "total_errors": len(errors),
            "total_memories": len(all_mems),
            "total_sec": round(total_time, 2),
            "errors_sample": errors[:5],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_reflect_under_load():
    """
    Test reflect() performance with large graph and many memories.
    Measures consolidation/decay under realistic data volumes.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_refl_")
    try:
        mem = make_mem(tmpdir, "refl")

        # Build up data
        for i in range(1000):
            topic = f"反思主题{i % 50}"
            mem.remember(
                f"{topic}: 第{i}条知识，重要性测试",
                importance=0.3 + (i % 7) * 0.1,
            )

        # Measure reflect latency
        t0 = time.perf_counter()
        report = mem.reflect("反思主题0")
        reflect_time = time.perf_counter() - t0

        return {
            "memories": 1000,
            "reflect_sec": round(reflect_time, 2),
            "report_length": len(report),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Category 9: OPTIMIZATION BENCHMARKS (opt4–opt8 验证)
# 验证查询缓存、批量编码、图谱批量持久化等优化效果
# ==========================================================================


def bench_query_cache_speedup():
    """
    Test query embedding cache: repeated identical queries should skip
    re-embedding and return from cache.  Measures cold vs warm latency.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_qcache_")
    try:
        mem = make_mem(tmpdir, "qcache")
        # Pre-populate
        for i in range(200):
            mem.remember(f"缓存测试 {i}: 主题{i % 20}相关知识")

        query = "缓存主题5相关的知识点"

        # Cold run (first call, must embed)
        cold_lats = measure_latencies(lambda: mem.recall(query, limit=10), 10)

        # Warm runs (should hit cache)
        warm_lats = measure_latencies(lambda: mem.recall(query, limit=10), 50)

        cold_p50 = percentile(cold_lats, 50)
        warm_p50 = percentile(warm_lats, 50)
        speedup = cold_p50 / warm_p50 if warm_p50 > 0 else 0

        return {
            "cold_p50_ms": round(cold_p50, 2),
            "warm_p50_ms": round(warm_p50, 2),
            "speedup_ratio": round(speedup, 2),
            "cache_effective": speedup > 1.0,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_entity_batch_encoding():
    """
    Test EntityResolver batch encoding: entity resolution on content with
    many entities should benefit from batched encode() calls.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_ebatch_")
    try:
        mem = make_mem(tmpdir, "ebatch")
        # Create memories with lots of entity-dense content
        entities = [f"组织{i}" for i in range(30)]
        for i in range(300):
            e1, e2, e3 = entities[i % 30], entities[(i + 7) % 30], entities[(i + 13) % 30]
            mem.remember(f"{e1}与{e2}和{e3}之间存在合作关系，事件{i}")

        # Measure recall that triggers entity resolution
        lats = measure_latencies(
            lambda: mem.recall("组织0与组织7的合作关系", limit=10), 50
        )

        return {
            "memories": 300,
            "p50_ms": round(percentile(lats, 50), 2),
            "p95_ms": round(percentile(lats, 95), 2),
            "p99_ms": round(percentile(lats, 99), 2),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_graph_batch_persistence():
    """
    Test KnowledgeGraph batch() context manager: bulk entity operations
    should use deferred persistence with a single flush().
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_gbatch_")
    try:
        mem = make_mem(tmpdir, "gbatch")

        # Phase 1: without batch (each remember triggers graph persist)
        t0 = time.perf_counter()
        for i in range(300):
            mem.remember(f"图谱测试 {i}: 实体{i % 20}和实体{(i + 1) % 20}")
        unbatched_sec = time.perf_counter() - t0

        # Phase 2: with batch context manager
        mem2 = make_mem(tmpdir, "gbatch2")
        t0 = time.perf_counter()
        with mem2._knowledge_graph.batch():
            for i in range(300):
                mem2.remember(f"批量图谱测试 {i}: 实体{i % 20}和实体{(i + 1) % 20}")
        batched_sec = time.perf_counter() - t0

        speedup = unbatched_sec / batched_sec if batched_sec > 0 else 0

        # Verify graph integrity
        kg = mem2._knowledge_graph
        nodes = len(kg.graph.nodes)
        edges = len(kg.graph.edges)

        return {
            "unbatched_sec": round(unbatched_sec, 2),
            "batched_sec": round(batched_sec, 2),
            "speedup_ratio": round(speedup, 2),
            "graph_nodes": nodes,
            "graph_edges": edges,
            "batch_effective": speedup > 0.9,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_batch_embedding_throughput():
    """
    Compare encode_one() vs encode() batch throughput.
    The batch encode() should be faster per-item than sequential encode_one().
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_bemb_")
    try:
        mem = make_mem(tmpdir, "bemb")
        embedder = mem._embedding_provider
        texts = [f"批量编码测试文本 {i}: 这是一段用于测试编码吞吐量的内容" for i in range(100)]

        # Sequential encode_one
        t0 = time.perf_counter()
        for t in texts:
            embedder.encode_one(t)
        sequential_sec = time.perf_counter() - t0

        # Batch encode
        t0 = time.perf_counter()
        embedder.encode(texts)
        batch_sec = time.perf_counter() - t0

        speedup = sequential_sec / batch_sec if batch_sec > 0 else 0

        return {
            "texts": len(texts),
            "sequential_sec": round(sequential_sec, 3),
            "batch_sec": round(batch_sec, 3),
            "speedup_ratio": round(speedup, 2),
            "batch_effective": speedup > 0.9,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_fts5_keyword_search():
    """
    Test FTS5 keyword search performance vs vector search.
    FTS5 should handle pure keyword queries efficiently.
    """
    tmpdir = tempfile.mkdtemp(prefix="nm_bench_fts_")
    try:
        mem = make_mem(tmpdir, "fts")
        # Insert with distinct keywords
        keywords = ["机器学习", "深度学习", "自然语言处理", "计算机视觉", "强化学习",
                     "知识图谱", "推荐系统", "数据挖掘", "分布式系统", "量子计算"]
        for i in range(500):
            kw = keywords[i % len(keywords)]
            mem.remember(f"FTS5测试 {i}: {kw}是人工智能的重要分支，应用广泛")

        # Keyword-focused query
        kw_lats = measure_latencies(
            lambda: mem.recall("机器学习 应用", limit=10), 50
        )
        # Semantic query
        sem_lats = measure_latencies(
            lambda: mem.recall("人工智能相关的技术领域", limit=10), 50
        )

        return {
            "memories": 500,
            "keyword_p50_ms": round(percentile(kw_lats, 50), 2),
            "keyword_p99_ms": round(percentile(kw_lats, 99), 2),
            "semantic_p50_ms": round(percentile(sem_lats, 50), 2),
            "semantic_p99_ms": round(percentile(sem_lats, 99), 2),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Runner
# ==========================================================================

ALL_BENCHMARKS = [
    # Category 1: Throughput
    (bench_remember_throughput, "remember_throughput", "throughput"),
    (bench_recall_throughput, "recall_throughput", "throughput"),
    (bench_filtered_recall_latency, "filtered_recall_latency", "throughput"),
    # Category 2: Recall Accuracy
    (bench_recall_accuracy, "recall_accuracy", "accuracy"),
    # Category 3: Large-Scale
    (bench_large_scale_5k, "large_scale_5k", "scale"),
    # Category 4: Concurrency
    (bench_concurrent_mixed_workload, "concurrent_mixed_workload", "concurrency"),
    (bench_concurrent_high_contention, "concurrent_high_contention", "concurrency"),
    # Category 5: WAL Stress
    (bench_wal_checkpoint_stress, "wal_checkpoint_stress", "wal"),
    # Category 6: Memory Pressure
    (bench_memory_pressure, "memory_pressure", "memory"),
    # Category 7: Entity Graph
    (bench_entity_graph_density, "entity_graph_density", "graph"),
    # Category 8: Resilience
    (bench_rapid_write_delete_cycle, "rapid_write_delete_cycle", "resilience"),
    (bench_session_layer_stress, "session_layer_stress", "resilience"),
    (bench_reflect_under_load, "reflect_under_load", "resilience"),
    # Category 9: Optimization Benchmarks (opt4–opt8)
    (bench_query_cache_speedup, "query_cache_speedup", "optimization"),
    (bench_entity_batch_encoding, "entity_batch_encoding", "optimization"),
    (bench_graph_batch_persistence, "graph_batch_persistence", "optimization"),
    (bench_batch_embedding_throughput, "batch_embedding_throughput", "optimization"),
    (bench_fts5_keyword_search, "fts5_keyword_search", "optimization"),
]


def main():
    print("=" * 72)
    print("  NeuralMem 业界级压测套件")
    print("  对标: VectorDBBench · ANN-Benchmarks · Qdrant Benchmarks")
    print("=" * 72)
    print()

    results: list[BenchResult] = []
    t_start = time.monotonic()

    for fn, name, category in ALL_BENCHMARKS:
        print(f"  [{category.upper():12s}] {name} ...", end=" ", flush=True)
        r = run_bench(fn, name, category)
        if r.passed:
            m = ", ".join(f"{k}={v}" for k, v in r.metrics.items())
            print(f"OK ({r.duration:.1f}s)")
            if m:
                print(f"{'':20s}  {m}")
        else:
            print(f"FAIL ({r.duration:.1f}s)")
            print(f"{'':20s}  ERROR: {r.error}")
        results.append(r)

    total_time = time.monotonic() - t_start

    # ── Summary ──
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    print("=" * 72)
    print(f"  Results: {passed} passed, {failed} failed, {len(results)} total")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 72)

    # ── Category breakdown ──
    categories: dict[str, list[BenchResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print()
    print("  Category Breakdown:")
    print("  " + "-" * 68)
    for cat, rs in categories.items():
        p = sum(1 for r in rs if r.passed)
        f = sum(1 for r in rs if not r.passed)
        print(f"    {cat:15s}: {p} passed, {f} failed")

    # ── Key metrics summary (对标 VectorDBBench) ──
    print()
    print("  Key Metrics Summary (VectorDBBench Style):")
    print("  " + "-" * 68)
    for r in results:
        if not r.passed:
            continue
        if "qps" in r.metrics:
            print(f"    {r.name:30s}: {r.metrics['qps']} QPS")
        if "recall_at_10" in r.metrics:
            grade = r.metrics.get('recall_grade', '?')
            r10 = r.metrics['recall_at_10']
            print(f"    {r.name:30s}: Recall@10={r10} ({grade})")
        if "latency_p99_ms" in r.metrics:
            print(f"    {r.name:30s}: P99={r.metrics['latency_p99_ms']}ms")

    # ── Save report ──
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_sec": round(total_time, 2),
        "passed": passed,
        "failed": failed,
        "results": [
            {
                "name": r.name,
                "category": r.category,
                "passed": r.passed,
                "duration": round(r.duration, 2),
                "metrics": r.metrics,
                "error": r.error,
            }
            for r in results
        ],
    }
    report_path = os.path.join(os.path.dirname(__file__), "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Report saved to: {report_path}")

    if failed:
        print("\n  ⚠️  FAILED BENCHMARKS:")
        for r in results:
            if not r.passed:
                print(f"    ✗ {r.name}: {r.error}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
