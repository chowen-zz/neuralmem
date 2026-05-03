"""Performance utilities for NeuralMem V1.2 — sub-100ms P99 optimization."""
from neuralmem.perf.batch import BatchProcessor, BatchResult, ItemResult
from neuralmem.perf.batch_embedding import (
    BatchEmbeddingProcessor,
    BatchEmbeddingResult,
)
from neuralmem.perf.batch_processor import BatchProcessor as BatchProcessorV2
from neuralmem.perf.benchmark import BenchmarkReport, LatencyBenchmark
from neuralmem.perf.incremental_index import IncrementalIndex, IndexStats
from neuralmem.perf.metrics import PerformanceMetrics, OperationMetrics, SystemMetrics
from neuralmem.perf.prefetch import PrefetchEngine, PrefetchStats
from neuralmem.perf.query_cache import QueryCache, CacheStats, CacheEntry
from neuralmem.perf.query_planner import (
    QueryPlanner,
    QueryProfile,
    StrategyWeights,
)

__all__ = [
    "BatchProcessor",
    "BatchProcessorV2",
    "BatchResult",
    "ItemResult",
    "BatchEmbeddingProcessor",
    "BatchEmbeddingResult",
    "LatencyBenchmark",
    "BenchmarkReport",
    "IncrementalIndex",
    "IndexStats",
    "QueryPlanner",
    "QueryProfile",
    "StrategyWeights",
    "QueryCache",
    "CacheStats",
    "CacheEntry",
    "PrefetchEngine",
    "PrefetchStats",
    "PerformanceMetrics",
    "OperationMetrics",
    "SystemMetrics",
]
