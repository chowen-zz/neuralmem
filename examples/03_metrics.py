"""
NeuralMem Metrics Example — Monitor operation performance

Demonstrates how to use MetricsCollector to track remember/recall
latency and call counts.

Run: python examples/03_metrics.py
"""
from neuralmem import NeuralMem
from neuralmem.core.config import NeuralMemConfig

# 1. Enable metrics collection
config = NeuralMemConfig(enable_metrics=True)
mem = NeuralMem(config=config)

# 2. Store some memories
print("=== Storing memories ===")
mem.remember("User prefers Python for backend development")
mem.remember("User deploys to AWS using Terraform")
mem.remember("User's favorite editor is VS Code")
mem.remember("User writes tests with pytest")

# 3. Recall some memories
print("\n=== Recalling memories ===")
results = mem.recall("What tools does the user prefer?")
for r in results:
    print(f"  [{r.score:.2f}] {r.memory.content}")

# 4. Inspect collected metrics
print("\n=== Metrics ===")
metrics = mem.metrics.get_metrics()

print("\nCounters:")
for key, value in metrics["counters"].items():
    print(f"  {key}: {value}")

print("\nHistograms:")
for key, stats in metrics["histograms"].items():
    print(f"  {key}:")
    print(f"    count = {stats['count']}")
    print(f"    mean  = {stats['mean']:.4f}s")
    print(f"    p50   = {stats['p50']:.4f}s")
    print(f"    p95   = {stats['p95']:.4f}s")
    print(f"    p99   = {stats['p99']:.4f}s")

# 5. Reset and re-measure
print("\n=== After reset ===")
mem.metrics.reset()
mem.recall("Python")
after = mem.metrics.get_metrics()
print(f"  recall.calls counter after 1 more recall: {after['counters']}")
