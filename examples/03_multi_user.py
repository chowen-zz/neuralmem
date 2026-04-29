"""
NeuralMem 多用户记忆隔离示例

演示如何为不同用户维护独立的记忆空间
"""
from neuralmem import NeuralMem

mem = NeuralMem()

# 用户 A 的记忆
mem.remember("Alice 喜欢 Python 和机器学习", user_id="alice")
mem.remember("Alice 的项目是 NeuralMem", user_id="alice")

# 用户 B 的记忆
mem.remember("Bob 是前端工程师，专注 React", user_id="bob")
mem.remember("Bob 的框架偏好是 Next.js", user_id="bob")

# 各自独立检索
alice_results = mem.recall("编程语言", user_id="alice")
bob_results = mem.recall("前端框架", user_id="bob")

print("Alice 的记忆:")
for r in alice_results:
    print(f"  [{r.score:.2f}] {r.memory.content}")

print("\nBob 的记忆:")
for r in bob_results:
    print(f"  [{r.score:.2f}] {r.memory.content}")

# 遗忘：GDPR 合规完全删除用户数据
deleted = mem.forget(user_id="alice")
print(f"\n已删除 Alice 的 {deleted} 条记忆")
