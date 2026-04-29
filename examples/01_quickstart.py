"""
NeuralMem 快速开始示例 — 3 行代码给 Agent 添加长期记忆

运行前: pip install neuralmem
"""
from neuralmem import NeuralMem

# 1. 创建记忆引擎（自动初始化 SQLite + 本地 Embedding）
mem = NeuralMem()

# 2. 存储记忆
memories = mem.remember("用户偏好用 TypeScript 写前端，讨厌 JavaScript")
print(f"存储了 {len(memories)} 条记忆")

# 3. 检索相关记忆
results = mem.recall("用户喜欢什么编程语言？")
for r in results:
    print(f"[{r.score:.2f}] ({r.retrieval_method}) {r.memory.content}")

# 4. 反思（多轮检索 + 图谱遍历）
report = mem.reflect("技术偏好")
print(report)

# 5. 查看统计
stats = mem.get_stats()
print(f"记忆库: {stats}")
