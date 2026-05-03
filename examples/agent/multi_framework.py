#!/usr/bin/env python3
"""Agent 框架集成示例 — 展示多框架共享 NeuralMem 记忆。"""
from neuralmem import NeuralMem, NeuralMemConfig
from neuralmem.integrations import (
    NeuralMemLangChainMemory,
    NeuralMemLlamaIndexMemory,
    CrewAIMemory,
)


def agent_integration_demo():
    """展示多个 Agent 框架共享同一个 NeuralMem 实例。"""
    cfg = NeuralMemConfig(
        db_path="./agent_shared.db",
        enable_llm_extraction=True,
        enable_reranker=True,
    )

    # 创建共享记忆
    shared_mem = NeuralMem(config=cfg)

    # 1. LangChain Agent 存储发现
    print("🦜 LangChain Agent 存储研究...")
    lc_memory = NeuralMemLangChainMemory.from_neuralmem(shared_mem)
    lc_memory.save_context(
        {"input": "Research quantum computing"},
        {"output": "Quantum computing uses qubits instead of classical bits."}
    )
    print("✅ LangChain 存储完成")

    # 2. LlamaIndex Agent 添加洞察
    print("🦙 LlamaIndex Agent 添加洞察...")
    li_memory = NeuralMemLlamaIndexMemory.from_neuralmem(shared_mem)
    li_memory.put("Shor's algorithm can factor integers in polynomial time on a quantum computer.")
    print("✅ LlamaIndex 存储完成")

    # 3. CrewAI Agent 执行任务
    print("🚀 CrewAI Agent 执行任务...")
    crew_memory = CrewAIMemory.from_neuralmem(shared_mem)
    crew_memory.save(
        task="Summarize quantum computing",
        result="Quantum computing leverages superposition and entanglement."
    )
    print("✅ CrewAI 存储完成")

    # 4. 任意 Agent 查询共享记忆
    print("\n🔍 跨框架查询共享记忆:")
    results = shared_mem.recall("quantum computing applications", limit=5)
    print(f"找到 {len(results)} 条相关记忆:")
    for r in results:
        print(f"  • {r.content[:80]}...")

    # 5. 统计
    stats = shared_mem.get_stats()
    print(f"\n📊 共享记忆统计: {stats}")

    # 6. 展示记忆隔离
    print("\n🔒 用户隔离示例:")
    user_a = NeuralMem(db_path="./agent_shared.db", user_id="alice")
    user_b = NeuralMem(db_path="./agent_shared.db", user_id="bob")

    user_a.remember("Alice's secret project: Project X")
    user_b.remember("Bob's favorite language: Rust")

    a_results = user_a.recall("secret project")
    b_results = user_b.recall("favorite language")

    print(f"Alice 查询 'secret project': {len(a_results)} 条")
    print(f"Bob 查询 'favorite language': {len(b_results)} 条")


if __name__ == "__main__":
    agent_integration_demo()
