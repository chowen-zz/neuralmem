#!/usr/bin/env python3
"""RAG 应用示例 — 使用 NeuralMem 作为向量存储 + 记忆系统。"""
from neuralmem import NeuralMem, NeuralMemConfig


def rag_demo():
    """简单的 RAG 应用，结合文档检索和对话记忆。"""
    cfg = NeuralMemConfig(
        db_path="./rag_memory.db",
        enable_reranker=True,
        enable_importance_reinforcement=True,
    )
    mem = NeuralMem(config=cfg)

    # 1. 索引文档
    documents = [
        "NeuralMem 是一个本地优先的 MCP 原生记忆库。",
        "它支持 SQLite、PostgreSQL、Pinecone、Milvus 等多种存储后端。",
        "检索系统使用 4 路策略：语义搜索、关键词搜索、图谱遍历、时间衰减。",
        "RRF 融合算法将多路结果合并为统一排序列表。",
        "可选的 Cross-Encoder 重排序进一步提升检索质量。",
        "治理模块提供 6 态状态机、审计日志和投毒检测。",
        "支持 LangChain、LlamaIndex、CrewAI、AutoGen、Semantic Kernel 集成。",
    ]

    print("📚 索引文档...")
    for doc in documents:
        mem.remember(doc, tags=["documentation"])
    print(f"✅ 已索引 {len(documents)} 条文档")

    # 2. 模拟对话
    questions = [
        "NeuralMem 是什么？",
        "它支持哪些存储后端？",
        "检索系统怎么工作的？",
        "有哪些安全功能？",
    ]

    print("\n💬 开始问答:")
    for q in questions:
        # 检索相关文档
        results = mem.recall(q, limit=3, tags=["documentation"])

        # 模拟生成回答（实际应调用 LLM）
        context = "\n".join([f"- {r.content}" for r in results])
        answer = f"基于检索到的信息:\n{context}\n\n答案是: [LLM 生成的回答]"

        # 记住对话
        mem.remember(f"Q: {q}", tags=["question"])
        mem.remember(f"A: {answer}", tags=["answer"])

        print(f"\nQ: {q}")
        print(f"A: {answer[:200]}...")

    # 3. 后续问题（利用对话历史）
    print("\n🔄 后续问题（利用记忆）:")
    follow_up = "刚才提到的检索策略有哪些？"
    results = mem.recall(follow_up, limit=5)
    print(f"Q: {follow_up}")
    print(f"找到 {len(results)} 条相关记忆（包括之前的对话）")

    # 4. 统计
    stats = mem.get_stats()
    print(f"\n📊 最终统计: {stats}")


if __name__ == "__main__":
    rag_demo()
