#!/usr/bin/env python3
"""对话机器人示例 — 使用 NeuralMem 作为长期记忆。"""
from neuralmem import NeuralMem


def chatbot_demo():
    """简单的长期记忆对话机器人。"""
    mem = NeuralMem(db_path="./chatbot_memory.db")

    print("🤖 长期记忆对话机器人")
    print("命令: /remember <内容> | /recall <查询> | /forget <id> | /stats | /quit")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "/quit":
            print("再见！")
            break

        if user_input.startswith("/remember "):
            content = user_input[10:]
            result = mem.remember(content)
            print(f"✅ 已记住 {len(result)} 条记忆")
            continue

        if user_input.startswith("/recall "):
            query = user_input[8:]
            results = mem.recall(query, limit=5)
            print(f"🔍 找到 {len(results)} 条相关记忆:")
            for r in results:
                print(f"  • [{r.id[:8]}] {r.content[:60]}...")
            continue

        if user_input.startswith("/forget "):
            mid = user_input[8:]
            mem.forget(memory_id=mid)
            print(f"🗑️ 已删除记忆 {mid[:8]}")
            continue

        if user_input == "/stats":
            stats = mem.get_stats()
            print(f"📊 统计: {stats}")
            continue

        # 普通对话 — 自动记住并检索相关历史
        mem.remember(f"User: {user_input}")
        related = mem.recall(user_input, limit=3)
        context = "\n".join([f"- {r.content}" for r in related])

        # 模拟 AI 回复（实际应调用 LLM）
        reply = f"我注意到你提到了'{user_input}'。"
        if related:
            reply += f" 这让我想起之前的话题:\n{context}"

        mem.remember(f"AI: {reply}")
        print(f"AI: {reply}")


if __name__ == "__main__":
    chatbot_demo()
