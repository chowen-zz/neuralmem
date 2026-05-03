# AutoGen 集成指南

将 NeuralMem 作为 AutoGen Agent 的持久化记忆。

## 安装

```bash
pip install neuralmem pyautogen
```

## 基本用法

```python
from neuralmem.integrations import AutoGenMemory
from autogen import ConversableAgent, UserProxyAgent

# 创建 NeuralMem 记忆
memory = AutoGenMemory(
    db_path="./autogen_memory.db",
    enable_reranker=True
)

# 创建 Assistant
assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful AI assistant with long-term memory.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": "..."}]},
    memory=memory
)

# 创建 User Proxy
user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)

# 开始对话（自动利用历史记忆）
user_proxy.initiate_chat(
    assistant,
    message="What did we discuss about Python yesterday?"
)
```

## 多会话记忆

```python
# 会话 1
memory1 = AutoGenMemory(db_path="./shared.db", session_id="morning")
assistant1 = ConversableAgent(name="assistant", memory=memory1, ...)

# 会话 2 — 继承记忆
memory2 = AutoGenMemory(db_path="./shared.db", session_id="afternoon")
assistant2 = ConversableAgent(name="assistant", memory=memory2, ...)
# assistant2 可以引用 morning 会话的内容
```

## 群组聊天记忆

```python
from autogen import GroupChat, GroupChatManager

shared_memory = AutoGenMemory(db_path="./group.db")

agents = [
    ConversableAgent(name="coder", memory=shared_memory, ...),
    ConversableAgent(name="reviewer", memory=shared_memory, ...),
    ConversableAgent(name="tester", memory=shared_memory, ...),
]

group_chat = GroupChat(agents=agents, messages=[], max_round=12)
manager = GroupChatManager(groupchat=group_chat, memory=shared_memory)
```
