"""NeuralMem Tiered Memory Module — V1.1

分层记忆模块：根据访问频率自动将数据在 Hot 层（内存 LRU）和 Deep 层（磁盘 SQLite）之间迁移。

Exports:
    TieredStorage   - 分层存储抽象基类
    HotStore        - 内存热层（LRU 缓存）
    DeepStore       - 磁盘冷层（SQLite 持久化）
    TieredManager   - 自动分层管理器（对外统一接口）
"""
from __future__ import annotations

from neuralmem.tiered.base import TieredStorage
from neuralmem.tiered.deep_store import DeepStore
from neuralmem.tiered.hot_store import HotStore
from neuralmem.tiered.manager import TieredManager

__all__ = [
    "TieredStorage",
    "HotStore",
    "DeepStore",
    "TieredManager",
]
