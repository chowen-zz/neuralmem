"""NeuralMem 自定义异常层级"""
from __future__ import annotations


class NeuralMemError(Exception):
    """所有 NeuralMem 异常的基类"""


class StorageError(NeuralMemError):
    """存储层异常（SQLite 操作失败、连接错误等）"""


class EmbeddingError(NeuralMemError):
    """Embedding 层异常（模型加载失败、编码错误等）"""


class ExtractionError(NeuralMemError):
    """提取层异常（解析失败、格式错误等）"""


class RetrievalError(NeuralMemError):
    """检索层异常（搜索失败、索引错误等）"""


class GraphError(NeuralMemError):
    """图谱层异常（节点不存在、关系错误等）"""


class ConfigError(NeuralMemError):
    """配置错误（参数无效、路径不存在等）"""
