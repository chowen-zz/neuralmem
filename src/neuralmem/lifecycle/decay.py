"""DecayManager — stub 实现（Week 9 完整实现）"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod

_logger = logging.getLogger(__name__)


class _LifecycleBase(ABC):
    """Week 9 将继承此基类实现真实逻辑"""

    @abstractmethod
    def apply_decay(self, user_id: str | None = None) -> int:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def remove_forgotten(self, user_id: str | None = None) -> int:
        raise NotImplementedError  # pragma: no cover


class DecayManager(_LifecycleBase):
    """
    记忆衰减管理器（Week 1-8 stub）。
    # TODO(week-9): 实现谢宾斯基遗忘曲线 + 访问频次加权
    接口稳定，Week 9 不会改变方法签名。
    """

    def apply_decay(self, user_id: str | None = None) -> int:
        _logger.debug("DecayManager.apply_decay stub called (Week 9 TODO)")
        return 0

    def remove_forgotten(self, user_id: str | None = None) -> int:
        _logger.debug("DecayManager.remove_forgotten stub called (Week 9 TODO)")
        return 0
