"""NeuralMem 用户画像模块 — V1.5

从记忆行为推断用户画像，支持意图识别、偏好建模、知识图谱和交互风格分析。
新增 DeepProfileEngine V2 支持深度行为建模与持续学习。
"""
from __future__ import annotations

from neuralmem.profiles.base import ProfileAttribute, UserProfile
from neuralmem.profiles.engine import ProfileEngine
from neuralmem.profiles.types import (
    Intent,
    InteractionStyle,
    Knowledge,
    Preference,
    ProfileDimension,
)
from neuralmem.profiles.updater import ProfileUpdater
from neuralmem.profiles.v2_engine import AccessLogEntry, DeepProfileEngine

__all__ = [
    "UserProfile",
    "ProfileAttribute",
    "ProfileEngine",
    "ProfileUpdater",
    "DeepProfileEngine",
    "AccessLogEntry",
    "Intent",
    "Preference",
    "Knowledge",
    "InteractionStyle",
    "ProfileDimension",
]
