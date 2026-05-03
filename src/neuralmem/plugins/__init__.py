"""NeuralMem plugin ecosystem — V1.3 extensible memory pipeline."""
from neuralmem.plugins.base import Plugin, PluginContext
from neuralmem.plugins.builtin import LoggingPlugin, MetricsPlugin, ValidationPlugin
from neuralmem.plugins.builtins import DedupPlugin, ImportancePlugin, RecencyBoostPlugin
from neuralmem.plugins.manager import PluginManager
from neuralmem.plugins.registry import PluginRegistry

__all__ = [
    "Plugin",
    "PluginContext",
    "PluginManager",
    "PluginRegistry",
    "DedupPlugin",
    "ImportancePlugin",
    "RecencyBoostPlugin",
    "LoggingPlugin",
    "MetricsPlugin",
    "ValidationPlugin",
]
