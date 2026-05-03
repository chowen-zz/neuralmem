"""NeuralMem plugin ecosystem — extensible memory pipeline."""
from neuralmem.plugins.base import Plugin, PluginContext
from neuralmem.plugins.builtins import DedupPlugin, ImportancePlugin, RecencyBoostPlugin
from neuralmem.plugins.manager import PluginManager

__all__ = [
    "Plugin",
    "PluginContext",
    "PluginManager",
    "DedupPlugin",
    "ImportancePlugin",
    "RecencyBoostPlugin",
]
