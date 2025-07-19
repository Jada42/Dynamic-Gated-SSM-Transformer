"""
Dynamic SSM Transformer
A hybrid attention-SSM architecture with adaptive gating
"""

__version__ = "0.1.0"

from .models.adaptive_ssm import (
    AdaptiveHybridAttentionSSM,
    MemoryAugmentedAdaptiveSSM
)

from .models.hybrid_model import (
    ConservativeHybridModel,
    CompleteAdaptiveHybridModel
)

from .tools.mcp_interface import MCPInterface
from .monitoring.model_monitor import ModelMonitor, ModelMetrics
from .utils.visualization import visualize_gates, plot_memory_usage

__all__ = [
    "AdaptiveHybridAttentionSSM",
    "MemoryAugmentedAdaptiveSSM",
    "ConservativeHybridModel", 
    "CompleteAdaptiveHybridModel",
    "MCPInterface",
    "ModelMonitor",
    "ModelMetrics",
    "visualize_gates",
    "plot_memory_usage"
]
