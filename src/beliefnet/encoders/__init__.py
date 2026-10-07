"""beliefnet.encoders: frozen LLM backbone wrappers."""

from beliefnet.encoders.frozen_backbone import (
    FrozenBackbone,
    FrozenBackboneConfig,
    Pooling,
    pool_hidden,
)

__all__ = ["FrozenBackbone", "FrozenBackboneConfig", "Pooling", "pool_hidden"]
