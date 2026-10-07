"""Checkpoint save/load for BeliefNet torch components (lazy torch import)."""
from __future__ import annotations

from pathlib import Path


def save_checkpoint(module, path: str | Path) -> None:
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)


def load_checkpoint(module, path: str | Path, map_location: str = "cpu"):
    import torch
    module.load_state_dict(torch.load(path, map_location=map_location))
    return module
