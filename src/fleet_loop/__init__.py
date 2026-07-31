"""robot-fleet-loop — nodes triage and sync selectively, the hub retrains and canaries back."""

from __future__ import annotations

__version__ = "0.1.0"

from .hub import Hub, HubConfig
from .loop import FleetConfig, LoopReport, run_loop
from .node import FleetNode, NodeConfig, sync

__all__ = [
    "FleetConfig",
    "FleetNode",
    "Hub",
    "HubConfig",
    "LoopReport",
    "NodeConfig",
    "__version__",
    "run_loop",
    "sync",
]
