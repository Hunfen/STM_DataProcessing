"""Reusable parallel helpers.

Currently one component: :mod:`~stm_data_processing.parallel.shard_driver`, the
shard-and-merge driver used by the Lindhard parallel calculator (shard files per
worker + an ordered merge in the parent, the only viable parallel write pattern
for a serial-only HDF5 build).
"""

from __future__ import annotations

from stm_data_processing.parallel.shard_driver import (
    CheckpointLock,
    RunStats,
    ShardSpec,
    ShardStore,
    absorb_progress,
    available_memory_bytes,
    log_aggregate_progress,
    log_plan,
    memory_guard,
    plan_slices,
    remove_tree,
    run_shards,
    scan_shards,
)

__all__ = [
    "CheckpointLock",
    "RunStats",
    "ShardSpec",
    "ShardStore",
    "absorb_progress",
    "available_memory_bytes",
    "log_aggregate_progress",
    "log_plan",
    "memory_guard",
    "plan_slices",
    "remove_tree",
    "run_shards",
    "scan_shards",
]
