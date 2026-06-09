"""Runtime scope for GPUCLOUD yaml — goal session vs explicit tool/CLI paths."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path

from gpucloud_cli.gpucloud_config import (
    GpucloudConfigError,
    GpucloudPreparedConfig,
    prepare_gpucloud_config,
)

_prepared: ContextVar[Optional[GpucloudPreparedConfig]] = ContextVar(
    "gpucloud_prepared", default=None
)
_goal_active: ContextVar[bool] = ContextVar("gpucloud_goal_active", default=False)

GOAL_REQUIRED_MSG = (
    "GPUCLOUD cluster tools need an active /goal session or an explicit "
    "config_file argument. Normal chat does not load gpucloud.yaml."
)


def set_goal_gpucloud_config(prepared: GpucloudPreparedConfig) -> None:
    _prepared.set(prepared)
    _goal_active.set(True)


def clear_goal_gpucloud_config() -> None:
    _prepared.set(None)
    _goal_active.set(False)


def get_active_gpucloud_config() -> Optional[GpucloudPreparedConfig]:
    return _prepared.get()


def is_goal_gpucloud_active() -> bool:
    return _goal_active.get()


def resolve_config_for_tool(
    config_file: Optional[Union[str, Path]] = None,
    *,
    allow_discover_without_goal: bool = False,
) -> GpucloudPreparedConfig:
    if config_file:
        return prepare_gpucloud_config(config_file)
    active = _prepared.get()
    if active is not None:
        return active
    if _goal_active.get() or allow_discover_without_goal:
        return prepare_gpucloud_config()
    raise GpucloudConfigError(GOAL_REQUIRED_MSG)


def iter_cluster_nodes(
    merged: Dict[str, Any],
    *,
    cluster_name: Optional[str] = None,
) -> Iterator[Tuple[str, int, Dict[str, Any]]]:
    """Yield (cluster_name, node_index, node_dict) from merged config."""
    for cluster in merged.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        cname = str(cluster.get("name") or "cluster")
        if cluster_name and cname != cluster_name:
            continue
        nodes = cluster.get("nodes") or []
        if not isinstance(nodes, list):
            continue
        for idx, node in enumerate(nodes):
            if isinstance(node, dict):
                yield cname, idx, node


def node_ssh_key_path(node: Dict[str, Any]) -> str:
    return str(Path(str(node.get("ssh_key", "")).strip()).expanduser())


def node_label(cluster: str, index: int, node: Dict[str, Any]) -> str:
    host = node.get("host", "?")
    return f"{cluster}/nodes[{index}] ({node.get('user')}@{host})"
