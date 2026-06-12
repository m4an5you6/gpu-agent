"""Cluster control-plane configuration loading and role resolution."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from gpucloud_constants import get_gpucloud_home

ClusterRole = Literal["master", "worker", "auto"]
RouteMode = Literal["record", "queue", "guide", "interrupt", "execute_direct"]


@dataclass
class ClusterConfig:
    enabled: bool = False
    role: ClusterRole = "auto"
    node_id: str = ""
    master_url: str = "http://127.0.0.1:8765"
    bind_host: str = "127.0.0.1"
    bind_port: int = 8765
    secret_env: str = "GPUCLOUD_CLUSTER_SECRET"
    heartbeat_interval_sec: int = 5
    heartbeat_ttl_sec: int = 20
    data_dir: Path = field(default_factory=lambda: Path("/tmp/gpucloud-cluster"))
    database_url: str = ""
    master_epoch: int = 0
    event_session_key: str = ""
    event_routing: Dict[str, RouteMode] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    # Per-node logical path mappings for heterogeneous workers.
    node_paths: Dict[str, Any] = field(default_factory=dict)
    # Logical conda env name -> local prefix or python path.
    conda: Dict[str, Any] = field(default_factory=dict)

    @property
    def secret(self) -> str:
        return os.environ.get(self.secret_env, "").strip()

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def jsonl_path(self) -> Path:
        return self.data_dir / "cluster_audit.jsonl"


def _default_data_dir() -> Path:
    custom = os.environ.get("GPUCLOUD_CLUSTER_DATA_DIR", "").strip()
    if custom:
        return Path(custom)
    return get_gpucloud_home() / "cluster"


def _default_node_id() -> str:
    explicit = os.environ.get("GPUCLOUD_CLUSTER_NODE_ID", "").strip()
    if explicit:
        return explicit
    host = socket.gethostname().split(".")[0]
    return f"{host}-{uuid.uuid4().hex[:8]}"


def load_cluster_config(raw: Optional[Dict[str, Any]] = None) -> ClusterConfig:
    """Load cluster config from gpucloud config dict or DEFAULT_CONFIG defaults."""
    if raw is None:
        from gpucloud_cli.config import load_config

        cfg = load_config()
        raw = cfg.get("cluster") or {}
    if not isinstance(raw, dict):
        raw = {}

    data_dir = raw.get("data_dir") or ""
    path = Path(data_dir) if data_dir else _default_data_dir()

    routing = raw.get("event_routing") or {}
    if not isinstance(routing, dict):
        routing = {}

    return ClusterConfig(
        enabled=bool(raw.get("enabled", False)),
        role=str(raw.get("role") or "auto"),  # type: ignore[arg-type]
        node_id=str(raw.get("node_id") or _default_node_id()),
        master_url=str(raw.get("master_url") or "http://127.0.0.1:8765").rstrip("/"),
        bind_host=str(raw.get("bind_host") or "127.0.0.1"),
        bind_port=int(raw.get("bind_port") or 8765),
        secret_env=str(raw.get("secret_env") or "GPUCLOUD_CLUSTER_SECRET"),
        heartbeat_interval_sec=int(raw.get("heartbeat_interval_sec") or 5),
        heartbeat_ttl_sec=int(raw.get("heartbeat_ttl_sec") or 20),
        data_dir=path,
        database_url=str(raw.get("database_url") or "").strip(),
        master_epoch=int(raw.get("master_epoch") or 0),
        event_session_key=str(raw.get("event_session_key") or ""),
        event_routing={str(k): str(v) for k, v in routing.items()},  # type: ignore[misc]
        logging=dict(raw.get("logging") or {}),
        training=dict(raw.get("training") or {}),
        node_paths=dict(raw.get("node_paths") or {}),
        conda=dict(raw.get("conda") or {}),
    )


def resolve_role(cfg: ClusterConfig) -> ClusterRole:
    """Resolve effective runtime role from config.

    ``auto`` becomes ``master`` when this host binds the configured master port
    and ``master_url`` points at this host; otherwise ``worker``. First version
    does not perform automatic leader election.
    """
    if cfg.role in ("master", "worker"):
        return cfg.role

    parsed = urlparse(cfg.master_url)
    master_host = parsed.hostname or "127.0.0.1"
    master_port = parsed.port or cfg.bind_port

    local_addrs = {"127.0.0.1", "localhost", "::1"}
    try:
        local_addrs.add(socket.gethostname())
        local_addrs.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass

    if master_host in local_addrs and master_port == cfg.bind_port:
        return "master"
    return "worker"


def compute_config_hash(cfg: ClusterConfig, extra: Optional[Dict[str, Any]] = None) -> str:
    """Stable hash of node-relevant static config for assignment validation."""
    payload = {
        "node_id": cfg.node_id,
        "master_url": cfg.master_url,
        "training": cfg.training,
        "node_paths": cfg.node_paths,
        "conda": cfg.conda,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
