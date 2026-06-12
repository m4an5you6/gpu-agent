"""Pydantic-style dataclasses for cluster control-plane messages."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


JobState = Literal[
    "pending", "assigning", "running", "succeeded", "failed", "cancelled", "stopped"
]
NodeState = Literal["registering", "ready", "busy", "draining", "lost", "quarantined"]
RouteMode = Literal["record", "queue", "guide", "interrupt", "execute_direct"]


def _now() -> float:
    return time.time()


def new_id(prefix: str = "") -> str:
    token = uuid.uuid4().hex[:12]
    return f"{prefix}{token}" if prefix else token


@dataclass
class GpuInfo:
    index: int
    name: str = ""
    memory_mb: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeRecord:
    node_id: str
    advertised_addr: str
    state: NodeState = "registering"
    gpus: List[GpuInfo] = field(default_factory=list)
    agent_version: str = ""
    config_hash: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gpus"] = [g.to_dict() for g in self.gpus]
        return d


@dataclass
class HeartbeatPayload:
    node_id: str
    state: NodeState = "ready"
    gpus: List[GpuInfo] = field(default_factory=list)
    config_hash: str = ""
    running_job_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gpus"] = [g.to_dict() for g in self.gpus]
        return d


@dataclass
class JobSpec:
    script: str
    script_args: List[str] = field(default_factory=list)
    nnodes: int = 1
    nproc_per_node: int = 1
    framework: str = "torchrun"
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "."
    job_id: str = field(default_factory=lambda: new_id("job-"))
    idempotency_key: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankAssignment:
    assignment_id: str
    job_id: str
    node_id: str
    node_rank: int
    nproc_per_node: int
    nnodes: int
    world_size: int
    master_addr: str
    master_port: int
    master_epoch: int
    job_generation: int
    gpus: List[int] = field(default_factory=list)
    launch_command: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "."
    job_spec: Dict[str, Any] = field(default_factory=dict)
    state: str = "pending"
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JobRecord:
    job_id: str
    spec: JobSpec
    state: JobState = "pending"
    master_epoch: int = 0
    job_generation: int = 1
    master_addr: str = ""
    master_port: int = 0
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    error_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["spec"] = self.spec.to_dict()
        return d


@dataclass
class ClusterEvent:
    event_id: str
    event_type: str
    payload: Dict[str, Any]
    route_mode: RouteMode = "record"
    job_id: str = ""
    node_id: str = ""
    request_id: str = ""
    ts: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentActionLog:
    action_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    decision: str
    result_summary: str
    session_id: str = ""
    turn_id: str = ""
    ts: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessRunLog:
    run_id: str
    job_id: str
    node_id: str
    command: List[str]
    cwd: str
    env_keys: List[str]
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    ts_start: float = field(default_factory=_now)
    ts_end: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalized: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
