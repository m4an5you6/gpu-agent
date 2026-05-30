"""GPUCLOUD distributed worker task files.

Worker task files are explicit per-node inputs produced by a coordinator or
user script. They are intentionally separate from ``gpucloud.yaml``: a worker
agent consumes one local task file and manages only the local Megatron rank.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml


class WorkerTaskError(Exception):
    """Worker task load or validation failure."""

    def __init__(self, message: str, *, errors: Optional[Sequence[str]] = None):
        super().__init__(message)
        self.errors: Tuple[str, ...] = tuple(errors or ())


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _as_mapping(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _coerce_int(value: Any, path: str, errors: List[str], *, min_value: Optional[int] = None) -> Optional[int]:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        errors.append(path)
        return None
    if min_value is not None and ivalue < min_value:
        errors.append(path)
        return None
    return ivalue


def resolve_worker_task_path(explicit: Union[str, Path]) -> Path:
    path = Path(explicit).expanduser()
    if not path.is_file():
        raise WorkerTaskError(f"worker task file not found: {path}")
    return path.resolve()


def load_raw_worker_task(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkerTaskError(f"cannot read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise WorkerTaskError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise WorkerTaskError(f"{path}: root must be a mapping")
    return data


def merge_worker_task_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(data)
    job_id = str(out.get("job_id") or "worker-job")

    out.setdefault("framework", "megatron-lm")
    out.setdefault("role", "worker")

    distributed = out.setdefault("distributed", {})
    if not isinstance(distributed, dict):
        distributed = {}
        out["distributed"] = distributed
    distributed.setdefault("nproc_per_node", 1)
    distributed.setdefault("start_timeout_sec", 900)

    runtime = out.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
        out["runtime"] = runtime
    runtime.setdefault("workdir", f"~/gpucloud/jobs/{job_id}")
    runtime.setdefault("megatron_lm_dir", "./Megatron-LM")
    runtime.setdefault("python", "python")
    if not isinstance(runtime.get("env"), dict):
        runtime["env"] = {}

    training = out.setdefault("training", {})
    if not isinstance(training, dict):
        training = {}
        out["training"] = training
    training.setdefault("entrypoint", "pretrain_gpt.py")
    training.setdefault("checkpoint_dir", f"{runtime['workdir']}/checkpoints")
    training.setdefault("log_dir", f"{runtime['workdir']}/logs")
    if not isinstance(training.get("extra_args"), list):
        training["extra_args"] = []

    preflight = out.setdefault("preflight", {})
    if not isinstance(preflight, dict):
        preflight = {}
        out["preflight"] = preflight
    preflight.setdefault("require_gpu_count", distributed.get("nproc_per_node", 1))
    preflight.setdefault("min_vram_gb", 0)
    preflight.setdefault("heterogeneous_policy", "warn")

    return out


def validate_worker_task(data: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["(root): must be a mapping"]

    if _is_blank(data.get("job_id")):
        errors.append("job_id")

    framework = str(data.get("framework") or "megatron-lm").strip().lower()
    if framework != "megatron-lm":
        errors.append("framework")

    role = str(data.get("role") or "worker").strip().lower()
    if role != "worker":
        errors.append("role")

    distributed = _as_mapping(data, "distributed")
    for key in ("nnodes", "node_rank", "master_addr", "master_port"):
        if _is_blank(distributed.get(key)):
            errors.append(f"distributed.{key}")

    nnodes = _coerce_int(distributed.get("nnodes"), "distributed.nnodes", errors, min_value=1)
    node_rank = _coerce_int(
        distributed.get("node_rank"),
        "distributed.node_rank",
        errors,
        min_value=0,
    )
    _coerce_int(
        distributed.get("nproc_per_node", 1),
        "distributed.nproc_per_node",
        errors,
        min_value=1,
    )
    port = _coerce_int(
        distributed.get("master_port"),
        "distributed.master_port",
        errors,
        min_value=1,
    )
    if port is not None and port > 65535:
        errors.append("distributed.master_port")
    if nnodes is not None and node_rank is not None and node_rank >= nnodes:
        errors.append("distributed.node_rank")

    runtime = _as_mapping(data, "runtime")
    if _is_blank(runtime.get("workdir")):
        errors.append("runtime.workdir")
    if _is_blank(runtime.get("megatron_lm_dir")):
        errors.append("runtime.megatron_lm_dir")

    training = _as_mapping(data, "training")
    command_template = str(training.get("command_template") or "").strip()
    if not command_template and _is_blank(training.get("entrypoint")):
        errors.append("training.entrypoint")
    if _is_blank(training.get("data_path")):
        errors.append("training.data_path")
    if _is_blank(training.get("checkpoint_dir")):
        errors.append("training.checkpoint_dir")
    if _is_blank(training.get("log_dir")):
        errors.append("training.log_dir")

    preflight = _as_mapping(data, "preflight")
    policy = str(preflight.get("heterogeneous_policy") or "warn").strip().lower()
    if policy not in {"reject", "warn", "allow"}:
        errors.append("preflight.heterogeneous_policy")

    return sorted(set(errors))


@dataclass(frozen=True)
class WorkerTask:
    path: Path
    raw: Dict[str, Any]
    merged: Dict[str, Any]

    @property
    def job_id(self) -> str:
        return str(self.merged["job_id"])

    @property
    def framework(self) -> str:
        return str(self.merged.get("framework") or "megatron-lm")

    @property
    def role(self) -> str:
        return str(self.merged.get("role") or "worker")

    @property
    def distributed(self) -> Dict[str, Any]:
        return _as_mapping(self.merged, "distributed")

    @property
    def runtime(self) -> Dict[str, Any]:
        return _as_mapping(self.merged, "runtime")

    @property
    def training(self) -> Dict[str, Any]:
        return _as_mapping(self.merged, "training")

    @property
    def preflight(self) -> Dict[str, Any]:
        return _as_mapping(self.merged, "preflight")

    @property
    def node_rank(self) -> int:
        return int(self.distributed["node_rank"])

    @property
    def nnodes(self) -> int:
        return int(self.distributed["nnodes"])

    @property
    def nproc_per_node(self) -> int:
        return int(self.distributed.get("nproc_per_node") or 1)

    @property
    def master_addr(self) -> str:
        return str(self.distributed["master_addr"])

    @property
    def master_port(self) -> int:
        return int(self.distributed["master_port"])

    def summary(self) -> Dict[str, Any]:
        return {
            "task_file": str(self.path),
            "job_id": self.job_id,
            "framework": self.framework,
            "role": self.role,
            "node_rank": self.node_rank,
            "nnodes": self.nnodes,
            "nproc_per_node": self.nproc_per_node,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "workdir": self.runtime.get("workdir"),
            "megatron_lm_dir": self.runtime.get("megatron_lm_dir"),
            "data_path": self.training.get("data_path"),
            "checkpoint_dir": self.training.get("checkpoint_dir"),
            "log_dir": self.training.get("log_dir"),
            "heterogeneous_policy": self.preflight.get("heterogeneous_policy"),
        }


def load_worker_task(explicit: Union[str, Path]) -> WorkerTask:
    path = resolve_worker_task_path(explicit)
    raw = load_raw_worker_task(path)
    merged = merge_worker_task_defaults(raw)
    errors = validate_worker_task(merged)
    if errors:
        raise WorkerTaskError("worker task validation failed", errors=errors)
    return WorkerTask(path=path, raw=raw, merged=merged)


__all__ = [
    "WorkerTask",
    "WorkerTaskError",
    "load_raw_worker_task",
    "load_worker_task",
    "merge_worker_task_defaults",
    "resolve_worker_task_path",
    "validate_worker_task",
]
