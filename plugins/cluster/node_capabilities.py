"""Heterogeneous worker capabilities — metrics, scheduling, and local path resolution."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from plugins.cluster.config import ClusterConfig
from plugins.cluster.models import GpuInfo, JobSpec, NodeRecord, RankAssignment, ValidationResult
from plugins.cluster.training import build_torchrun_command


@dataclass
class LogicalJobRequirements:
    """Logical job requirements parsed from a job spec (top-level or extra)."""

    env_name: str = ""
    project: str = ""
    release: str = ""
    dataset: str = ""
    output_run_id: str = ""
    min_scratch_gb: float = 0.0
    min_gpu_count: int = 0
    uses_logical_paths: bool = False

    @classmethod
    def from_spec_dict(cls, raw: Dict[str, Any]) -> "LogicalJobRequirements":
        extra = raw.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}

        def _pick(key: str, default: Any = "") -> Any:
            if raw.get(key) not in (None, ""):
                return raw.get(key)
            return extra.get(key, default)

        env_name = str(_pick("env_name") or "").strip()
        project = str(_pick("project") or "").strip()
        release = str(_pick("release") or "").strip()
        dataset = str(_pick("dataset") or "").strip()
        output_run_id = str(_pick("output_run_id") or "").strip()
        try:
            min_scratch_gb = float(_pick("min_scratch_gb", 0) or 0)
        except (TypeError, ValueError):
            min_scratch_gb = 0.0
        try:
            min_gpu_count = int(_pick("min_gpu_count", 0) or 0)
        except (TypeError, ValueError):
            min_gpu_count = 0

        uses_logical = bool(
            env_name or project or release or dataset or output_run_id or min_scratch_gb > 0
        )
        return cls(
            env_name=env_name,
            project=project,
            release=release,
            dataset=dataset,
            output_run_id=output_run_id,
            min_scratch_gb=min_scratch_gb,
            min_gpu_count=min_gpu_count,
            uses_logical_paths=uses_logical,
        )


@dataclass
class ResolvedLaunch:
    working_dir: str
    launch_command: List[str]
    env: Dict[str, str]
    python_executable: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except OSError:
        return 0.0


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _conda_python(cfg: ClusterConfig, env_name: str) -> Optional[str]:
    if not env_name:
        return None
    conda_cfg = cfg.conda if hasattr(cfg, "conda") else {}
    envs = conda_cfg.get("envs") or {}
    if not isinstance(envs, dict):
        return None
    raw = envs.get(env_name)
    if not raw:
        return None
    raw_s = str(raw).strip()
    if raw_s.endswith("/bin/python") or raw_s.endswith("/bin/python3"):
        return raw_s
    prefix = Path(raw_s).expanduser()
    for candidate in (prefix / "bin" / "python", prefix / "bin" / "python3"):
        if candidate.is_file():
            return str(candidate)
    return raw_s if Path(raw_s).is_file() else None


def resolve_python_executable(cfg: ClusterConfig, env_name: str = "") -> Tuple[str, List[str]]:
    """Resolve Python executable from job env_name or cluster defaults."""
    errors: List[str] = []
    if env_name:
        resolved = _conda_python(cfg, env_name)
        if resolved and Path(resolved).is_file():
            return resolved, errors
        errors.append(f"conda env '{env_name}' not found or python missing on this node")
    default = str(cfg.training.get("python_executable") or "python")
    if default != "python" and not Path(default).is_file():
        errors.append(f"python_executable not found: {default}")
    return default, errors


def collect_local_metrics(cfg: ClusterConfig, gpus: Optional[List[GpuInfo]] = None) -> Dict[str, Any]:
    """Build heartbeat metrics from local node config and runtime state."""
    node_paths = cfg.node_paths if hasattr(cfg, "node_paths") else {}
    conda_cfg = cfg.conda if hasattr(cfg, "conda") else {}

    code_roots = node_paths.get("code_roots") or {}
    data_roots = node_paths.get("data_roots") or {}
    checkpoint_roots = node_paths.get("checkpoint_roots") or {}
    scratch_roots = node_paths.get("scratch_roots") or {}

    scratch_free: Dict[str, float] = {}
    if isinstance(scratch_roots, dict):
        for key, path_str in scratch_roots.items():
            scratch_free[str(key)] = round(_disk_free_gb(Path(str(path_str)).expanduser()), 2)

    conda_envs = list((conda_cfg.get("envs") or {}).keys()) if isinstance(conda_cfg.get("envs"), dict) else []

    gpu_list = gpus or []
    gpu_count = len(gpu_list)
    cuda_version = ""
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = str(getattr(torch.version, "cuda", "") or "")
    except Exception:
        pass

    return {
        "conda_envs": sorted(conda_envs),
        "code_roots": sorted(str(k) for k in code_roots.keys()) if isinstance(code_roots, dict) else [],
        "data_roots": sorted(str(k) for k in data_roots.keys()) if isinstance(data_roots, dict) else [],
        "checkpoint_roots": sorted(str(k) for k in checkpoint_roots.keys()) if isinstance(checkpoint_roots, dict) else [],
        "scratch_free_gb": scratch_free,
        "gpu_count": gpu_count,
        "cuda_version": cuda_version,
    }


def _node_metrics(node: NodeRecord, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if metrics:
        return metrics
    caps = getattr(node, "capabilities", None) or {}
    return caps if isinstance(caps, dict) else {}


def node_matches_job(
    node: NodeRecord,
    req: LogicalJobRequirements,
    *,
    nproc_per_node: int,
    stale: bool,
    metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Return whether a node satisfies logical job requirements."""
    reasons: List[str] = []
    if stale:
        reasons.append("stale heartbeat")
    if node.state in ("lost", "quarantined", "draining"):
        reasons.append(f"state={node.state}")
    if node.state == "busy":
        reasons.append("busy")

    node_metrics = _node_metrics(node, metrics)
    gpu_count = _gpu_count(node, node_metrics)
    required_gpus = max(req.min_gpu_count, nproc_per_node)
    if gpu_count < required_gpus:
        reasons.append(f"need {required_gpus} gpus, have {gpu_count}")

    if req.env_name:
        available = set(node_metrics.get("conda_envs") or [])
        if req.env_name not in available:
            reasons.append(f"missing conda env '{req.env_name}'")

    if req.project:
        available = set(node_metrics.get("code_roots") or [])
        if req.project not in available:
            reasons.append(f"missing code root for project '{req.project}'")

    if req.dataset:
        available = set(node_metrics.get("data_roots") or [])
        if req.dataset not in available:
            reasons.append(f"missing dataset '{req.dataset}'")

    if req.output_run_id and req.project:
        available = set(node_metrics.get("checkpoint_roots") or [])
        if req.project not in available:
            reasons.append(f"missing checkpoint root for project '{req.project}'")

    if req.min_scratch_gb > 0:
        scratch_free = node_metrics.get("scratch_free_gb") or {}
        if not isinstance(scratch_free, dict) or not scratch_free:
            reasons.append(f"no scratch roots reporting free space (need {req.min_scratch_gb} GB)")
        else:
            best = max(float(v) for v in scratch_free.values())
            if best < req.min_scratch_gb:
                reasons.append(f"insufficient scratch (need {req.min_scratch_gb} GB, best {best:.1f} GB)")

    return (not reasons, reasons)


def _gpu_count(node: NodeRecord, metrics: Dict[str, Any]) -> int:
    if metrics.get("gpu_count"):
        return int(metrics["gpu_count"])
    return len(node.gpus)


def select_nodes_for_job(
    nodes: List[NodeRecord],
    req: LogicalJobRequirements,
    *,
    nnodes: int,
    nproc_per_node: int,
    stale_ids: Optional[set] = None,
    metrics_by_node: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[NodeRecord], List[str]]:
    """Select up to nnodes that satisfy job requirements."""
    stale_ids = stale_ids or set()
    metrics_by_node = metrics_by_node or {}
    eligible: List[NodeRecord] = []
    rejections: List[str] = []

    for node in nodes:
        ok, reasons = node_matches_job(
            node,
            req,
            nproc_per_node=nproc_per_node,
            stale=node.node_id in stale_ids,
            metrics=metrics_by_node.get(node.node_id),
        )
        if ok:
            eligible.append(node)
        else:
            rejections.append(f"{node.node_id}: {'; '.join(reasons)}")

    if len(eligible) < nnodes:
        return [], rejections + [f"need {nnodes} eligible nodes, have {len(eligible)}"]

    return eligible[:nnodes], rejections


def _resolve_working_dir(cfg: ClusterConfig, spec: JobSpec, assignment: RankAssignment) -> Tuple[str, List[str]]:
    req = LogicalJobRequirements.from_spec_dict(spec.to_dict())
    errors: List[str] = []

    if req.project and req.release:
        code_roots = (cfg.node_paths or {}).get("code_roots") or {}
        root = code_roots.get(req.project)
        if not root:
            errors.append(f"code root not configured for project '{req.project}'")
            return assignment.working_dir, errors
        path = Path(str(root)).expanduser() / req.release
        if not _path_exists(path):
            errors.append(f"code path does not exist: {path}")
        return str(path), errors

    if req.project and not req.release:
        code_roots = (cfg.node_paths or {}).get("code_roots") or {}
        root = code_roots.get(req.project)
        if root:
            path = Path(str(root)).expanduser()
            if not _path_exists(path):
                errors.append(f"code root path does not exist: {path}")
            return str(path), errors

    wd = assignment.working_dir or spec.working_dir or "."
    path = Path(wd).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if wd != "." and not _path_exists(path):
        errors.append(f"working_dir does not exist: {path}")
    return str(path), errors


def _pick_scratch(cfg: ClusterConfig, min_gb: float) -> Tuple[str, List[str]]:
    errors: List[str] = []
    scratch_roots = (cfg.node_paths or {}).get("scratch_roots") or {}
    if not isinstance(scratch_roots, dict) or not scratch_roots:
        if min_gb > 0:
            errors.append("no scratch_roots configured")
        return "", errors

    best_key = ""
    best_free = -1.0
    best_path = ""
    for key, path_str in scratch_roots.items():
        path = Path(str(path_str)).expanduser()
        free = _disk_free_gb(path)
        if free >= min_gb and free > best_free:
            best_free = free
            best_key = str(key)
            best_path = str(path)

    if min_gb > 0 and not best_path:
        errors.append(f"no scratch root with >= {min_gb} GB free")
    return best_path, errors


def resolve_local_launch(
    cfg: ClusterConfig,
    spec: JobSpec,
    assignment: RankAssignment,
) -> ResolvedLaunch:
    """Resolve logical paths and rebuild launch command on the worker."""
    req = LogicalJobRequirements.from_spec_dict(spec.to_dict())
    errors: List[str] = []
    warnings: List[str] = []

    working_dir, wd_errors = _resolve_working_dir(cfg, spec, assignment)
    errors.extend(wd_errors)

    env = dict(assignment.env or spec.env)
    node_paths = cfg.node_paths or {}

    if req.dataset:
        data_roots = node_paths.get("data_roots") or {}
        data_path = data_roots.get(req.dataset)
        if not data_path:
            errors.append(f"dataset '{req.dataset}' not configured on this node")
        else:
            dp = Path(str(data_path)).expanduser()
            if not _path_exists(dp):
                errors.append(f"dataset path does not exist: {dp}")
            else:
                env.setdefault("DATA_DIR", str(dp))

    if req.output_run_id and req.project:
        ckpt_roots = node_paths.get("checkpoint_roots") or {}
        ckpt_root = ckpt_roots.get(req.project)
        if not ckpt_root:
            errors.append(f"checkpoint root not configured for project '{req.project}'")
        else:
            out_path = Path(str(ckpt_root)).expanduser() / req.output_run_id
            out_path.mkdir(parents=True, exist_ok=True)
            env.setdefault("OUTPUT_DIR", str(out_path))

    scratch_path, scratch_errors = _pick_scratch(cfg, req.min_scratch_gb)
    errors.extend(scratch_errors)
    if scratch_path:
        scratch_job = Path(scratch_path) / assignment.job_id
        scratch_job.mkdir(parents=True, exist_ok=True)
        env.setdefault("TMPDIR", str(scratch_job))
        env.setdefault("SCRATCH_DIR", str(scratch_job))

    python_executable, py_errors = resolve_python_executable(cfg, req.env_name)
    errors.extend(py_errors)

    local_assignment = RankAssignment(
        assignment_id=assignment.assignment_id,
        job_id=assignment.job_id,
        node_id=assignment.node_id,
        node_rank=assignment.node_rank,
        nproc_per_node=assignment.nproc_per_node,
        nnodes=assignment.nnodes,
        world_size=assignment.world_size,
        master_addr=assignment.master_addr,
        master_port=assignment.master_port,
        master_epoch=assignment.master_epoch,
        job_generation=assignment.job_generation,
        gpus=assignment.gpus,
        working_dir=working_dir,
        env=env,
    )

    cmd, cmd_env = build_torchrun_command(
        spec,
        local_assignment,
        cfg,
        python_executable=python_executable,
    )
    env.update(cmd_env)

    script_path = Path(working_dir) / spec.script
    if spec.framework.lower() != "placeholder" and not _path_exists(script_path):
        errors.append(f"training script not found: {script_path}")

    return ResolvedLaunch(
        working_dir=working_dir,
        launch_command=cmd,
        env=env,
        python_executable=python_executable,
        errors=errors,
        warnings=warnings,
    )


def validate_resolved_launch(resolved: ResolvedLaunch) -> ValidationResult:
    if resolved.ok:
        return ValidationResult(ok=True, warnings=resolved.warnings)
    return ValidationResult(
        ok=False,
        errors=resolved.errors,
        warnings=resolved.warnings,
    )
