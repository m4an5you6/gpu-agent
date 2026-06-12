"""Map validated job assignments to torchrun / DeepSpeed launch commands."""

from __future__ import annotations

import random
import socket
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from plugins.cluster.config import ClusterConfig
from plugins.cluster.models import JobSpec, RankAssignment, ValidationResult, new_id


def pick_rendezvous_port(cfg: ClusterConfig) -> int:
    lo, hi = cfg.training.get("rendezvous_port_range") or [29500, 29600]
    try:
        lo_i, hi_i = int(lo), int(hi)
    except (TypeError, ValueError):
        lo_i, hi_i = 29500, 29600
    if lo_i >= hi_i:
        hi_i = lo_i + 100
    return random.randint(lo_i, hi_i)


def parse_master_addr(master_url: str, fallback_host: str = "127.0.0.1") -> str:
    parsed = urlparse(master_url)
    host = parsed.hostname or fallback_host
    if host in ("0.0.0.0", "::"):
        return fallback_host
    return host


def validate_job_spec(raw: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    script = str(raw.get("script") or "").strip()
    if not script:
        errors.append("script is required")

    nnodes = int(raw.get("nnodes") or 1)
    nproc = int(raw.get("nproc_per_node") or 1)
    if nnodes < 1:
        errors.append("nnodes must be >= 1")
    if nproc < 1:
        errors.append("nproc_per_node must be >= 1")

    framework = str(raw.get("framework") or "torchrun").lower()
    if framework not in ("torchrun", "deepspeed", "placeholder"):
        warnings.append(f"framework '{framework}' falls back to torchrun semantics")

    script_args = raw.get("script_args") or []
    if not isinstance(script_args, list):
        errors.append("script_args must be a list")

    env = raw.get("env") or {}
    if not isinstance(env, dict):
        errors.append("env must be an object")

    working_dir = str(raw.get("working_dir") or ".")

    extra = raw.get("extra") or {}
    if not isinstance(extra, dict):
        errors.append("extra must be an object")
        extra = {}

    # Logical fields may appear at top level or inside extra.
    logical_keys = (
        "env_name", "project", "release", "dataset",
        "output_run_id", "min_scratch_gb", "min_gpu_count",
    )
    for key in logical_keys:
        if raw.get(key) not in (None, "") and key not in extra:
            extra[key] = raw.get(key)

    req = None
    try:
        from plugins.cluster.node_capabilities import LogicalJobRequirements

        req = LogicalJobRequirements.from_spec_dict({**raw, "extra": extra})
    except Exception:
        pass

    if req and req.uses_logical_paths:
        if req.project and req.release:
            pass  # working_dir resolved per-node
        elif not working_dir or working_dir == ".":
            if not (req.project or req.dataset or req.env_name):
                errors.append("logical job requires project/release, env_name, or dataset")
        if req.env_name and not req.project and working_dir == ".":
            warnings.append("env_name set without project/release; ensure working_dir exists on each node")

    normalized = {
        "script": script,
        "script_args": [str(a) for a in script_args] if isinstance(script_args, list) else [],
        "nnodes": nnodes,
        "nproc_per_node": nproc,
        "framework": framework,
        "env": {str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {},
        "working_dir": working_dir,
        "job_id": str(raw.get("job_id") or new_id("job-")),
        "idempotency_key": str(raw.get("idempotency_key") or raw.get("request_id") or ""),
        "extra": extra,
    }

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, normalized=normalized)


def build_torchrun_command(
    spec: JobSpec,
    assignment: RankAssignment,
    cfg: ClusterConfig,
    *,
    python_executable: Optional[str] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Build launch command and env for a rank assignment."""
    py = python_executable or str(cfg.training.get("python_executable") or "python")
    framework = spec.framework.lower()

    env = dict(spec.env)
    env.update({
        "MASTER_ADDR": assignment.master_addr,
        "MASTER_PORT": str(assignment.master_port),
        "WORLD_SIZE": str(assignment.world_size),
        "RANK": str(assignment.node_rank * assignment.nproc_per_node),
        "LOCAL_RANK": "0",
        "NODE_RANK": str(assignment.node_rank),
    })

    ifname = str(cfg.training.get("nccl_socket_ifname") or "").strip()
    if ifname:
        env["NCCL_SOCKET_IFNAME"] = ifname

    if framework == "placeholder":
        return (
            [py, "-c", f"print('cluster placeholder job {spec.job_id} rank {assignment.node_rank}')"],
            env,
        )

    if framework == "deepspeed":
        cmd = [
            py, "-m", "deepspeed.launcher.launch",
            f"--nnodes={assignment.nnodes}",
            f"--node_rank={assignment.node_rank}",
            f"--nproc_per_node={assignment.nproc_per_node}",
            f"--master_addr={assignment.master_addr}",
            f"--master_port={assignment.master_port}",
            spec.script,
            *spec.script_args,
        ]
        return cmd, env

    cmd = [
        py, "-m", "torch.distributed.run",
        f"--nnodes={assignment.nnodes}",
        f"--nproc_per_node={assignment.nproc_per_node}",
        f"--node_rank={assignment.node_rank}",
        f"--master_addr={assignment.master_addr}",
        f"--master_port={assignment.master_port}",
        spec.script,
        *spec.script_args,
    ]
    return cmd, env


def validate_local_assignment(
    assignment: RankAssignment,
    *,
    node_id: str,
    config_hash: str,
    local_addrs: List[str],
) -> ValidationResult:
    errors: List[str] = []
    if assignment.node_id != node_id:
        errors.append(f"node_id mismatch: expected {node_id}, got {assignment.node_id}")
    if assignment.master_addr not in local_addrs and assignment.node_rank == 0:
        # rank 0 should bind master addr on a local interface when it's this node
        try:
            socket.getaddrinfo(assignment.master_addr, assignment.master_port)
        except socket.gaierror:
            errors.append(f"master_addr {assignment.master_addr} not reachable locally on rank 0")
    if not config_hash:
        errors.append("missing local config_hash")
    return ValidationResult(ok=not errors, errors=errors)
