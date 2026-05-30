"""GPUCLOUD training start, dry-run, and job queries (phase 6)."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.gpucloud_config import (
    GpucloudConfigError,
    GpucloudPreparedConfig,
    generate_training_command,
)
from hermes_cli.gpucloud_context import (
    iter_cluster_nodes,
    node_ssh_key_path,
    resolve_config_for_tool,
)
from hermes_cli.gpucloud_jobs import (
    TrainingJob,
    get_job,
    list_recent_jobs,
    new_job_id,
    save_job,
    update_job_status,
)
from hermes_cli.gpucloud_probe import build_ssh_display
from hermes_cli.gpucloud_ssh import quote_remote_path, run_ssh_command


def _training_section(merged: Dict[str, Any]) -> Dict[str, Any]:
    sec = merged.get("training")
    return sec if isinstance(sec, dict) else {}


def _security(merged: Dict[str, Any]) -> Dict[str, Any]:
    sec = merged.get("security")
    return sec if isinstance(sec, dict) else {}


def training_paths(
    merged: Dict[str, Any],
    job_id: str,
) -> Tuple[str, str, str]:
    """Return workdir, log_path, checkpoint_path."""
    training = _training_section(merged)
    workdir = "~/gpucloud"
    for _c, _i, node in iter_cluster_nodes(merged):
        workdir = str(node.get("workdir") or workdir)
        break
    log_dir = str(training.get("log_dir") or f"{workdir}/logs")
    checkpoint_dir = str(training.get("checkpoint_dir") or f"{workdir}/checkpoints")
    log_path = f"{log_dir}/{job_id}.log"
    return workdir, log_path, checkpoint_dir


def validate_training_command(command: str) -> Optional[str]:
    text = (command or "").strip()
    if not text:
        return "training command is empty"
    first = text.split(None, 1)[0]
    if first != "torchrun":
        return "phase 6 requires a Megatron-LM launcher via torchrun"
    return None


def resolve_train_node(
    merged: Dict[str, Any],
    *,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
) -> Tuple[str, int, Dict[str, Any]]:
    for cname, idx, node in iter_cluster_nodes(merged, cluster_name=cluster_name):
        if idx == node_index:
            return cname, idx, node
    raise GpucloudConfigError(f"node index {node_index} not found")


def build_remote_launch_command(
    *,
    workdir: str,
    log_path: str,
    train_command: str,
) -> str:
    """cd workdir, ensure dirs, nohup train command, print PID."""
    inner = train_command.replace("'", "'\"'\"'")
    return (
        f"cd {quote_remote_path(workdir)} && "
        f"mkdir -p {quote_remote_path(str(Path(log_path).parent))} && "
        f"nohup bash -lc '{inner}' >> {quote_remote_path(log_path)} 2>&1 & echo $!"
    )


def plan_training_job(
    prepared: GpucloudPreparedConfig,
    *,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    merged = prepared.merged
    cname, nidx, node = resolve_train_node(
        merged, cluster_name=cluster_name, node_index=node_index
    )
    jid = job_id or new_job_id()
    command = generate_training_command(merged)
    err = validate_training_command(command)
    if err:
        return {"ok": False, "error": err}

    workdir, log_path, checkpoint_path = training_paths(merged, jid)
    remote_cmd = build_remote_launch_command(
        workdir=workdir,
        log_path=log_path,
        train_command=command,
    )

    return {
        "ok": True,
        "job_id": jid,
        "cluster": cname,
        "node_index": nidx,
        "host": node.get("host"),
        "workdir": workdir,
        "log_path": log_path,
        "checkpoint_path": checkpoint_path,
        "launch_command": command,
        "remote_launch_command": remote_cmd,
        "ssh_command": build_ssh_display(node, remote_cmd),
        "dataset": prepared.effective_dataset,
        "model": prepared.effective_model,
        "node": {
            "user": node.get("user"),
            "port": node.get("port"),
            "ssh_key_path": str(node.get("ssh_key", "")),
        },
    }


def run_train_start(
    *,
    config_file: Optional[str] = None,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    dry_run: Optional[bool] = None,
    confirm_execute: bool = False,
    allow_discover_without_goal: bool = False,
) -> Dict[str, Any]:
    try:
        prepared = resolve_config_for_tool(
            config_file,
            allow_discover_without_goal=allow_discover_without_goal,
        )
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc)}

    plan = plan_training_job(
        prepared,
        cluster_name=cluster_name,
        node_index=node_index,
    )
    if not plan.get("ok"):
        return plan

    merged = prepared.merged
    sec = _security(merged)
    do_dry = sec.get("dry_run_required", True) if dry_run is None else bool(dry_run)

    if do_dry or not confirm_execute:
        return {
            **plan,
            "dry_run": True,
            "message": (
                "Dry-run only. Re-run with confirm_execute=true (or gpucloud train start --yes) "
                "to launch on the remote node."
            ),
        }

    cname, nidx, node = resolve_train_node(
        merged, cluster_name=cluster_name, node_index=node_index
    )
    job = TrainingJob(
        job_id=plan["job_id"],
        job_type="train",
        cluster=cname,
        status="pending",
        launch_command=plan["launch_command"],
        workdir=plan["workdir"],
        log_path=plan["log_path"],
        checkpoint_path=plan["checkpoint_path"],
        node_index=nidx,
        host=str(node.get("host", "")),
        dataset=plan["dataset"],
        model=plan["model"],
    )
    save_job(job)

    timeout = int(sec.get("command_timeout_sec") or 3600)
    result = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=node_ssh_key_path(node),
        remote_command=plan["remote_launch_command"],
        timeout_sec=min(120, timeout),
        output_limit=512,
    )

    if not result.ok:
        update_job_status(
            job.job_id,
            "failed",
            last_error=result.error or f"ssh exit {result.exit_code}",
        )
        job = get_job(job.job_id) or job
        return {
            "ok": False,
            "dry_run": False,
            "job": job.to_dict(),
            "error": result.error,
            "ssh": result.as_dict(),
        }

    pid = (result.stdout or "").strip().splitlines()[-1] if result.stdout else ""
    update_job_status(job.job_id, "running", remote_pid=pid or None)
    job = get_job(job.job_id) or job

    return {
        "ok": True,
        "dry_run": False,
        "job": job.to_dict(),
        "remote_pid": pid,
        "log_path": job.log_path,
        "checkpoint_path": job.checkpoint_path,
        "message": f"Training job {job.job_id} started (pid={pid or 'unknown'})",
    }


def run_train_status(
    job_id: Optional[str] = None,
    *,
    limit: int = 10,
) -> Dict[str, Any]:
    if job_id:
        job = get_job(job_id)
        if not job:
            return {"ok": False, "error": f"job not found: {job_id}"}
        return {"ok": True, "job": job.to_dict()}
    jobs = list_recent_jobs(limit=limit)
    return {
        "ok": True,
        "jobs": [j.to_dict() for j in jobs],
        "count": len(jobs),
    }


def run_train_logs(
    job_id: str,
    *,
    lines: int = 50,
) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job:
        return {"ok": False, "error": f"job not found: {job_id}"}

    try:
        prepared = resolve_config_for_tool(allow_discover_without_goal=True)
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc), "job_id": job_id, "log_path": job.log_path}

    merged = prepared.merged
    _c, _i, node = resolve_train_node(
        merged, cluster_name=job.cluster or None, node_index=job.node_index
    )

    n = max(1, min(int(lines), 500))
    tail_cmd = f"tail -n {n} {shlex.quote(job.log_path)} 2>/dev/null || echo '[log not found yet]'"
    sec = _security(merged)
    timeout = int(sec.get("command_timeout_sec") or 3600)
    result = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=node_ssh_key_path(node),
        remote_command=tail_cmd,
        timeout_sec=min(120, timeout),
        output_limit=int(sec.get("max_output_chars") or 8000),
    )

    return {
        "ok": result.ok,
        "job_id": job_id,
        "log_path": job.log_path,
        "lines_requested": n,
        "tail": result.stdout,
        "stderr": result.stderr,
        "truncated": result.truncated,
        "error": result.error,
        "note": "Full log is on the remote host; only the tail is returned here.",
    }
