"""GPUCLOUD checkpoint list, validation, resume, and cleanup (phase 7)."""

from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional, Tuple

from gpucloud_cli.gpucloud_config import (
    GpucloudConfigError,
    GpucloudPreparedConfig,
    generate_training_command,
)
from gpucloud_cli.gpucloud_context import node_ssh_key_path, resolve_config_for_tool
from gpucloud_cli.gpucloud_jobs import (
    TrainingJob,
    get_job,
    new_job_id,
    save_job,
    update_job_status,
)
from gpucloud_cli.gpucloud_probe import build_ssh_display
from gpucloud_cli.gpucloud_ssh import quote_remote_path, run_ssh_command
from gpucloud_cli.gpucloud_train import (
    build_remote_launch_command,
    resolve_train_node,
    training_paths,
    validate_training_command,
)

CHECKPOINT_MARKERS: Tuple[str, ...] = (
    "trainer_state.json",
    "pytorch_model.bin",
    "model.safetensors",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "checkpoint.pt",
    "ckpt.pt",
    "state.pt",
)


def _training_section(merged: Dict[str, Any]) -> Dict[str, Any]:
    sec = merged.get("training")
    return sec if isinstance(sec, dict) else {}


def _security(merged: Dict[str, Any]) -> Dict[str, Any]:
    sec = merged.get("security")
    return sec if isinstance(sec, dict) else {}


def _timeout(merged: Dict[str, Any]) -> int:
    return int(_security(merged).get("command_timeout_sec") or 3600)


def _output_limit(merged: Dict[str, Any]) -> int:
    return int(_security(merged).get("max_output_chars") or 8000)


def checkpoint_root(merged: Dict[str, Any], effective_model: str = "") -> str:
    training = _training_section(merged)
    if training.get("checkpoint_dir"):
        return str(training["checkpoint_dir"])

    workdir = "~/gpucloud"
    for cluster in merged.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        for node in cluster.get("nodes") or []:
            if isinstance(node, dict) and node.get("workdir"):
                workdir = str(node["workdir"])
                break
        break
    model = effective_model or str(merged.get("model_name") or "model")
    return f"{workdir}/checkpoints/{model}"


def _bash(script: str) -> str:
    return "bash -lc " + shlex.quote(script)


def _parse_checkpoint_listing(stdout: str) -> tuple[str, List[Dict[str, Any]]]:
    root = ""
    checkpoints: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in (stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("__GPUCLOUD_ROOT__|"):
            root = line.split("|", 1)[1]
            continue
        if line.startswith("__GPUCLOUD_"):
            continue
        mtime_text, sep, path = line.partition("|")
        if not sep or not path or path in seen:
            continue
        seen.add(path)
        try:
            mtime = float(mtime_text)
        except ValueError:
            mtime = 0.0
        checkpoints.append(
            {
                "path": path,
                "name": path.rstrip("/").rsplit("/", 1)[-1],
                "mtime": mtime,
            }
        )
    return root, checkpoints


def _resolve_prepared(
    config_file: Optional[str],
    *,
    allow_discover_without_goal: bool,
) -> tuple[Optional[GpucloudPreparedConfig], Optional[Dict[str, Any]]]:
    try:
        return (
            resolve_config_for_tool(
                config_file,
                allow_discover_without_goal=allow_discover_without_goal,
            ),
            None,
        )
    except GpucloudConfigError as exc:
        return None, {"ok": False, "error": str(exc)}


def run_checkpoint_list(
    *,
    config_file: Optional[str] = None,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    limit: int = 20,
    allow_discover_without_goal: bool = False,
) -> Dict[str, Any]:
    prepared, err = _resolve_prepared(
        config_file,
        allow_discover_without_goal=allow_discover_without_goal,
    )
    if err:
        return err
    assert prepared is not None

    merged = prepared.merged
    try:
        cname, nidx, node = resolve_train_node(
            merged, cluster_name=cluster_name, node_index=node_index
        )
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc)}

    root = checkpoint_root(merged, prepared.effective_model)
    n = max(1, min(int(limit), 200))
    script = "\n".join(
        [
            f"root={quote_remote_path(root)}",
            'printf "__GPUCLOUD_ROOT__|%s\\n" "$root"',
            'if [ ! -d "$root" ]; then',
            '  printf "__GPUCLOUD_MISSING_ROOT__|%s\\n" "$root"',
            "  exit 2",
            "fi",
            (
                'find "$root" -maxdepth 2 -mindepth 1 -type d '
                "-printf '%T@|%p\\n' 2>/dev/null | sort -nr | "
                f"head -n {n}"
            ),
        ]
    )
    remote_cmd = _bash(script)
    result = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=node_ssh_key_path(node),
        remote_command=remote_cmd,
        timeout_sec=min(120, _timeout(merged)),
        output_limit=_output_limit(merged),
    )
    expanded_root, checkpoints = _parse_checkpoint_listing(result.stdout)
    root_for_output = expanded_root or root

    if result.exit_code == 2:
        return {
            "ok": False,
            "error": f"checkpoint root not found: {root_for_output}",
            "checkpoint_root": root_for_output,
            "configured_checkpoint_root": root,
            "ssh": result.as_dict(),
        }
    if not result.ok:
        return {
            "ok": False,
            "error": result.error or f"ssh exit {result.exit_code}",
            "checkpoint_root": root_for_output,
            "configured_checkpoint_root": root,
            "ssh": result.as_dict(),
        }

    return {
        "ok": True,
        "config_path": str(prepared.path),
        "cluster": cname,
        "node_index": nidx,
        "host": node.get("host"),
        "checkpoint_root": root_for_output,
        "configured_checkpoint_root": root,
        "checkpoints": checkpoints,
        "latest": checkpoints[0] if checkpoints else None,
        "count": len(checkpoints),
    }


def run_checkpoint_latest(**kwargs: Any) -> Dict[str, Any]:
    data = run_checkpoint_list(**kwargs)
    if not data.get("ok"):
        return data
    latest = data.get("latest")
    if not latest:
        return {
            **data,
            "ok": False,
            "error": f"no checkpoints found under {data.get('checkpoint_root')}",
        }
    return {**data, "checkpoint": latest}


def _parse_validation_markers(stdout: str) -> tuple[str, List[str]]:
    checkpoint = ""
    markers: List[str] = []
    seen: set[str] = set()
    for raw in (stdout or "").splitlines():
        line = raw.strip()
        if line.startswith("__GPUCLOUD_CHECKPOINT__|"):
            checkpoint = line.split("|", 1)[1]
            continue
        if not line.startswith("marker|"):
            continue
        marker = line.split("|", 1)[1]
        if marker and marker not in seen:
            seen.add(marker)
            markers.append(marker)
    return checkpoint, markers


def run_checkpoint_validate(
    *,
    config_file: Optional[str] = None,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    checkpoint_path: Optional[str] = None,
    allow_discover_without_goal: bool = False,
) -> Dict[str, Any]:
    prepared, err = _resolve_prepared(
        config_file,
        allow_discover_without_goal=allow_discover_without_goal,
    )
    if err:
        return err
    assert prepared is not None

    merged = prepared.merged
    try:
        cname, nidx, node = resolve_train_node(
            merged, cluster_name=cluster_name, node_index=node_index
        )
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc)}

    selected_checkpoint = checkpoint_path
    if not selected_checkpoint:
        latest = run_checkpoint_latest(
            config_file=config_file,
            cluster_name=cluster_name,
            node_index=node_index,
            allow_discover_without_goal=allow_discover_without_goal,
        )
        if not latest.get("ok"):
            return latest
        checkpoint = latest.get("checkpoint") or {}
        selected_checkpoint = checkpoint.get("path")

    if not selected_checkpoint:
        return {"ok": False, "error": "checkpoint_path is required"}

    marker_words = " ".join(shlex.quote(x) for x in CHECKPOINT_MARKERS)
    script = "\n".join(
        [
            f"cp={quote_remote_path(selected_checkpoint)}",
            'printf "__GPUCLOUD_CHECKPOINT__|%s\\n" "$cp"',
            'if [ ! -d "$cp" ]; then',
            '  printf "__GPUCLOUD_MISSING_DIR__|%s\\n" "$cp"',
            "  exit 2",
            "fi",
            "found=0",
            f"for rel in {marker_words}; do",
            '  if [ -e "$cp/$rel" ]; then',
            '    printf "marker|%s\\n" "$rel"',
            "    found=1",
            "  fi",
            "done",
            'for f in "$cp"/*.safetensors "$cp"/*.bin "$cp"/*.pt "$cp"/*.pth; do',
            '  [ -e "$f" ] || continue',
            '  rel="${f#$cp/}"',
            '  printf "marker|%s\\n" "$rel"',
            "  found=1",
            "done",
            'if [ "$found" -eq 0 ]; then',
            '  printf "__GPUCLOUD_NO_MARKERS__|%s\\n" "$cp"',
            "  exit 3",
            "fi",
        ]
    )
    result = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=node_ssh_key_path(node),
        remote_command=_bash(script),
        timeout_sec=min(120, _timeout(merged)),
        output_limit=_output_limit(merged),
    )
    expanded_checkpoint, markers = _parse_validation_markers(result.stdout)
    cp_for_output = expanded_checkpoint or selected_checkpoint

    if result.exit_code == 2:
        return {
            "ok": False,
            "error": f"checkpoint not found: {cp_for_output}",
            "checkpoint_path": cp_for_output,
            "required_any": list(CHECKPOINT_MARKERS),
            "ssh": result.as_dict(),
        }
    if result.exit_code == 3:
        return {
            "ok": False,
            "error": f"checkpoint appears damaged or incomplete: {cp_for_output}",
            "checkpoint_path": cp_for_output,
            "markers": [],
            "required_any": list(CHECKPOINT_MARKERS),
            "ssh": result.as_dict(),
        }
    if not result.ok:
        return {
            "ok": False,
            "error": result.error or f"ssh exit {result.exit_code}",
            "checkpoint_path": cp_for_output,
            "ssh": result.as_dict(),
        }

    return {
        "ok": True,
        "config_path": str(prepared.path),
        "cluster": cname,
        "node_index": nidx,
        "checkpoint_path": cp_for_output,
        "markers": markers,
        "valid": bool(markers),
    }


def build_resume_training_command(
    merged: Dict[str, Any],
    checkpoint_path: str,
) -> str:
    training = _training_section(merged)
    base = generate_training_command(merged).strip()
    cp = quote_remote_path(checkpoint_path)
    if "{checkpoint_path}" in base or "{checkpoint}" in base:
        return base.replace("{checkpoint_path}", cp).replace("{checkpoint}", cp)
    resume_arg = str(training.get("resume_arg") or "--resume_from_checkpoint").strip()
    return f"{base} {resume_arg} {cp}".strip()


def plan_train_resume(
    prepared: GpucloudPreparedConfig,
    *,
    checkpoint_path: str,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    merged = prepared.merged
    try:
        cname, nidx, node = resolve_train_node(
            merged, cluster_name=cluster_name, node_index=node_index
        )
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc)}

    command = build_resume_training_command(merged, checkpoint_path)
    err = validate_training_command(command)
    if err:
        return {"ok": False, "error": err}

    jid = job_id or new_job_id()
    workdir, log_path, checkpoint_root_path = training_paths(merged, jid)
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
        "checkpoint_path": checkpoint_root_path,
        "source_checkpoint": checkpoint_path,
        "launch_command": command,
        "remote_launch_command": remote_cmd,
        "ssh_command": build_ssh_display(node, remote_cmd),
        "dataset": prepared.effective_dataset,
        "model": prepared.effective_model,
    }


def run_train_resume(
    *,
    config_file: Optional[str] = None,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    checkpoint_path: Optional[str] = None,
    dry_run: Optional[bool] = None,
    confirm_execute: bool = False,
    allow_discover_without_goal: bool = False,
) -> Dict[str, Any]:
    prepared, err = _resolve_prepared(
        config_file,
        allow_discover_without_goal=allow_discover_without_goal,
    )
    if err:
        return err
    assert prepared is not None

    validation = run_checkpoint_validate(
        config_file=config_file,
        cluster_name=cluster_name,
        node_index=node_index,
        checkpoint_path=checkpoint_path,
        allow_discover_without_goal=allow_discover_without_goal,
    )
    if not validation.get("ok"):
        return validation

    source_checkpoint = str(validation["checkpoint_path"])
    plan = plan_train_resume(
        prepared,
        checkpoint_path=source_checkpoint,
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
            "validation": validation,
            "message": (
                "Dry-run only. Re-run with confirm_execute=true "
                "(or gpucloud checkpoint resume --yes) to launch from this checkpoint."
            ),
        }

    cname, nidx, node = resolve_train_node(
        merged, cluster_name=cluster_name, node_index=node_index
    )
    job = TrainingJob(
        job_id=plan["job_id"],
        job_type="train_resume",
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

    result = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=node_ssh_key_path(node),
        remote_command=plan["remote_launch_command"],
        timeout_sec=min(120, _timeout(merged)),
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
            "validation": validation,
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
        "source_checkpoint": source_checkpoint,
        "validation": validation,
        "message": f"Resume job {job.job_id} started (pid={pid or 'unknown'})",
    }


def _checkpoint_path_is_under(root: str, path: str) -> bool:
    clean_root = str(root or "").rstrip("/")
    clean_path = str(path or "").rstrip("/")
    return bool(clean_root and clean_path.startswith(clean_root + "/"))


def run_checkpoint_cleanup(
    *,
    config_file: Optional[str] = None,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    keep: int = 3,
    dry_run: Optional[bool] = None,
    confirm_delete: bool = False,
    allow_discover_without_goal: bool = False,
) -> Dict[str, Any]:
    listed = run_checkpoint_list(
        config_file=config_file,
        cluster_name=cluster_name,
        node_index=node_index,
        limit=max(1, int(keep)) + 100,
        allow_discover_without_goal=allow_discover_without_goal,
    )
    if not listed.get("ok"):
        return listed

    checkpoints = listed.get("checkpoints") or []
    keep_n = max(0, int(keep))
    delete_items = checkpoints[keep_n:]
    root = str(listed.get("checkpoint_root") or "")
    delete_paths = [
        str(item.get("path"))
        for item in delete_items
        if _checkpoint_path_is_under(root, str(item.get("path")))
    ]

    prepared, err = _resolve_prepared(
        config_file,
        allow_discover_without_goal=allow_discover_without_goal,
    )
    if err:
        return err
    assert prepared is not None
    sec = _security(prepared.merged)
    do_dry = sec.get("dry_run_required", True) if dry_run is None else bool(dry_run)
    delete_command = (
        "rm -rf -- " + " ".join(quote_remote_path(path) for path in delete_paths)
        if delete_paths
        else ""
    )

    if not delete_paths:
        return {
            "ok": True,
            "dry_run": True,
            "checkpoint_root": root,
            "keep": keep_n,
            "delete_count": 0,
            "delete_paths": [],
            "message": "No checkpoints selected for cleanup.",
        }

    plan = {
        "ok": True,
        "dry_run": True,
        "checkpoint_root": root,
        "keep": keep_n,
        "delete_count": len(delete_paths),
        "delete_paths": delete_paths,
        "delete_command": delete_command,
    }
    if do_dry or not confirm_delete:
        return {
            **plan,
            "message": (
                "Dry-run only. Re-run with confirm_delete=true "
                "(or gpucloud checkpoint cleanup --yes) to delete remote checkpoints."
            ),
        }

    cname, nidx, node = resolve_train_node(
        prepared.merged, cluster_name=cluster_name, node_index=node_index
    )
    result = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=node_ssh_key_path(node),
        remote_command=delete_command,
        timeout_sec=min(120, _timeout(prepared.merged)),
        output_limit=_output_limit(prepared.merged),
    )
    return {
        **plan,
        "dry_run": False,
        "ok": result.ok,
        "cluster": cname,
        "node_index": nidx,
        "ssh": result.as_dict(),
        "error": result.error,
    }


__all__ = [
    "CHECKPOINT_MARKERS",
    "checkpoint_root",
    "run_checkpoint_list",
    "run_checkpoint_latest",
    "run_checkpoint_validate",
    "build_resume_training_command",
    "plan_train_resume",
    "run_train_resume",
    "run_checkpoint_cleanup",
]
