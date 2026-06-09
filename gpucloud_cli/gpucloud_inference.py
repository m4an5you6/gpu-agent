"""GPUCLOUD vLLM inference service management (phase 8)."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from gpucloud_cli.gpucloud_config import GpucloudConfigError, GpucloudPreparedConfig
from gpucloud_cli.gpucloud_context import (
    iter_cluster_nodes,
    node_ssh_key_path,
    resolve_config_for_tool,
)
from gpucloud_cli.gpucloud_jobs import (
    TrainingJob,
    get_job,
    list_recent_jobs,
    new_job_id,
    save_job,
    update_job_status,
)
from gpucloud_cli.gpucloud_probe import build_ssh_display
from gpucloud_cli.gpucloud_ssh import quote_remote_path, run_ssh_command

INFERENCE_JOB_TYPE = "inference"


def _inference_section(merged: Dict[str, Any]) -> Dict[str, Any]:
    sec = merged.get("inference")
    return sec if isinstance(sec, dict) else {}


def _security(merged: Dict[str, Any]) -> Dict[str, Any]:
    sec = merged.get("security")
    return sec if isinstance(sec, dict) else {}


def _timeout(merged: Dict[str, Any]) -> int:
    return int(_security(merged).get("command_timeout_sec") or 3600)


def _output_limit(merged: Dict[str, Any]) -> int:
    return int(_security(merged).get("max_output_chars") or 8000)


def resolve_inference_node(
    merged: Dict[str, Any],
    *,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
) -> Tuple[str, int, Dict[str, Any]]:
    for cname, idx, node in iter_cluster_nodes(merged, cluster_name=cluster_name):
        if idx == node_index:
            return cname, idx, node
    raise GpucloudConfigError(f"node index {node_index} not found")


def _first_workdir(merged: Dict[str, Any]) -> str:
    for _c, _i, node in iter_cluster_nodes(merged):
        return str(node.get("workdir") or "~/gpucloud")
    return "~/gpucloud"


def _coerce_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("inference.port must be an integer") from exc
    if port < 1 or port > 65535:
        raise ValueError("inference.port must be between 1 and 65535")
    return port


def _coerce_tensor_parallel(value: Any) -> int:
    try:
        tp = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("inference.tensor_parallel must be an integer") from exc
    if tp < 1:
        raise ValueError("inference.tensor_parallel must be >= 1")
    return tp


def inference_settings(merged: Dict[str, Any]) -> Dict[str, Any]:
    inf = _inference_section(merged)
    engine = str(inf.get("engine") or "vllm").strip().lower()
    port = _coerce_port(inf.get("port", 8000))
    tensor_parallel = _coerce_tensor_parallel(inf.get("tensor_parallel", 1))
    model_path = str(inf.get("model_path") or "").strip()
    if not model_path:
        model_name = inf.get("model_name") or merged.get("model_name") or "model"
        model_path = f"{_first_workdir(merged)}/models/{model_name}"
    host = str(inf.get("host") or "0.0.0.0").strip() or "0.0.0.0"
    return {
        "engine": engine,
        "port": port,
        "tensor_parallel": tensor_parallel,
        "model_path": model_path,
        "host": host,
        "served_model_name": str(inf.get("served_model_name") or "").strip(),
        "command": str(inf.get("command") or "").strip(),
        "extra_args": inf.get("extra_args"),
    }


def validate_inference_engine(merged: Dict[str, Any]) -> Optional[str]:
    settings = inference_settings(merged)
    if settings["engine"] != "vllm":
        return (
            "phase 8 supports inference.engine=vllm only "
            f"(got {settings['engine']!r})"
        )
    return None


def build_vllm_command(merged: Dict[str, Any]) -> str:
    settings = inference_settings(merged)
    command = settings.get("command") or ""
    model_path = str(settings["model_path"])
    port = str(settings["port"])
    tensor_parallel = str(settings["tensor_parallel"])
    if command:
        return (
            command.replace("{model_path}", quote_remote_path(model_path))
            .replace("{port}", port)
            .replace("{tensor_parallel}", tensor_parallel)
        )

    parts = [
        "vllm",
        "serve",
        quote_remote_path(model_path),
        "--host",
        shlex.quote(str(settings["host"])),
        "--port",
        port,
        "--tensor-parallel-size",
        tensor_parallel,
    ]
    if settings["served_model_name"]:
        parts.extend(["--served-model-name", shlex.quote(settings["served_model_name"])])

    extra = settings.get("extra_args")
    if isinstance(extra, list):
        parts.extend(shlex.quote(str(item)) for item in extra if str(item).strip())
    elif isinstance(extra, str) and extra.strip():
        parts.append(extra.strip())

    return " ".join(parts)


def validate_vllm_command(command: str) -> Optional[str]:
    text = (command or "").strip()
    if not text:
        return "inference command is empty"
    first = text.split(None, 1)[0]
    if first == "vllm":
        return None
    if first == "python" and ("-m vllm" in text or " vllm." in text):
        return None
    return "phase 8 inference launch command must use vLLM"


def inference_paths(merged: Dict[str, Any], job_id: str) -> Tuple[str, str]:
    inf = _inference_section(merged)
    workdir = _first_workdir(merged)
    log_dir = str(inf.get("log_dir") or f"{workdir}/logs/inference")
    return workdir, f"{log_dir}/{job_id}.log"


def build_remote_inference_launch_command(
    *,
    workdir: str,
    log_path: str,
    inference_command: str,
) -> str:
    inner = inference_command.replace("'", "'\"'\"'")
    return (
        f"cd {quote_remote_path(workdir)} && "
        f"mkdir -p {quote_remote_path(str(Path(log_path).parent))} && "
        f"nohup bash -lc '{inner}' >> {quote_remote_path(log_path)} 2>&1 & echo $!"
    )


def plan_inference_service(
    prepared: GpucloudPreparedConfig,
    *,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    merged = prepared.merged
    try:
        engine_error = validate_inference_engine(merged)
        if engine_error:
            return {"ok": False, "error": engine_error}
        settings = inference_settings(merged)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    cname, nidx, node = resolve_inference_node(
        merged, cluster_name=cluster_name, node_index=node_index
    )
    jid = job_id or new_job_id("infer")
    command = build_vllm_command(merged)
    err = validate_vllm_command(command)
    if err:
        return {"ok": False, "error": err}

    workdir, log_path = inference_paths(merged, jid)
    remote_cmd = build_remote_inference_launch_command(
        workdir=workdir,
        log_path=log_path,
        inference_command=command,
    )
    service_url = f"http://{node.get('host')}:{settings['port']}"

    return {
        "ok": True,
        "job_id": jid,
        "job_type": INFERENCE_JOB_TYPE,
        "cluster": cname,
        "node_index": nidx,
        "host": node.get("host"),
        "workdir": workdir,
        "log_path": log_path,
        "launch_command": command,
        "remote_launch_command": remote_cmd,
        "ssh_command": build_ssh_display(node, remote_cmd),
        "engine": settings["engine"],
        "model_path": settings["model_path"],
        "port": settings["port"],
        "tensor_parallel": settings["tensor_parallel"],
        "service_url": service_url,
        "node": {
            "user": node.get("user"),
            "port": node.get("port"),
            "ssh_key_path": str(node.get("ssh_key", "")),
        },
    }


def run_infer_start(
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

    plan = plan_inference_service(
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
                "Dry-run only. Re-run with confirm_execute=true "
                "(or gpucloud infer start --yes) to launch vLLM."
            ),
        }

    cname, nidx, node = resolve_inference_node(
        merged, cluster_name=cluster_name, node_index=node_index
    )
    job = TrainingJob(
        job_id=plan["job_id"],
        job_type=INFERENCE_JOB_TYPE,
        cluster=cname,
        status="pending",
        launch_command=plan["launch_command"],
        workdir=plan["workdir"],
        log_path=plan["log_path"],
        checkpoint_path="",
        node_index=nidx,
        host=str(node.get("host", "")),
        model=str(plan["model_path"]),
        port=int(plan["port"]),
        service_url=str(plan["service_url"]),
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
        "service_url": job.service_url,
        "log_path": job.log_path,
        "message": f"vLLM service {job.job_id} started (pid={pid or 'unknown'})",
    }


def _select_inference_job(job_id: Optional[str]) -> Tuple[Optional[TrainingJob], Optional[str]]:
    if job_id:
        job = get_job(job_id)
        if not job:
            return None, f"job not found: {job_id}"
        if job.job_type != INFERENCE_JOB_TYPE:
            return None, f"job is not an inference service: {job_id}"
        return job, None
    jobs = list_recent_jobs(limit=1, job_type=INFERENCE_JOB_TYPE)
    if not jobs:
        return None, "no inference services found"
    return jobs[0], None


def run_infer_status(job_id: Optional[str] = None, *, limit: int = 10) -> Dict[str, Any]:
    if job_id:
        job, error = _select_inference_job(job_id)
        if error:
            return {"ok": False, "error": error}
        assert job is not None
        return {"ok": True, "job": job.to_dict()}
    jobs = list_recent_jobs(limit=limit, job_type=INFERENCE_JOB_TYPE)
    return {"ok": True, "jobs": [j.to_dict() for j in jobs], "count": len(jobs)}


def build_health_command(port: int) -> str:
    code = (
        "import urllib.request;"
        f"url='http://127.0.0.1:{int(port)}/health';"
        "r=urllib.request.urlopen(url, timeout=5);"
        "body=r.read(512).decode('utf-8','replace');"
        "print(r.status);"
        "print(body)"
    )
    return "python -c " + shlex.quote(code)


def run_infer_health(
    job_id: Optional[str] = None,
    *,
    config_file: Optional[str] = None,
    allow_discover_without_goal: bool = False,
) -> Dict[str, Any]:
    job, error = _select_inference_job(job_id)
    if error:
        return {"ok": False, "error": error}
    assert job is not None

    try:
        prepared = resolve_config_for_tool(
            config_file,
            allow_discover_without_goal=allow_discover_without_goal,
        )
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc), "job_id": job.job_id}

    merged = prepared.merged
    _c, _i, node = resolve_inference_node(
        merged, cluster_name=job.cluster or None, node_index=job.node_index
    )
    port = int(job.port or inference_settings(merged)["port"])
    result = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=node_ssh_key_path(node),
        remote_command=build_health_command(port),
        timeout_sec=min(120, _timeout(merged)),
        output_limit=_output_limit(merged),
    )
    first_line = (result.stdout or "").splitlines()[0:1]
    status_code = None
    if first_line:
        try:
            status_code = int(first_line[0].strip())
        except ValueError:
            status_code = None
    healthy = result.ok and (status_code is None or 200 <= status_code < 400)
    return {
        "ok": healthy,
        "healthy": healthy,
        "job_id": job.job_id,
        "port": port,
        "service_url": job.service_url or f"http://{job.host}:{port}",
        "status_code": status_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "truncated": result.truncated,
        "error": result.error,
    }


def build_stop_command(pid: str) -> str:
    text = str(pid or "").strip()
    if not text.isdigit():
        raise ValueError("remote_pid is missing; cannot safely stop service")
    return (
        f"kill {text} 2>/dev/null || true; "
        "sleep 1; "
        f"if kill -0 {text} 2>/dev/null; then kill -TERM {text} 2>/dev/null || true; fi; "
        f"echo stopped {text}"
    )


def run_infer_stop(
    job_id: Optional[str] = None,
    *,
    config_file: Optional[str] = None,
    dry_run: Optional[bool] = None,
    confirm_stop: bool = False,
    allow_discover_without_goal: bool = False,
) -> Dict[str, Any]:
    job, error = _select_inference_job(job_id)
    if error:
        return {"ok": False, "error": error}
    assert job is not None

    try:
        stop_command = build_stop_command(job.remote_pid or "")
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "job": job.to_dict()}

    try:
        prepared = resolve_config_for_tool(
            config_file,
            allow_discover_without_goal=allow_discover_without_goal,
        )
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc), "job_id": job.job_id}

    merged = prepared.merged
    _c, _i, node = resolve_inference_node(
        merged, cluster_name=job.cluster or None, node_index=job.node_index
    )
    sec = _security(merged)
    do_dry = sec.get("dry_run_required", True) if dry_run is None else bool(dry_run)
    plan = {
        "ok": True,
        "job_id": job.job_id,
        "job": job.to_dict(),
        "stop_command": stop_command,
        "ssh_command": build_ssh_display(node, stop_command),
    }
    if do_dry or not confirm_stop:
        return {
            **plan,
            "dry_run": True,
            "message": (
                "Dry-run only. Re-run with confirm_stop=true "
                "(or gpucloud infer stop --yes) to stop the service."
            ),
        }

    result = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=node_ssh_key_path(node),
        remote_command=stop_command,
        timeout_sec=min(120, _timeout(merged)),
        output_limit=512,
    )
    if not result.ok:
        update_job_status(job.job_id, "failed", last_error=result.error)
        job = get_job(job.job_id) or job
        return {
            "ok": False,
            "dry_run": False,
            "job": job.to_dict(),
            "error": result.error,
            "ssh": result.as_dict(),
        }

    update_job_status(job.job_id, "stopped")
    job = get_job(job.job_id) or job
    return {
        "ok": True,
        "dry_run": False,
        "job": job.to_dict(),
        "stdout": result.stdout,
        "message": f"vLLM service {job.job_id} stopped",
    }
