"""Local GPUCLOUD distributed Megatron worker runtime."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hermes_constants import get_hermes_home
from hermes_cli.gpucloud_distributed import (
    build_worker_plan,
    redact_text,
    write_worker_runtime_artifacts,
    worker_env,
)
from hermes_cli.gpucloud_probe import probe_local_gpu
from hermes_cli.gpucloud_worker_task import WorkerTask, WorkerTaskError, load_worker_task

WORKER_STATUSES = frozenset({"pending", "running", "failed", "stopped", "completed"})


def _now() -> float:
    return time.time()


def _expand_path(value: Any) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def _safe_job_id(job_id: str) -> str:
    safe = "".join(ch for ch in str(job_id) if ch.isalnum() or ch in "-_")
    return safe or "worker-job"


def worker_state_dir() -> Path:
    override = os.environ.get("GPUCLOUD_WORKER_STATE_DIR", "").strip()
    path = Path(override).expanduser() if override else get_hermes_home() / "gpucloud" / "worker_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def worker_state_path(job_id: str) -> Path:
    return worker_state_dir() / f"{_safe_job_id(job_id)}.json"


@dataclass
class PreflightCheck:
    name: str
    ok: bool
    severity: str = "error"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "severity": self.severity,
            "detail": self.detail,
        }


def _check(checks: List[PreflightCheck], name: str, ok: bool, detail: str = "", *, severity: str = "error") -> None:
    checks.append(PreflightCheck(name=name, ok=bool(ok), severity=severity, detail=detail))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_worker_state(job_id: str) -> Optional[Dict[str, Any]]:
    return _read_json(worker_state_path(job_id))


def save_worker_state(state: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(state.get("job_id") or "")
    if not job_id:
        raise ValueError("worker state requires job_id")
    state["updated_at"] = _now()
    path = worker_state_path(job_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return state


def _pid_running(pid: Union[int, str, None]) -> bool:
    if pid in (None, ""):
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OSError):
        return False
    return True


def _terminate_pid(pid: Union[int, str], timeout_sec: float = 10.0) -> str:
    ipid = int(pid)
    try:
        os.killpg(ipid, signal.SIGTERM)
    except ProcessLookupError:
        return "not-running"
    except OSError:
        try:
            os.kill(ipid, signal.SIGTERM)
        except ProcessLookupError:
            return "not-running"

    deadline = time.time() + max(0.5, timeout_sec)
    while time.time() < deadline:
        if not _pid_running(ipid):
            return "terminated"
        time.sleep(0.1)

    try:
        os.killpg(ipid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(ipid, signal.SIGKILL)
        except OSError:
            pass
    return "killed"


def _port_bindable(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _connectable(host: str, port: int, timeout_sec: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _path_writable(path: Path, *, directory: bool = True) -> bool:
    try:
        if directory:
            path.mkdir(parents=True, exist_ok=True)
        probe = path / ".gpucloud-write-test" if directory else path
        if directory:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _torch_probe(python: str) -> Dict[str, Any]:
    script = (
        "import json, torch; "
        "print(json.dumps({"
        "'torch_version': getattr(torch, '__version__', ''), "
        "'cuda_version': getattr(torch.version, 'cuda', None), "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'distributed_available': bool(torch.distributed.is_available()), "
        "'nccl_available': bool(getattr(torch.distributed, 'is_nccl_available', lambda: False)())"
        "}))"
    )
    try:
        proc = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {
            "ok": False,
            "exit_code": proc.returncode,
            "stderr": (proc.stderr or "").strip()[:1000],
        }
    try:
        data = json.loads((proc.stdout or "{}").strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "error": "unable to parse torch probe output"}
    data["ok"] = True
    return data


def _entrypoint_path(task: WorkerTask) -> Path:
    entrypoint = str(task.training.get("entrypoint") or "pretrain_gpt.py")
    if entrypoint.startswith("/"):
        return _expand_path(entrypoint)
    return _expand_path(task.runtime.get("megatron_lm_dir")) / entrypoint


def _gpu_memory_mib(gpu: Dict[str, Any]) -> Optional[int]:
    raw = gpu.get("memory_total_mib")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def run_worker_preflight(
    *,
    task_file: Union[str, Path],
    check_network: bool = True,
) -> Dict[str, Any]:
    try:
        task = load_worker_task(task_file)
    except WorkerTaskError as exc:
        return {"ok": False, "error": str(exc), "errors": list(exc.errors)}

    checks: List[PreflightCheck] = []

    workdir = _expand_path(task.runtime["workdir"])
    data_path = _expand_path(task.training["data_path"])
    checkpoint_dir = _expand_path(task.training["checkpoint_dir"])
    log_dir = _expand_path(task.training["log_dir"])
    entrypoint = _entrypoint_path(task)
    python = str(task.runtime.get("python") or "python")

    gpu = probe_local_gpu()
    gpus = gpu.get("gpus") if isinstance(gpu.get("gpus"), list) else []
    gpu_count = int(gpu.get("gpu_count") or len(gpus) or 0)
    required_gpu_count = int(task.preflight.get("require_gpu_count") or 1)
    _check(
        checks,
        "gpu_count",
        bool(gpu.get("available")) and gpu_count >= required_gpu_count,
        f"detected={gpu_count} required={required_gpu_count}",
    )

    policy = str(task.preflight.get("heterogeneous_policy") or "warn").strip().lower()
    hetero_severity = "error" if policy == "reject" else ("warning" if policy == "warn" else "info")

    min_vram_gb = float(task.preflight.get("min_vram_gb") or 0)
    if min_vram_gb > 0 and gpus:
        min_vram_mib = int(min_vram_gb * 1024)
        bad = [
            str(g.get("name") or g.get("index"))
            for g in gpus
            if (_gpu_memory_mib(g) or 0) < min_vram_mib
        ]
        _check(
            checks,
            "gpu_vram",
            not bad,
            f"min_vram_gb={min_vram_gb} below_min={bad}",
            severity=hetero_severity,
        )
    elif min_vram_gb > 0:
        _check(
            checks,
            "gpu_vram",
            False,
            f"min_vram_gb={min_vram_gb}; no GPU memory data",
            severity=hetero_severity,
        )
    else:
        _check(checks, "gpu_vram", True, "no minimum VRAM requested", severity="warning")

    expected_name = (
        task.preflight.get("expected_gpu_name")
        or task.preflight.get("gpu_type_expected")
        or task.preflight.get("gpu_name")
    )
    if expected_name and gpus:
        expected = str(expected_name).strip().lower()
        mismatched = [
            str(g.get("name") or g.get("index"))
            for g in gpus
            if expected not in str(g.get("name") or "").lower()
        ]
        _check(
            checks,
            "gpu_type",
            not mismatched,
            f"expected={expected_name} mismatched={mismatched}",
            severity=hetero_severity,
        )
    elif expected_name:
        _check(
            checks,
            "gpu_type",
            False,
            f"expected={expected_name}; no GPU name data",
            severity=hetero_severity,
        )

    _check(checks, "heterogeneous_policy", True, policy, severity="warning")

    _check(checks, "workdir_writable", _path_writable(workdir), str(workdir))
    auto_data = bool(
        task.training_runner == "swift_megatron"
        or task.training.get("dataset_config")
        or (isinstance(task.training.get("megatron"), dict) and task.training["megatron"].get("auto_data"))
    )
    _check(
        checks,
        "data_path_readable",
        auto_data or (data_path.exists() and os.access(data_path, os.R_OK)),
        str(data_path) if not auto_data else f"{data_path} (auto_data/runner-managed)",
        severity="warning" if auto_data else "error",
    )
    _check(checks, "checkpoint_dir_writable", _path_writable(checkpoint_dir), str(checkpoint_dir))
    _check(checks, "log_dir_writable", _path_writable(log_dir), str(log_dir))
    if task.training_runner == "swift_megatron":
        _check(checks, "megatron_swift_cli", shutil.which("megatron") is not None, "megatron CLI in PATH")
        if task.nnodes > 1:
            shared_cache = str(worker_env(task).get("MODELSCOPE_CACHE") or "").strip()
            _check(
                checks,
                "modelscope_cache_shared",
                bool(shared_cache),
                "MODELSCOPE_CACHE must point to shared storage for multi-node Megatron-SWIFT",
            )
    else:
        _check(checks, "megatron_entrypoint", entrypoint.is_file(), str(entrypoint))

    torch = _torch_probe(python)
    _check(checks, "torch_import", bool(torch.get("ok")), json.dumps(torch, ensure_ascii=False))
    if torch.get("ok"):
        _check(checks, "torch_cuda", bool(torch.get("cuda_available")), json.dumps(torch, ensure_ascii=False))
        _check(checks, "torch_distributed", bool(torch.get("distributed_available")), json.dumps(torch, ensure_ascii=False))
        _check(checks, "torch_nccl", bool(torch.get("nccl_available")), json.dumps(torch, ensure_ascii=False))

    if check_network:
        if task.node_rank == 0:
            _check(
                checks,
                "rendezvous_port_bindable",
                _port_bindable(task.master_addr, task.master_port),
                f"{task.master_addr}:{task.master_port}",
            )
        else:
            _check(
                checks,
                "rendezvous_reachable",
                _connectable(task.master_addr, task.master_port),
                f"{task.master_addr}:{task.master_port}",
            )
    else:
        _check(checks, "rendezvous_network", True, "skipped", severity="warning")

    checks_out = [c.to_dict() for c in checks]
    ok = all(c.ok or c.severity != "error" for c in checks)
    return {
        "ok": ok,
        "task": task.summary(),
        "node_rank": task.node_rank,
        "job_id": task.job_id,
        "checks": checks_out,
        "gpu": gpu,
        "message": "preflight passed" if ok else "preflight failed",
    }


def run_worker_wait(
    *,
    task_file: Union[str, Path],
    timeout_sec: int = 30,
    poll_sec: float = 2.0,
    wait_for_master: bool = False,
) -> Dict[str, Any]:
    path = Path(task_file).expanduser()
    deadline = time.time() + max(0, int(timeout_sec))
    while True:
        if path.is_file():
            try:
                task = load_worker_task(path)
            except WorkerTaskError as exc:
                return {"ok": False, "error": str(exc), "errors": list(exc.errors)}
            if wait_for_master and task.node_rank != 0:
                if not _connectable(task.master_addr, task.master_port):
                    if time.time() >= deadline:
                        return {
                            "ok": False,
                            "error": "rendezvous not reachable before timeout",
                            "task": task.summary(),
                        }
                    time.sleep(max(0.1, poll_sec))
                    continue
            return {"ok": True, "task": task.summary(), "message": "worker task ready"}
        if time.time() >= deadline:
            return {"ok": False, "error": f"worker task file not found: {path}"}
        time.sleep(max(0.1, poll_sec))


def run_worker_dry_run(*, task_file: Union[str, Path]) -> Dict[str, Any]:
    try:
        task = load_worker_task(task_file)
    except WorkerTaskError as exc:
        return {"ok": False, "error": str(exc), "errors": list(exc.errors)}
    plan = build_worker_plan(task)
    return {
        **plan,
        "dry_run": True,
        "message": "Dry-run only. Re-run with gpucloud worker start --yes to launch this local rank.",
    }


def _initial_state(task: WorkerTask, plan: Dict[str, Any], *, status: str) -> Dict[str, Any]:
    now = _now()
    env = worker_env(task)
    return {
        "job_id": task.job_id,
        "framework": task.framework,
        "role": task.role,
        "node_rank": task.node_rank,
        "nnodes": task.nnodes,
        "nproc_per_node": task.nproc_per_node,
        "master_addr": task.master_addr,
        "master_port": task.master_port,
        "status": status,
        "pid": None,
        "workdir": plan["workdir"],
        "launch_command": redact_text(plan["launch_command"], env),
        "log_path": plan["log_path"],
        "checkpoint_dir": plan["checkpoint_dir"],
        "started_at": None,
        "created_at": now,
        "updated_at": now,
        "exit_code": None,
        "last_error": None,
        "heartbeat_at": now,
    }


def run_worker_start(
    *,
    task_file: Union[str, Path],
    confirm_execute: bool = False,
    skip_preflight: bool = False,
) -> Dict[str, Any]:
    if not confirm_execute:
        return {
            "ok": False,
            "error": "refusing to start without confirm_execute=true or --yes",
        }
    try:
        task = load_worker_task(task_file)
    except WorkerTaskError as exc:
        return {"ok": False, "error": str(exc), "errors": list(exc.errors)}

    plan = build_worker_plan(task)
    state = _initial_state(task, plan, status="pending")
    state["exit_code_path"] = str(_expand_path(plan["log_path"]).with_suffix(".exitcode"))
    state["training_runner"] = task.training_runner
    state["runtime_artifacts"] = {}
    save_worker_state(state)

    if not skip_preflight:
        preflight = run_worker_preflight(task_file=task.path)
        if not preflight.get("ok"):
            state["status"] = "failed"
            state["last_error"] = "preflight failed"
            save_worker_state(state)
            return {"ok": False, "error": "preflight failed", "preflight": preflight, "job": state}

    env = os.environ.copy()
    env.update(worker_env(task))
    try:
        state["runtime_artifacts"] = write_worker_runtime_artifacts(task, plan)
        save_worker_state(state)
        proc = launch_wrapped_command(
            command=plan["launch_command"],
            workdir=plan["workdir"],
            log_path=plan["log_path"],
            exit_code_path=state["exit_code_path"],
            env=env,
        )
    except OSError as exc:
        state["status"] = "failed"
        state["last_error"] = str(exc)
        save_worker_state(state)
        return {"ok": False, "error": str(exc), "job": state}

    state["status"] = "running"
    state["pid"] = proc.pid
    state["started_at"] = _now()
    state["heartbeat_at"] = state["started_at"]
    save_worker_state(state)
    return {
        "ok": True,
        "dry_run": False,
        "job": state,
        "pid": proc.pid,
        "log_path": str(_expand_path(plan["log_path"])),
        "checkpoint_dir": plan["checkpoint_dir"],
        "message": f"Worker job {task.job_id} rank {task.node_rank} started (pid={proc.pid})",
    }


def run_worker_status(*, job_id: str) -> Dict[str, Any]:
    if not job_id:
        return {"ok": False, "error": "job_id is required"}
    state = load_worker_state(job_id)
    if not state:
        return {"ok": False, "error": f"worker job not found: {job_id}"}

    status = str(state.get("status") or "pending")
    pid = state.get("pid")
    exit_code = read_exit_code(state.get("exit_code_path"))
    if exit_code is not None:
        state["exit_code"] = exit_code
        if status == "running":
            state["status"] = "completed" if exit_code == 0 else "failed"
            if exit_code != 0:
                state["last_error"] = f"worker process exited {exit_code}"
            state["heartbeat_at"] = _now()
            save_worker_state(state)
        return {"ok": True, "job": state, "running": False}

    running = _pid_running(pid)
    if status == "running" and not running:
        state["status"] = "completed"
        state["heartbeat_at"] = _now()
        save_worker_state(state)
    elif status == "running":
        state["heartbeat_at"] = _now()
        save_worker_state(state)
    return {"ok": True, "job": state, "running": running}


def _tail_text(path: Path, lines: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    split = content.splitlines()
    return "\n".join(split[-max(1, min(int(lines), 500)):])


def read_exit_code(path: Union[str, Path, None]) -> Optional[int]:
    if not path:
        return None
    try:
        text = _expand_path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def launch_wrapped_command(
    *,
    command: str,
    workdir: Union[str, Path],
    log_path: Union[str, Path],
    exit_code_path: Union[str, Path],
    env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    workdir_path = _expand_path(workdir)
    log_file = _expand_path(log_path)
    exit_file = _expand_path(exit_code_path)
    workdir_path.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    exit_file.parent.mkdir(parents=True, exist_ok=True)
    exit_file.unlink(missing_ok=True)

    wrapper = (
        f"cd {shlex.quote(str(workdir_path))} && "
        f"( {command} ) >> {shlex.quote(str(log_file))} 2>&1; "
        "code=$?; "
        f"printf '%s\\n' \"$code\" > {shlex.quote(str(exit_file))}; "
        "exit \"$code\""
    )
    return subprocess.Popen(
        ["bash", "-lc", wrapper],
        cwd=str(workdir_path),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def run_worker_logs(*, job_id: str, lines: int = 50) -> Dict[str, Any]:
    state = load_worker_state(job_id)
    if not state:
        return {"ok": False, "error": f"worker job not found: {job_id}"}
    log_path = _expand_path(state.get("log_path") or "")
    return {
        "ok": True,
        "job_id": job_id,
        "log_path": str(log_path),
        "tail": _tail_text(log_path, lines),
        "lines": max(1, min(int(lines), 500)),
    }


def run_worker_stop(*, job_id: str, confirm_stop: bool = False) -> Dict[str, Any]:
    if not confirm_stop:
        return {"ok": False, "error": "refusing to stop without confirm_stop=true or --yes"}
    state = load_worker_state(job_id)
    if not state:
        return {"ok": False, "error": f"worker job not found: {job_id}"}
    pid = state.get("pid")
    if pid and _pid_running(pid):
        result = _terminate_pid(pid)
    else:
        result = "not-running"
    state["status"] = "stopped"
    state["last_error"] = None
    state["heartbeat_at"] = _now()
    save_worker_state(state)
    return {"ok": True, "job": state, "stop_result": result}


__all__ = [
    "WORKER_STATUSES",
    "launch_wrapped_command",
    "load_worker_state",
    "read_exit_code",
    "run_worker_dry_run",
    "run_worker_logs",
    "run_worker_preflight",
    "run_worker_start",
    "run_worker_status",
    "run_worker_stop",
    "run_worker_wait",
    "save_worker_state",
    "worker_state_dir",
    "worker_state_path",
]
