"""GPUCLOUD /goal orchestration for local worker agents."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Union

from hermes_constants import get_hermes_home
from hermes_cli.gpucloud_distributed import build_worker_plan, quote_shell_arg, worker_env
from hermes_cli.gpucloud_goal import infer_goal_intent
from hermes_cli.gpucloud_worker import (
    _expand_path,
    _pid_running,
    _safe_job_id,
    _tail_text,
    launch_wrapped_command,
    read_exit_code,
    run_worker_dry_run,
    run_worker_preflight,
    run_worker_start,
    run_worker_status,
)
from hermes_cli.gpucloud_worker_task import (
    WorkerTask,
    WorkerTaskError,
    load_worker_task,
    resolve_worker_task_discovery,
)

TERMINAL_STAGES = {
    "training_failed",
    "conversion_failed",
    "completed",
}


def worker_goal_state_dir() -> Path:
    override = os.environ.get("GPUCLOUD_WORKER_GOAL_STATE_DIR", "").strip()
    path = Path(override).expanduser() if override else get_hermes_home() / "gpucloud" / "worker_goal_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def worker_goal_state_path(job_id: str) -> Path:
    return worker_goal_state_dir() / f"{_safe_job_id(job_id)}.json"


def find_worker_goal_task(task_file: Optional[Union[str, Path]] = None) -> Optional[Path]:
    return resolve_worker_task_discovery(task_file)


def has_worker_goal_task() -> bool:
    return find_worker_goal_task() is not None


def load_worker_goal_task(task_file: Optional[Union[str, Path]] = None) -> WorkerTask:
    path = find_worker_goal_task(task_file)
    if path is None:
        raise WorkerTaskError("worker task file not found")
    return load_worker_task(path)


def build_worker_goal_context_block(task: WorkerTask, *, goal: str = "") -> str:
    intent = _resolve_mode(task, goal=goal)
    summary = json.dumps(task.summary(), ensure_ascii=False, indent=2)
    return (
        "[GPUCLOUD worker goal context — local child-agent mode]\n"
        f"Worker task: {task.path}\n"
        f"Intent: {intent}\n"
        f"{summary}\n\n"
        "Mandatory workflow:\n"
        "1. Call gpucloud_worker_goal_run first. Do not call gpucloud_goal_prepare.\n"
        "2. Do not run SSH cluster checks; this child agent manages only this local machine.\n"
        "3. gpucloud_worker_goal_run may auto-start local Megatron training when preflight passes.\n"
        "4. Re-call gpucloud_worker_goal_run on later turns until stage=completed or a *_failed stage.\n"
        "5. GPUCLOUD manages local processes, logs, conversion, and vLLM health checks. "
        "Megatron-LM/PyTorch distributed/NCCL handle training communication.\n"
        "6. The coordinator/master is responsible for task distribution; this child agent "
        "must not schedule or SSH into other machines."
    )


def _read_state(job_id: str) -> Optional[Dict[str, Any]]:
    path = worker_goal_state_path(job_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _save_state(state: Dict[str, Any]) -> Dict[str, Any]:
    state["updated_at"] = time.time()
    path = worker_goal_state_path(str(state["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return state


def _resolve_mode(task: WorkerTask, *, goal: str = "", mode: Optional[str] = None) -> str:
    requested = mode or task.goal.get("mode")
    return infer_goal_intent(goal, str(requested) if requested else None)


def _new_state(task: WorkerTask, *, goal: str = "", mode: Optional[str] = None) -> Dict[str, Any]:
    now = time.time()
    intent = _resolve_mode(task, goal=goal, mode=mode)
    return {
        "workflow_id": f"worker-goal-{task.job_id}",
        "job_id": task.job_id,
        "task_file": str(task.path),
        "backend": "worker_local",
        "intent": intent,
        "stage": "environment_preparing",
        "status": "active",
        "next_action": "prepare local environment",
        "created_at": now,
        "updated_at": now,
        "environment": {},
        "data": {},
        "train": {},
        "conversion": {},
        "inference": {},
        "logs": {},
        "last_error": None,
    }


def _base_response(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": state.get("stage") not in {"training_failed", "conversion_failed"},
        "workflow_id": state.get("workflow_id"),
        "job_id": state.get("job_id"),
        "backend": "worker_local",
        "stage": state.get("stage"),
        "status": state.get("status"),
        "next_action": state.get("next_action"),
        "environment": state.get("environment") or {},
        "data": state.get("data") or {},
        "train": state.get("train") or {},
        "conversion": state.get("conversion") or {},
        "inference": state.get("inference") or {},
        "logs": state.get("logs") or {},
        "last_error": state.get("last_error"),
        "task_file": state.get("task_file"),
        "message": "worker goal advanced",
    }


def _task_values(task: WorkerTask) -> Dict[str, str]:
    return {
        "job_id": task.job_id,
        "workdir": str(task.runtime.get("workdir") or ""),
        "megatron_lm_dir": str(task.runtime.get("megatron_lm_dir") or ""),
        "data_path": str(task.training.get("data_path") or ""),
        "checkpoint_dir": str(task.training.get("checkpoint_dir") or ""),
        "log_dir": str(task.training.get("log_dir") or ""),
        "conversion_output_dir": str(task.conversion.get("output_dir") or ""),
        "model_path": str(task.inference.get("model_path") or task.conversion.get("output_dir") or ""),
        "python": str(task.runtime.get("python") or "python"),
    }


def _format_template(template: str, task: WorkerTask) -> str:
    values = _task_values(task)
    class Safe(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"
    return template.format_map(Safe(values)).strip()


def _model_path_loadable(path: Union[str, Path, None]) -> bool:
    if not path:
        return False
    root = _expand_path(path)
    if not root.is_dir():
        return False
    markers = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    }
    return any((root / marker).exists() for marker in markers)


def _conversion_paths(task: WorkerTask) -> Dict[str, str]:
    base = _expand_path(task.training.get("log_dir") or _expand_path(task.runtime["workdir"]) / "logs")
    log_path = base / f"{task.job_id}.conversion.log"
    exit_code_path = base / f"{task.job_id}.conversion.exitcode"
    return {"log_path": str(log_path), "exit_code_path": str(exit_code_path)}


def _resolve_conversion_command(task: WorkerTask) -> Dict[str, Any]:
    template = str(task.conversion.get("command_template") or "").strip()
    output_dir = str(task.conversion.get("output_dir") or task.inference.get("model_path") or "")
    if template:
        return {"ok": True, "command": _format_template(template, task), "method": "command_template"}

    if not bool(task.conversion.get("auto_discover", True)):
        return {"ok": False, "error": "conversion auto_discover is disabled"}

    convert_py = _expand_path(task.runtime.get("megatron_lm_dir")) / "tools" / "checkpoint" / "convert.py"
    if not convert_py.is_file():
        return {
            "ok": False,
            "error": f"conversion script not found: {convert_py}",
            "hint": "provide conversion.command_template or install Megatron-LM conversion tools",
        }

    python = str(task.runtime.get("python") or "python")
    try:
        help_proc = subprocess.run(
            [python, str(convert_py), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"unable to inspect conversion script: {exc}"}
    help_text = (help_proc.stdout or "") + (help_proc.stderr or "")
    if "--loader" not in help_text or "--saver" not in help_text:
        return {
            "ok": False,
            "error": "conversion script shape is not recognized",
            "hint": "provide conversion.command_template",
        }

    command = " ".join(
        [
            quote_shell_arg(python),
            quote_shell_arg(str(convert_py)),
            "--loader",
            "megatron",
            "--saver",
            "transformers",
            "--load-dir",
            quote_shell_arg(str(task.training["checkpoint_dir"])),
            "--save-dir",
            quote_shell_arg(output_dir),
        ]
    )
    return {"ok": True, "command": command, "method": "auto_discovered_convert_py"}


def _inference_paths(task: WorkerTask) -> Dict[str, str]:
    base = _expand_path(task.inference.get("log_dir") or task.training.get("log_dir") or _expand_path(task.runtime["workdir"]) / "logs")
    return {
        "log_path": str(base / f"{task.job_id}.vllm.log"),
        "exit_code_path": str(base / f"{task.job_id}.vllm.exitcode"),
    }


def _build_vllm_command(task: WorkerTask) -> Dict[str, Any]:
    inf = task.inference
    engine = str(inf.get("engine") or "vllm").strip().lower()
    if engine != "vllm":
        return {"ok": False, "error": f"inference.engine=vllm required, got {engine!r}"}
    template = str(inf.get("command_template") or "").strip()
    if template:
        return {"ok": True, "command": _format_template(template, task)}

    model_path = str(inf.get("model_path") or task.conversion.get("output_dir") or "")
    host = str(inf.get("host") or "0.0.0.0")
    port = int(inf.get("port") or 8000)
    tp = int(inf.get("tensor_parallel") or 1)
    parts = [
        "vllm",
        "serve",
        quote_shell_arg(model_path),
        "--host",
        quote_shell_arg(host),
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
    ]
    extra = inf.get("extra_args")
    if isinstance(extra, list):
        parts.extend(quote_shell_arg(x) for x in extra if str(x).strip())
    elif isinstance(extra, str) and extra.strip():
        parts.append(extra.strip())
    return {"ok": True, "command": " ".join(parts), "port": port, "host": host}


def _health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/health", timeout=5) as resp:
            return 200 <= int(resp.status) < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _tail_for_state(state: Dict[str, Any], section: str, *, lines: int = 50) -> str:
    data = state.get(section) or {}
    return _tail_text(_expand_path(data.get("log_path") or ""), lines)


def _prepare_environment(task: WorkerTask) -> Dict[str, Any]:
    paths = [
        _expand_path(task.runtime["workdir"]),
        _expand_path(task.training["checkpoint_dir"]),
        _expand_path(task.training["log_dir"]),
    ]
    created: list[str] = []
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    warnings = []
    if task.environment.get("auto_install"):
        warnings.append(
            "environment.auto_install requested; GPUCLOUD prepared paths/env and expects the image or provisioner to provide packages"
        )
    return {
        "ok": True,
        "created_paths": created,
        "env_keys": sorted(worker_env(task).keys()),
        "warnings": warnings,
    }


def _prepare_data(task: WorkerTask) -> Dict[str, Any]:
    data_path = _expand_path(task.training.get("data_path") or _expand_path(task.runtime["workdir"]) / "data")
    auto_data = bool(
        task.training_runner == "swift_megatron"
        or task.training.get("dataset_config")
        or (isinstance(task.training.get("megatron"), dict) and task.training["megatron"].get("auto_data"))
    )
    if auto_data:
        data_path.mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "data_path": str(data_path),
            "mode": "auto_data",
            "message": "data path prepared; runner/dataset config handles dataset materialization",
        }
    return {
        "ok": data_path.exists(),
        "data_path": str(data_path),
        "mode": "existing_path",
        "message": "data path exists" if data_path.exists() else "data path does not exist",
    }


def run_worker_goal_status(
    *,
    job_id: Optional[str] = None,
    task_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    try:
        task = load_worker_goal_task(task_file) if not job_id else None
    except WorkerTaskError as exc:
        return {"ok": False, "error": str(exc), "errors": list(exc.errors)}
    jid = job_id or (task.job_id if task else "")
    state = _read_state(jid)
    if not state:
        return {"ok": False, "error": f"worker goal workflow not found: {jid}"}
    return _base_response(state)


def run_worker_goal_run(
    *,
    task_file: Optional[Union[str, Path]] = None,
    goal: str = "",
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        task = load_worker_goal_task(task_file)
    except WorkerTaskError as exc:
        return {"ok": False, "error": str(exc), "errors": list(exc.errors)}

    state = _read_state(task.job_id) or _new_state(task, goal=goal, mode=mode)
    if state.get("stage") in TERMINAL_STAGES:
        return _base_response(state)

    intent = str(state.get("intent") or _resolve_mode(task, goal=goal, mode=mode))
    auto_execute = bool(task.goal.get("auto_execute", True))

    if state["stage"] == "environment_preparing":
        prepared = _prepare_environment(task)
        state["environment"] = prepared
        if not prepared.get("ok"):
            state.update(
                stage="training_failed",
                status="failed",
                next_action="fix local environment preparation errors",
                last_error=prepared.get("error") or "environment preparation failed",
            )
        else:
            state.update(stage="data_preparing", next_action="prepare local data path")
        _save_state(state)
        return _base_response(state)

    if state["stage"] == "data_preparing":
        prepared = _prepare_data(task)
        state["data"] = prepared
        if not prepared.get("ok"):
            state.update(
                stage="training_failed",
                status="failed",
                next_action="fix local data preparation errors",
                last_error=prepared.get("message") or "data preparation failed",
            )
        else:
            state.update(stage="preflight", next_action="run local preflight")
        _save_state(state)
        return _base_response(state)

    if state["stage"] == "preflight":
        preflight = run_worker_preflight(task_file=task.path)
        state["preflight"] = preflight
        if not preflight.get("ok"):
            state.update(stage="training_failed", status="failed", next_action="fix local preflight errors", last_error="preflight failed")
            _save_state(state)
            return _base_response(state)
        dry = run_worker_dry_run(task_file=task.path)
        state["train"] = dry
        state.update(stage="train_dry_run", next_action="start local Megatron training")
        if auto_execute:
            started = run_worker_start(task_file=task.path, confirm_execute=True, skip_preflight=True)
            state["train"] = {**dry, "start": started}
            if not started.get("ok"):
                state.update(stage="training_failed", status="failed", next_action="inspect training launch error", last_error=started.get("error"))
            else:
                state.update(stage="training_running", status="running", next_action="poll training status")
        _save_state(state)
        return _base_response(state)

    if state["stage"] == "train_dry_run":
        started = run_worker_start(task_file=task.path, confirm_execute=True, skip_preflight=True)
        state["train"] = {**(state.get("train") or {}), "start": started}
        if not started.get("ok"):
            state.update(stage="training_failed", status="failed", next_action="inspect training launch error", last_error=started.get("error"))
        else:
            state.update(stage="training_running", status="running", next_action="poll training status")
        _save_state(state)
        return _base_response(state)

    if state["stage"] == "training_running":
        status = run_worker_status(job_id=task.job_id)
        state["train"] = {**(state.get("train") or {}), "status": status}
        state["logs"] = {"train_tail": _tail_for_state(state, "train")}
        job = (status.get("job") if status.get("ok") else {}) or {}
        if status.get("running"):
            state.update(status="running", next_action="poll training status")
        elif job.get("status") == "completed" and (job.get("exit_code") in (0, None)):
            state.update(stage="training_completed", status="running", next_action="resolve conversion")
        else:
            state.update(stage="training_failed", status="failed", next_action="inspect training logs", last_error=job.get("last_error") or status.get("error") or "training failed")
        _save_state(state)
        return _base_response(state)

    if state["stage"] == "training_completed":
        if intent == "train":
            state.update(stage="completed", status="completed", next_action="done")
            _save_state(state)
            return _base_response(state)
        state.update(stage="conversion_resolving", next_action="resolve Megatron checkpoint conversion")
        _save_state(state)

    if state["stage"] == "conversion_resolving":
        model_path = task.inference.get("model_path") or task.conversion.get("output_dir")
        if _model_path_loadable(model_path):
            state["conversion"] = {"status": "skipped", "reason": "model_path already loadable", "model_path": str(model_path)}
            state.update(stage="conversion_completed", next_action="start local vLLM")
        else:
            resolved = _resolve_conversion_command(task)
            state["conversion"] = resolved
            if not resolved.get("ok"):
                state.update(stage="conversion_failed", status="failed", next_action="provide conversion.command_template", last_error=resolved.get("error"))
            else:
                paths = _conversion_paths(task)
                env = os.environ.copy()
                env.update(worker_env(task))
                proc = launch_wrapped_command(
                    command=resolved["command"],
                    workdir=task.runtime["workdir"],
                    log_path=paths["log_path"],
                    exit_code_path=paths["exit_code_path"],
                    env=env,
                )
                state["conversion"] = {**resolved, **paths, "pid": proc.pid}
                state.update(stage="conversion_running", status="running", next_action="poll conversion status")
        _save_state(state)
        return _base_response(state)

    if state["stage"] == "conversion_running":
        conversion = state.get("conversion") or {}
        exit_code = read_exit_code(conversion.get("exit_code_path"))
        state["logs"] = {**(state.get("logs") or {}), "conversion_tail": _tail_for_state(state, "conversion")}
        if exit_code is None and _pid_running(conversion.get("pid")):
            state.update(status="running", next_action="poll conversion status")
        elif exit_code == 0:
            state["conversion"] = {**conversion, "exit_code": 0}
            state.update(stage="conversion_completed", status="running", next_action="start local vLLM")
        else:
            state["conversion"] = {**conversion, "exit_code": exit_code}
            state.update(stage="conversion_failed", status="failed", next_action="inspect conversion logs", last_error=f"conversion exited {exit_code}")
        _save_state(state)
        return _base_response(state)

    if state["stage"] == "conversion_completed":
        state.update(stage="inference_starting", next_action="start local vLLM")
        _save_state(state)

    if state["stage"] == "inference_starting":
        command = _build_vllm_command(task)
        if not command.get("ok"):
            state["inference"] = command
            state.update(stage="conversion_failed", status="failed", next_action="fix inference config", last_error=command.get("error"))
            _save_state(state)
            return _base_response(state)
        paths = _inference_paths(task)
        env = os.environ.copy()
        env.update(worker_env(task))
        proc = launch_wrapped_command(
            command=command["command"],
            workdir=task.runtime["workdir"],
            log_path=paths["log_path"],
            exit_code_path=paths["exit_code_path"],
            env=env,
        )
        state["inference"] = {
            **command,
            **paths,
            "pid": proc.pid,
            "service_url": f"http://127.0.0.1:{int(task.inference.get('port') or 8000)}",
        }
        state.update(stage="inference_running", status="running", next_action="poll local vLLM health")
        _save_state(state)
        return _base_response(state)

    if state["stage"] == "inference_running":
        inf = state.get("inference") or {}
        port = int(task.inference.get("port") or inf.get("port") or 8000)
        exit_code = read_exit_code(inf.get("exit_code_path"))
        state["logs"] = {**(state.get("logs") or {}), "inference_tail": _tail_for_state(state, "inference")}
        if _health_ok(port):
            state["inference"] = {**inf, "healthy": True}
            state.update(stage="completed", status="completed", next_action="done")
        elif exit_code is not None:
            state["inference"] = {**inf, "exit_code": exit_code, "healthy": False}
            state.update(stage="conversion_failed", status="failed", next_action="inspect vLLM logs", last_error=f"vLLM exited {exit_code}")
        else:
            state["inference"] = {**inf, "healthy": False}
            state.update(status="running", next_action="poll local vLLM health")
        _save_state(state)
        return _base_response(state)

    return _base_response(state)


__all__ = [
    "build_worker_goal_context_block",
    "find_worker_goal_task",
    "has_worker_goal_task",
    "load_worker_goal_task",
    "run_worker_goal_run",
    "run_worker_goal_status",
    "worker_goal_state_dir",
    "worker_goal_state_path",
]
