"""Megatron-LM distributed command rendering for GPUCLOUD workers."""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
from typing import Any, Dict

from hermes_cli.gpucloud_worker_task import WorkerTask


SENSITIVE_ENV_TOKENS = ("SECRET", "TOKEN", "KEY", "PASSWORD", "PASS")


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def quote_shell_arg(value: Any) -> str:
    text = str(value)
    if text == "~":
        return "$HOME"
    if text.startswith("~/"):
        return "$HOME/" + shlex.quote(text[2:])
    return shlex.quote(text)


def is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return any(token in upper for token in SENSITIVE_ENV_TOKENS)


def redact_env(env: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in env.items():
        text = str(value)
        out[str(key)] = "***" if is_sensitive_env_name(str(key)) else text
    return out


def redact_text(text: str, env: Dict[str, Any]) -> str:
    out = text
    for key, value in env.items():
        if is_sensitive_env_name(str(key)):
            raw = str(value)
            if raw:
                out = out.replace(raw, "***")
    return out


def worker_env(task: WorkerTask) -> Dict[str, str]:
    raw = task.runtime.get("env")
    env = raw if isinstance(raw, dict) else {}
    out = {str(k): str(v) for k, v in env.items()}
    out.setdefault("MASTER_ADDR", task.master_addr)
    out.setdefault("MASTER_PORT", str(task.master_port))
    out.setdefault("WORLD_SIZE", str(task.nnodes * task.nproc_per_node))
    out.setdefault("NODE_RANK", str(task.node_rank))
    out.setdefault("NPROC_PER_NODE", str(task.nproc_per_node))
    return out


def _entrypoint_path(task: WorkerTask) -> str:
    entrypoint = str(task.training.get("entrypoint") or "pretrain_gpt.py")
    if entrypoint.startswith("/"):
        return entrypoint
    root = str(task.runtime.get("megatron_lm_dir") or "./Megatron-LM")
    return str(PurePosixPath(root) / entrypoint)


def _extra_args(task: WorkerTask) -> str:
    extra = task.training.get("extra_args")
    if isinstance(extra, list):
        return " ".join(quote_shell_arg(item) for item in extra if str(item).strip())
    if isinstance(extra, str):
        return extra.strip()
    return ""


def build_megatron_worker_command(task: WorkerTask) -> str:
    """Return the local torchrun command for this worker's rank."""
    training = task.training
    env = worker_env(task)
    values = {
        "job_id": task.job_id,
        "nnodes": str(task.nnodes),
        "nproc_per_node": str(task.nproc_per_node),
        "node_rank": str(task.node_rank),
        "master_addr": task.master_addr,
        "master_port": str(task.master_port),
        "megatron_lm_dir": str(task.runtime.get("megatron_lm_dir") or "./Megatron-LM"),
        "entrypoint": str(training.get("entrypoint") or "pretrain_gpt.py"),
        "entrypoint_path": _entrypoint_path(task),
        "data_path": str(training.get("data_path") or ""),
        "checkpoint_dir": str(training.get("checkpoint_dir") or ""),
        "log_dir": str(training.get("log_dir") or ""),
        "extra_args": _extra_args(task),
        "python": str(task.runtime.get("python") or "python"),
    }
    template = str(training.get("command_template") or "").strip()
    if template:
        return template.format_map(_SafeFormatDict(values)).strip()

    parts = [
        "torchrun",
        f"--nnodes={task.nnodes}",
        f"--nproc-per-node={task.nproc_per_node}",
        f"--node-rank={task.node_rank}",
        f"--master-addr={quote_shell_arg(task.master_addr)}",
        f"--master-port={task.master_port}",
        quote_shell_arg(values["entrypoint_path"]),
        "--data-path",
        quote_shell_arg(values["data_path"]),
        "--save",
        quote_shell_arg(values["checkpoint_dir"]),
        "--load",
        quote_shell_arg(values["checkpoint_dir"]),
    ]
    extra_args = values["extra_args"]
    if extra_args:
        parts.append(extra_args)
    return " ".join(parts)


def build_worker_plan(task: WorkerTask) -> Dict[str, Any]:
    env = worker_env(task)
    command = build_megatron_worker_command(task)
    log_path = str(PurePosixPath(str(task.training["log_dir"])) / f"{task.job_id}.rank{task.node_rank}.log")
    return {
        "ok": True,
        "task_file": str(task.path),
        "job_id": task.job_id,
        "framework": task.framework,
        "role": task.role,
        "node_rank": task.node_rank,
        "nnodes": task.nnodes,
        "nproc_per_node": task.nproc_per_node,
        "master_addr": task.master_addr,
        "master_port": task.master_port,
        "workdir": str(task.runtime["workdir"]),
        "log_path": log_path,
        "checkpoint_dir": str(task.training["checkpoint_dir"]),
        "launch_command": command,
        "launch_command_redacted": redact_text(command, env),
        "env": redact_env(env),
        "communication": (
            "GPUCLOUD starts and monitors this local worker rank. "
            "Megatron-LM/PyTorch distributed/NCCL perform gradient communication."
        ),
    }


__all__ = [
    "build_megatron_worker_command",
    "build_worker_plan",
    "quote_shell_arg",
    "redact_env",
    "redact_text",
    "worker_env",
]
