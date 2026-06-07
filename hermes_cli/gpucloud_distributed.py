"""Megatron-LM distributed command rendering for GPUCLOUD workers."""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
from typing import Any, Dict

import yaml

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
    environment = task.environment
    if environment.get("hf_endpoint"):
        out.setdefault("HF_ENDPOINT", str(environment["hf_endpoint"]))
    if environment.get("pip_index_url"):
        out.setdefault("PIP_INDEX_URL", str(environment["pip_index_url"]))
    if environment.get("pip_trusted_host"):
        out.setdefault("PIP_TRUSTED_HOST", str(environment["pip_trusted_host"]))
    if environment.get("swift_te_extra_index_url"):
        out.setdefault("PIP_EXTRA_INDEX_URL", str(environment["swift_te_extra_index_url"]))
    out.setdefault("MASTER_ADDR", task.master_addr)
    out.setdefault("MASTER_PORT", str(task.master_port))
    out.setdefault("WORLD_SIZE", str(task.nnodes * task.nproc_per_node))
    out.setdefault("NODE_RANK", str(task.node_rank))
    out.setdefault("NPROC_PER_NODE", str(task.nproc_per_node))
    out.setdefault("NNODES", str(task.nnodes))
    out.setdefault("MEGATRON_LM_PATH", str(task.runtime.get("megatron_lm_dir") or "./Megatron-LM"))
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


def build_megatron_lm_worker_command(task: WorkerTask) -> str:
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


def _swift_mode(task: WorkerTask) -> str:
    swift = task.training.get("swift") if isinstance(task.training.get("swift"), dict) else {}
    train_type = str(swift.get("train_type") or task.training.get("training_type") or "sft").strip().lower()
    return "pt" if train_type in {"pt", "pretrain", "pretraining", "cpt"} else "sft"


def _swift_config_path(task: WorkerTask) -> str:
    return str(PurePosixPath(str(task.training["log_dir"])) / f"{task.job_id}.swift_megatron.yaml")


def _cli_key(key: str) -> str:
    return "--" + str(key).replace("-", "_")


def _append_cli_value(parts: list[str], key: str, value: Any) -> None:
    if value in (None, ""):
        return
    if isinstance(value, bool):
        value = "true" if value else "false"
    if isinstance(value, (list, tuple)):
        if not value:
            return
        parts.append(_cli_key(key))
        parts.extend(quote_shell_arg(item) for item in value)
        return
    if isinstance(value, dict):
        import json

        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    parts.extend([_cli_key(key), quote_shell_arg(value)])


def build_swift_megatron_config(task: WorkerTask) -> Dict[str, Any]:
    training = task.training
    megatron = training.get("megatron") if isinstance(training.get("megatron"), dict) else {}
    swift = training.get("swift") if isinstance(training.get("swift"), dict) else {}
    config: Dict[str, Any] = {}

    for key, value in swift.items():
        if key in {"extra_args", "validation_warnings"}:
            continue
        config[key] = value

    if training.get("batch_size") is not None:
        config.setdefault("micro_batch_size", training.get("batch_size"))
        config.setdefault("global_batch_size", training.get("batch_size"))
    if training.get("learning_rate") is not None:
        config.setdefault("lr", training.get("learning_rate"))
    if training.get("max_steps") is not None:
        config.setdefault("train_iters", training.get("max_steps"))
    if megatron.get("train_iters") is not None:
        config.setdefault("train_iters", megatron.get("train_iters"))
    if megatron.get("lr") is not None:
        config.setdefault("lr", megatron.get("lr"))

    config.setdefault("dataset", swift.get("dataset") or training.get("dataset_config") or megatron.get("data_name"))
    model = swift.get("model") or training.get("model")
    if model:
        config.setdefault("model", model)
    if swift.get("load") or training.get("load"):
        config.setdefault("load", swift.get("load") or training.get("load"))
    config.setdefault("save", training.get("checkpoint_dir"))
    config.setdefault("tensorboard_dir", str(PurePosixPath(str(training["log_dir"])) / "tensorboard"))

    if _swift_mode(task) == "sft":
        config.setdefault("finetune", True)
    if swift.get("max_length") is not None:
        config.setdefault("max_length", swift.get("max_length"))
    return {key: value for key, value in config.items() if value not in (None, "")}


def build_swift_megatron_worker_command(task: WorkerTask) -> str:
    config = build_swift_megatron_config(task)
    mode = _swift_mode(task)
    preferred = [
        "load",
        "model",
        "dataset",
        "train_type",
        "target_modules",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "max_length",
        "micro_batch_size",
        "global_batch_size",
        "lr",
        "train_iters",
        "finetune",
        "save",
        "tensorboard_dir",
    ]
    parts = ["megatron", mode]
    seen = set()
    for key in preferred:
        if key in config:
            _append_cli_value(parts, key, config[key])
            seen.add(key)
    for key in sorted(k for k in config if k not in seen):
        _append_cli_value(parts, key, config[key])
    return " ".join(parts)


def build_worker_command(task: WorkerTask) -> str:
    if task.training_runner == "swift_megatron":
        return build_swift_megatron_worker_command(task)
    return build_megatron_lm_worker_command(task)


def build_megatron_worker_command(task: WorkerTask) -> str:
    return build_megatron_lm_worker_command(task)


def write_worker_runtime_artifacts(task: WorkerTask, plan: Dict[str, Any]) -> Dict[str, str]:
    if task.training_runner != "swift_megatron":
        return {}
    from pathlib import Path

    target = Path(str(plan["swift_config_path"])).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(plan["swift_config"], sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"swift_config_path": str(target)}


def build_worker_plan(task: WorkerTask) -> Dict[str, Any]:
    env = worker_env(task)
    command = build_worker_command(task)
    log_path = str(PurePosixPath(str(task.training["log_dir"])) / f"{task.job_id}.rank{task.node_rank}.log")
    out = {
        "ok": True,
        "task_file": str(task.path),
        "job_id": task.job_id,
        "framework": task.framework,
        "training_runner": task.training_runner,
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
    if task.training_runner == "swift_megatron":
        out["swift_config_path"] = _swift_config_path(task)
        out["swift_config"] = build_swift_megatron_config(task)
        out["communication"] = (
            "GPUCLOUD starts and monitors this local Megatron-SWIFT worker rank. "
            "Megatron-SWIFT/PyTorch distributed/NCCL perform gradient communication."
        )
    return out


__all__ = [
    "build_megatron_lm_worker_command",
    "build_megatron_worker_command",
    "build_swift_megatron_config",
    "build_swift_megatron_worker_command",
    "build_worker_command",
    "build_worker_plan",
    "quote_shell_arg",
    "redact_env",
    "redact_text",
    "write_worker_runtime_artifacts",
    "worker_env",
]
