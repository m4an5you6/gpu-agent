#!/usr/bin/env python3
"""GPUCLOUD SSH/GPU, training, checkpoint, and vLLM inference tools."""

from __future__ import annotations

import json
from typing import Any, Dict

from hermes_cli.gpucloud_config import GpucloudConfigError
from hermes_cli.gpucloud_checkpoints import (
    run_checkpoint_cleanup,
    run_checkpoint_latest,
    run_checkpoint_list,
    run_checkpoint_validate,
    run_train_resume,
)
from hermes_cli.gpucloud_probe import (
    run_cluster_check,
    run_gpu_info,
    run_ssh_exec,
)
from hermes_cli.gpucloud_inference import (
    run_infer_health,
    run_infer_start,
    run_infer_status,
    run_infer_stop,
)
from hermes_cli.gpucloud_goal import run_goal_prepare
from hermes_cli.gpucloud_train import (
    run_train_logs,
    run_train_start,
    run_train_status,
)
from hermes_cli.gpucloud_worker import (
    run_worker_dry_run,
    run_worker_logs,
    run_worker_preflight,
    run_worker_start,
    run_worker_status,
    run_worker_stop,
    run_worker_wait,
)
from hermes_cli.gpucloud_worker_goal import (
    run_worker_goal_run,
    run_worker_goal_status,
)
from tools.registry import registry, tool_error, tool_result


def _json_payload(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def gpucloud_cluster_check_handler(args: Dict[str, Any], **kwargs) -> str:
    try:
        data = run_cluster_check(
            config_file=args.get("config_file"),
            cluster_name=args.get("cluster"),
        )
    except Exception as exc:
        return tool_error(str(exc), success=False)
    if not data.get("ok") and data.get("error"):
        return tool_error(data["error"], success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_ssh_exec_handler(args: Dict[str, Any], **kwargs) -> str:
    command = (args.get("command") or "").strip()
    if not command:
        return tool_error("command is required", success=False)
    try:
        data = run_ssh_exec(
            command=command,
            config_file=args.get("config_file"),
            cluster_name=args.get("cluster"),
            node_index=int(args.get("node_index", 0) or 0),
            dry_run=args.get("dry_run"),
        )
    except Exception as exc:
        return tool_error(str(exc), success=False)
    if data.get("error") and not data.get("ok"):
        return tool_error(data["error"], success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_gpu_probe_handler(args: Dict[str, Any], **kwargs) -> str:
    target = (args.get("target") or "remote").strip().lower()
    if target not in {"local", "remote"}:
        return tool_error("target must be 'local' or 'remote'", success=False)
    try:
        data = run_gpu_info(
            config_file=args.get("config_file"),
            cluster_name=args.get("cluster"),
            node_index=args.get("node_index"),
            target=target,
        )
    except GpucloudConfigError as exc:
        return tool_error(str(exc), success=False)
    except Exception as exc:
        return tool_error(str(exc), success=False)
    if data.get("error"):
        return tool_error(data["error"], success=False, detail=data)
    return tool_result(success=True, data=data)


def check_gpucloud_tools_requirements() -> bool:
    """Tools register always; SSH optional at runtime."""
    return True


CLUSTER_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster": {
            "type": "string",
            "description": "Optional cluster name filter from gpucloud.yaml",
        },
        "config_file": {
            "type": "string",
            "description": "Optional path to gpucloud.yaml (else /goal context or auto-discover)",
        },
    },
    "required": [],
}

SSH_EXEC_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Read-only remote command (must match allowed_remote_prefixes)",
        },
        "cluster": {"type": "string", "description": "Cluster name"},
        "node_index": {
            "type": "integer",
            "description": "Node index within cluster (default 0)",
        },
        "config_file": {"type": "string", "description": "Optional gpucloud.yaml path"},
        "dry_run": {
            "type": "boolean",
            "description": "If true, only show SSH command without executing",
        },
    },
    "required": ["command"],
}

GPU_PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "enum": ["local", "remote"],
            "description": "Probe local machine or a cluster node",
        },
        "cluster": {"type": "string"},
        "node_index": {
            "type": "integer",
            "description": "Required when target=remote",
        },
        "config_file": {"type": "string"},
    },
    "required": [],
}


registry.register(
    name="gpucloud_cluster_check",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_cluster_check",
            "description": (
                "Read-only check of all GPU cluster nodes: SSH connectivity, "
                "workdir access, optional shared dirs, and nvidia-smi GPU info. "
                "Requires active /goal or config_file. Does not modify remote hosts."
            ),
            "parameters": CLUSTER_CHECK_SCHEMA,
        },
    },
    handler=gpucloud_cluster_check_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[CLUSTER]",
)

registry.register(
    name="gpucloud_ssh_exec",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_ssh_exec",
            "description": (
                "Run a read-only command on a cluster node over SSH with timeout "
                "and output truncation. Defaults to dry-run when security.dry_run_required "
                "is true. Requires /goal or config_file."
            ),
            "parameters": SSH_EXEC_SCHEMA,
        },
    },
    handler=gpucloud_ssh_exec_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[CLUSTER]",
)

def gpucloud_train_start_handler(args: Dict[str, Any], **kwargs) -> str:
    try:
        data = run_train_start(
            config_file=args.get("config_file"),
            cluster_name=args.get("cluster"),
            node_index=int(args.get("node_index", 0) or 0),
            dry_run=args.get("dry_run"),
            confirm_execute=bool(args.get("confirm_execute")),
        )
    except Exception as exc:
        return tool_error(str(exc), success=False)
    if not data.get("ok"):
        return tool_error(data.get("error", "train start failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_train_status_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_train_status(
        job_id=args.get("job_id"),
        limit=int(args.get("limit", 10) or 10),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "not found"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_train_logs_handler(args: Dict[str, Any], **kwargs) -> str:
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return tool_error("job_id is required", success=False)
    data = run_train_logs(job_id, lines=int(args.get("lines", 50) or 50))
    if not data.get("ok") and data.get("error"):
        return tool_error(data["error"], success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_infer_start_handler(args: Dict[str, Any], **kwargs) -> str:
    try:
        data = run_infer_start(
            config_file=args.get("config_file"),
            cluster_name=args.get("cluster"),
            node_index=int(args.get("node_index", 0) or 0),
            dry_run=args.get("dry_run"),
            confirm_execute=bool(args.get("confirm_execute")),
        )
    except Exception as exc:
        return tool_error(str(exc), success=False)
    if not data.get("ok"):
        return tool_error(data.get("error", "infer start failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_infer_status_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_infer_status(
        job_id=args.get("job_id"),
        limit=int(args.get("limit", 10) or 10),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "not found"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_infer_health_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_infer_health(
        job_id=args.get("job_id"),
        config_file=args.get("config_file"),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "infer health failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_infer_stop_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_infer_stop(
        job_id=args.get("job_id"),
        config_file=args.get("config_file"),
        dry_run=args.get("dry_run"),
        confirm_stop=bool(args.get("confirm_stop")),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "infer stop failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_goal_prepare_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_goal_prepare(
        goal=args.get("goal") or "",
        mode=args.get("mode"),
        config_file=args.get("config_file"),
        cluster_name=args.get("cluster"),
        node_index=int(args.get("node_index", 0) or 0),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "goal preparation failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def _task_file_arg(args: Dict[str, Any]) -> str:
    return str(args.get("task_file") or "").strip()


def gpucloud_worker_wait_handler(args: Dict[str, Any], **kwargs) -> str:
    task_file = _task_file_arg(args)
    if not task_file:
        return tool_error("task_file is required", success=False)
    data = run_worker_wait(
        task_file=task_file,
        timeout_sec=int(args.get("timeout_sec", 30) or 30),
        wait_for_master=bool(args.get("wait_for_master")),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "worker wait failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_worker_preflight_handler(args: Dict[str, Any], **kwargs) -> str:
    task_file = _task_file_arg(args)
    if not task_file:
        return tool_error("task_file is required", success=False)
    data = run_worker_preflight(
        task_file=task_file,
        check_network=bool(args.get("check_network", True)),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "worker preflight failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_worker_dry_run_handler(args: Dict[str, Any], **kwargs) -> str:
    task_file = _task_file_arg(args)
    if not task_file:
        return tool_error("task_file is required", success=False)
    data = run_worker_dry_run(task_file=task_file)
    if not data.get("ok"):
        return tool_error(data.get("error", "worker dry-run failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_worker_start_handler(args: Dict[str, Any], **kwargs) -> str:
    task_file = _task_file_arg(args)
    if not task_file:
        return tool_error("task_file is required", success=False)
    data = run_worker_start(
        task_file=task_file,
        confirm_execute=bool(args.get("confirm_execute")),
        skip_preflight=False,
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "worker start failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_worker_status_handler(args: Dict[str, Any], **kwargs) -> str:
    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return tool_error("job_id is required", success=False)
    data = run_worker_status(job_id=job_id)
    if not data.get("ok"):
        return tool_error(data.get("error", "worker status failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_worker_logs_handler(args: Dict[str, Any], **kwargs) -> str:
    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return tool_error("job_id is required", success=False)
    data = run_worker_logs(job_id=job_id, lines=int(args.get("lines", 50) or 50))
    if not data.get("ok"):
        return tool_error(data.get("error", "worker logs failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_worker_stop_handler(args: Dict[str, Any], **kwargs) -> str:
    job_id = str(args.get("job_id") or "").strip()
    if not job_id:
        return tool_error("job_id is required", success=False)
    data = run_worker_stop(job_id=job_id, confirm_stop=bool(args.get("confirm_stop")))
    if not data.get("ok"):
        return tool_error(data.get("error", "worker stop failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_worker_goal_run_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_worker_goal_run(
        task_file=args.get("task_file"),
        goal=args.get("goal") or "",
        mode=args.get("mode"),
    )
    if not data.get("ok"):
        return tool_error(data.get("last_error") or data.get("error", "worker goal failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_worker_goal_status_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_worker_goal_status(
        job_id=args.get("job_id"),
        task_file=args.get("task_file"),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "worker goal status failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_checkpoint_list_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_checkpoint_list(
        config_file=args.get("config_file"),
        cluster_name=args.get("cluster"),
        node_index=int(args.get("node_index", 0) or 0),
        limit=int(args.get("limit", 20) or 20),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "checkpoint list failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_checkpoint_latest_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_checkpoint_latest(
        config_file=args.get("config_file"),
        cluster_name=args.get("cluster"),
        node_index=int(args.get("node_index", 0) or 0),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "checkpoint latest failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_checkpoint_validate_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_checkpoint_validate(
        config_file=args.get("config_file"),
        cluster_name=args.get("cluster"),
        node_index=int(args.get("node_index", 0) or 0),
        checkpoint_path=args.get("checkpoint_path"),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "checkpoint validation failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_train_resume_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_train_resume(
        config_file=args.get("config_file"),
        cluster_name=args.get("cluster"),
        node_index=int(args.get("node_index", 0) or 0),
        checkpoint_path=args.get("checkpoint_path"),
        dry_run=args.get("dry_run"),
        confirm_execute=bool(args.get("confirm_execute")),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "train resume failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


def gpucloud_checkpoint_cleanup_handler(args: Dict[str, Any], **kwargs) -> str:
    data = run_checkpoint_cleanup(
        config_file=args.get("config_file"),
        cluster_name=args.get("cluster"),
        node_index=int(args.get("node_index", 0) or 0),
        keep=int(args.get("keep", 3) or 3),
        dry_run=args.get("dry_run"),
        confirm_delete=bool(args.get("confirm_delete")),
    )
    if not data.get("ok"):
        return tool_error(data.get("error", "checkpoint cleanup failed"), success=False, detail=data)
    return tool_result(success=True, data=data)


TRAIN_START_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster": {"type": "string", "description": "Cluster name from gpucloud.yaml"},
        "node_index": {"type": "integer", "description": "Node index (default 0)"},
        "config_file": {"type": "string"},
        "dry_run": {
            "type": "boolean",
            "description": "If true, only show launch plan (default from security.dry_run_required)",
        },
        "confirm_execute": {
            "type": "boolean",
            "description": "Must be true to actually SSH-launch training after dry-run review",
        },
    },
    "required": [],
}

TRAIN_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string", "description": "Job id; omit to list recent jobs"},
        "limit": {"type": "integer", "description": "Max jobs when listing (default 10)"},
    },
    "required": [],
}

TRAIN_LOGS_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string"},
        "lines": {"type": "integer", "description": "Tail lines (default 50, max 500)"},
    },
    "required": ["job_id"],
}

INFER_START_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster": {"type": "string", "description": "Cluster name from gpucloud.yaml"},
        "node_index": {"type": "integer", "description": "Node index (default 0)"},
        "config_file": {"type": "string"},
        "dry_run": {
            "type": "boolean",
            "description": "If true, only show vLLM launch plan",
        },
        "confirm_execute": {
            "type": "boolean",
            "description": "Must be true to actually SSH-launch vLLM",
        },
    },
    "required": [],
}

INFER_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string", "description": "Job id; omit to list recent services"},
        "limit": {"type": "integer", "description": "Max services when listing (default 10)"},
    },
    "required": [],
}

INFER_HEALTH_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string", "description": "Inference job id; omit for latest"},
        "config_file": {"type": "string"},
    },
    "required": [],
}

INFER_STOP_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string", "description": "Inference job id; omit for latest"},
        "config_file": {"type": "string"},
        "dry_run": {
            "type": "boolean",
            "description": "If true, only show stop command",
        },
        "confirm_stop": {
            "type": "boolean",
            "description": "Must be true to stop the remote vLLM process",
        },
    },
    "required": [],
}

GOAL_PREPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "User goal text; used to infer train vs inference when mode is omitted",
        },
        "mode": {
            "type": "string",
            "enum": ["train", "infer", "train_and_infer"],
            "description": "Optional explicit workflow mode",
        },
        "cluster": {"type": "string", "description": "Cluster name from gpucloud.yaml"},
        "node_index": {"type": "integer", "description": "Node index (default 0)"},
        "config_file": {"type": "string"},
    },
    "required": [],
}

CHECKPOINT_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster": {"type": "string", "description": "Cluster name from gpucloud.yaml"},
        "node_index": {"type": "integer", "description": "Node index (default 0)"},
        "config_file": {"type": "string"},
        "limit": {
            "type": "integer",
            "description": "Max checkpoints to list (list only, default 20)",
        },
    },
    "required": [],
}

CHECKPOINT_VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster": {"type": "string"},
        "node_index": {"type": "integer"},
        "config_file": {"type": "string"},
        "checkpoint_path": {
            "type": "string",
            "description": "Checkpoint directory; omit to validate latest",
        },
    },
    "required": [],
}

TRAIN_RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster": {"type": "string"},
        "node_index": {"type": "integer"},
        "config_file": {"type": "string"},
        "checkpoint_path": {
            "type": "string",
            "description": "Checkpoint directory; omit to resume from latest",
        },
        "dry_run": {
            "type": "boolean",
            "description": "If true, only show resume launch plan",
        },
        "confirm_execute": {
            "type": "boolean",
            "description": "Must be true to actually launch resume training",
        },
    },
    "required": [],
}

CHECKPOINT_CLEANUP_SCHEMA = {
    "type": "object",
    "properties": {
        "cluster": {"type": "string"},
        "node_index": {"type": "integer"},
        "config_file": {"type": "string"},
        "keep": {"type": "integer", "description": "Newest checkpoints to keep"},
        "dry_run": {
            "type": "boolean",
            "description": "If true, only show deletion plan",
        },
        "confirm_delete": {
            "type": "boolean",
            "description": "Must be true to delete remote checkpoint directories",
        },
    },
    "required": [],
}

WORKER_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "task_file": {
            "type": "string",
            "description": "Path to the per-node gpucloud-worker-task.yaml file",
        },
    },
    "required": ["task_file"],
}

WORKER_WAIT_SCHEMA = {
    "type": "object",
    "properties": {
        **WORKER_TASK_SCHEMA["properties"],
        "timeout_sec": {
            "type": "integer",
            "description": "Seconds to wait for the task file (default 30)",
        },
        "wait_for_master": {
            "type": "boolean",
            "description": "For non-rank0 workers, also wait for master_addr:master_port",
        },
    },
    "required": ["task_file"],
}

WORKER_PREFLIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        **WORKER_TASK_SCHEMA["properties"],
        "check_network": {
            "type": "boolean",
            "description": "Check rendezvous port reachability/bindability (default true)",
        },
    },
    "required": ["task_file"],
}

WORKER_START_SCHEMA = {
    "type": "object",
    "properties": {
        **WORKER_TASK_SCHEMA["properties"],
        "confirm_execute": {
            "type": "boolean",
            "description": "Must be true to start the local worker process",
        },
    },
    "required": ["task_file"],
}

WORKER_JOB_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string", "description": "Distributed worker job id"},
    },
    "required": ["job_id"],
}

WORKER_LOGS_SCHEMA = {
    "type": "object",
    "properties": {
        **WORKER_JOB_SCHEMA["properties"],
        "lines": {"type": "integer", "description": "Tail lines (default 50, max 500)"},
    },
    "required": ["job_id"],
}

WORKER_STOP_SCHEMA = {
    "type": "object",
    "properties": {
        **WORKER_JOB_SCHEMA["properties"],
        "confirm_stop": {
            "type": "boolean",
            "description": "Must be true to stop the local worker process",
        },
    },
    "required": ["job_id"],
}

WORKER_GOAL_RUN_SCHEMA = {
    "type": "object",
    "properties": {
        "task_file": {
            "type": "string",
            "description": "Optional path to gpucloud-worker-task.yaml; omitted uses worker discovery order",
        },
        "goal": {
            "type": "string",
            "description": "Original /goal text for intent inference",
        },
        "mode": {
            "type": "string",
            "enum": ["train", "infer", "train_and_infer"],
            "description": "Optional explicit worker goal mode",
        },
    },
    "required": [],
}

WORKER_GOAL_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string", "description": "Worker goal job id"},
        "task_file": {
            "type": "string",
            "description": "Optional task file used to infer job_id when job_id is omitted",
        },
    },
    "required": [],
}


registry.register(
    name="gpucloud_train_start",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_train_start",
            "description": (
                "Plan or start single-node Megatron-LM training from gpucloud.yaml. "
                "Default is dry-run showing launch_command, log_path, checkpoint_path. "
                "Set confirm_execute=true to launch via SSH nohup. Requires /goal or config_file."
            ),
            "parameters": TRAIN_START_SCHEMA,
        },
    },
    handler=gpucloud_train_start_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_train_status",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_train_status",
            "description": "Get one training job by job_id or list recent persisted jobs.",
            "parameters": TRAIN_STATUS_SCHEMA,
        },
    },
    handler=gpucloud_train_status_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_train_logs",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_train_logs",
            "description": (
                "Tail remote training log file for a job (returns tail only, not full log)."
            ),
            "parameters": TRAIN_LOGS_SCHEMA,
        },
    },
    handler=gpucloud_train_logs_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_infer_start",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_infer_start",
            "description": (
                "Plan or start a vLLM inference service from gpucloud.yaml. "
                "Default is dry-run showing launch_command, service_url, and log_path. "
                "Set confirm_execute=true to launch via SSH nohup."
            ),
            "parameters": INFER_START_SCHEMA,
        },
    },
    handler=gpucloud_infer_start_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[INFER]",
)

registry.register(
    name="gpucloud_infer_status",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_infer_status",
            "description": "Get one vLLM service job or list recent persisted services.",
            "parameters": INFER_STATUS_SCHEMA,
        },
    },
    handler=gpucloud_infer_status_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[INFER]",
)

registry.register(
    name="gpucloud_infer_health",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_infer_health",
            "description": "Check a remote vLLM service /health endpoint over SSH.",
            "parameters": INFER_HEALTH_SCHEMA,
        },
    },
    handler=gpucloud_infer_health_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[INFER]",
)

registry.register(
    name="gpucloud_infer_stop",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_infer_stop",
            "description": (
                "Dry-run or stop a persisted vLLM service by remote pid. "
                "Set confirm_stop=true to execute the remote stop command."
            ),
            "parameters": INFER_STOP_SCHEMA,
        },
    },
    handler=gpucloud_infer_stop_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[INFER]",
)

registry.register(
    name="gpucloud_goal_prepare",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_goal_prepare",
            "description": (
                "Phase-9 /goal preparation: run GPUCLOUD cluster check first, then "
                "produce dry-run training and/or vLLM inference plans. Never starts "
                "remote work and stops before dry-run if the cluster probe fails."
            ),
            "parameters": GOAL_PREPARE_SCHEMA,
        },
    },
    handler=gpucloud_goal_prepare_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[GOAL]",
)

registry.register(
    name="gpucloud_checkpoint_list",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_checkpoint_list",
            "description": (
                "List checkpoint directories under training.checkpoint_dir on a cluster node. "
                "Requires /goal or config_file."
            ),
            "parameters": CHECKPOINT_NODE_SCHEMA,
        },
    },
    handler=gpucloud_checkpoint_list_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[CHECKPOINT]",
)

registry.register(
    name="gpucloud_checkpoint_latest",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_checkpoint_latest",
            "description": "Return the newest checkpoint directory by remote mtime.",
            "parameters": CHECKPOINT_NODE_SCHEMA,
        },
    },
    handler=gpucloud_checkpoint_latest_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[CHECKPOINT]",
)

registry.register(
    name="gpucloud_checkpoint_validate",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_checkpoint_validate",
            "description": (
                "Validate a checkpoint directory by checking for common model/training "
                "state marker files. Omit checkpoint_path to validate latest."
            ),
            "parameters": CHECKPOINT_VALIDATE_SCHEMA,
        },
    },
    handler=gpucloud_checkpoint_validate_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[CHECKPOINT]",
)

registry.register(
    name="gpucloud_train_resume",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_train_resume",
            "description": (
                "Validate a checkpoint and dry-run or launch a Megatron-LM resume command. "
                "Default is dry-run; set confirm_execute=true to run remotely."
            ),
            "parameters": TRAIN_RESUME_SCHEMA,
        },
    },
    handler=gpucloud_train_resume_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[CHECKPOINT]",
)

registry.register(
    name="gpucloud_checkpoint_cleanup",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_checkpoint_cleanup",
            "description": (
                "Plan or delete old checkpoint directories. Defaults to dry-run and "
                "requires confirm_delete=true for remote deletion."
            ),
            "parameters": CHECKPOINT_CLEANUP_SCHEMA,
        },
    },
    handler=gpucloud_checkpoint_cleanup_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[CHECKPOINT]",
)

registry.register(
    name="gpucloud_worker_wait",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_worker_wait",
            "description": (
                "Wait for a per-node GPUCLOUD worker task file, validate rank settings, "
                "and optionally wait for the Megatron rendezvous address. Does not start training."
            ),
            "parameters": WORKER_WAIT_SCHEMA,
        },
    },
    handler=gpucloud_worker_wait_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_worker_preflight",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_worker_preflight",
            "description": (
                "Run local distributed Megatron worker preflight checks: GPU, CUDA/PyTorch/NCCL, "
                "paths, Megatron entrypoint, and rendezvous network."
            ),
            "parameters": WORKER_PREFLIGHT_SCHEMA,
        },
    },
    handler=gpucloud_worker_preflight_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_worker_dry_run",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_worker_dry_run",
            "description": (
                "Render the local torchrun/Megatron command for this worker rank from "
                "gpucloud-worker-task.yaml. Never starts a process."
            ),
            "parameters": WORKER_TASK_SCHEMA,
        },
    },
    handler=gpucloud_worker_dry_run_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_worker_start",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_worker_start",
            "description": (
                "Start this machine's local Megatron worker rank from a distributed task file. "
                "Requires confirm_execute=true; GPUCLOUD starts/monitors only the local process."
            ),
            "parameters": WORKER_START_SCHEMA,
        },
    },
    handler=gpucloud_worker_start_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_worker_status",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_worker_status",
            "description": "Get local status for a distributed Megatron worker job.",
            "parameters": WORKER_JOB_SCHEMA,
        },
    },
    handler=gpucloud_worker_status_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_worker_logs",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_worker_logs",
            "description": "Tail the local log file for a distributed Megatron worker job.",
            "parameters": WORKER_LOGS_SCHEMA,
        },
    },
    handler=gpucloud_worker_logs_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_worker_stop",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_worker_stop",
            "description": (
                "Stop a local distributed Megatron worker process. Requires confirm_stop=true."
            ),
            "parameters": WORKER_STOP_SCHEMA,
        },
    },
    handler=gpucloud_worker_stop_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[TRAIN]",
)

registry.register(
    name="gpucloud_worker_goal_run",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_worker_goal_run",
            "description": (
                "Advance the local child-agent /goal workflow from gpucloud-worker-task.yaml: "
                "preflight, local Megatron training, checkpoint conversion, local vLLM start, "
                "and health polling. Does not use SSH or schedule other machines."
            ),
            "parameters": WORKER_GOAL_RUN_SCHEMA,
        },
    },
    handler=gpucloud_worker_goal_run_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[GOAL]",
)

registry.register(
    name="gpucloud_worker_goal_status",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_worker_goal_status",
            "description": "Return persisted status for the local child-agent GPUCLOUD /goal workflow.",
            "parameters": WORKER_GOAL_STATUS_SCHEMA,
        },
    },
    handler=gpucloud_worker_goal_status_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[GOAL]",
)

registry.register(
    name="gpucloud_gpu_probe",
    toolset="gpucloud",
    schema={
        "type": "function",
        "function": {
            "name": "gpucloud_gpu_probe",
            "description": (
                "Probe GPU info via nvidia-smi (local or remote node). "
                "Skips gracefully when nvidia-smi is unavailable."
            ),
            "parameters": GPU_PROBE_SCHEMA,
        },
    },
    handler=gpucloud_gpu_probe_handler,
    check_fn=check_gpucloud_tools_requirements,
    emoji="[GPU]",
)
