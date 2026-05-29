#!/usr/bin/env python3
"""GPUCLOUD phase 5: read-only SSH, GPU, and cluster probe tools."""

from __future__ import annotations

import json
from typing import Any, Dict

from hermes_cli.gpucloud_config import GpucloudConfigError
from hermes_cli.gpucloud_probe import (
    cluster_check_json,
    run_cluster_check,
    run_gpu_info,
    run_ssh_exec,
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
