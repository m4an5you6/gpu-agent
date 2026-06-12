"""cluster_* tools exposing control-plane actions to model agents."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from plugins.cluster.client import ClusterClient
from plugins.cluster.config import load_cluster_config
from plugins.cluster.cluster_logging import ClusterLogger
from plugins.cluster.events import ClusterEventBridge
from plugins.cluster.store import open_store

# Module-level runtime wired by CLI serve/worker or tests
_runtime: Dict[str, Any] = {}


def _get_client() -> ClusterClient:
    cfg = load_cluster_config()
    return ClusterClient(cfg)


def _log_tool(tool_name: str, args: Dict[str, Any], decision: str, summary: str) -> None:
    logger: Optional[ClusterLogger] = _runtime.get("logger")
    if logger:
        logger.log_tool_action(
            tool_name=tool_name,
            tool_args=args,
            decision=decision,
            result_summary=summary,
        )


def check_cluster_available() -> bool:
    cfg = load_cluster_config()
    if not cfg.enabled:
        return False
    if os.environ.get("GPUCLOUD_CLUSTER_FORCE", "").strip() in ("1", "true", "yes"):
        return True
    # Tools available when plugin enabled and master URL reachable OR local master running
    if _runtime.get("controller"):
        return True
    try:
        client = ClusterClient(cfg, timeout=2.0)
        client.health()
        return True
    except Exception:
        return bool(cfg.master_url)


def _ok(data: Any) -> str:
    return json.dumps({"success": True, "data": data})


def _err(msg: str, **extra: Any) -> str:
    return json.dumps({"success": False, "error": msg, **extra})


def handle_cluster_status(args: Dict[str, Any], **kwargs: Any) -> str:
    try:
        client = _get_client()
        data = client.status()
        _log_tool("cluster_status", args, "allow", "ok")
        return _ok(data)
    except Exception as exc:
        ctrl = _runtime.get("controller")
        if ctrl:
            data = ctrl.status()
            _log_tool("cluster_status", args, "allow", "local")
            return _ok(data)
        _log_tool("cluster_status", args, "reject", str(exc))
        return _err(str(exc))


def handle_cluster_validate_config(args: Dict[str, Any], **kwargs: Any) -> str:
    spec = args.get("spec") or args
    try:
        client = _get_client()
        data = client.validate_config(spec if isinstance(spec, dict) else {})
        _log_tool("cluster_validate_config", args, "allow", "ok")
        return json.dumps(data)
    except Exception as exc:
        ctrl = _runtime.get("controller")
        if ctrl:
            result = ctrl.validate_config(spec if isinstance(spec, dict) else {})
            return json.dumps(result.to_dict())
        return _err(str(exc))


def handle_cluster_submit_job(args: Dict[str, Any], **kwargs: Any) -> str:
    spec = args.get("spec") or args
    if not isinstance(spec, dict):
        return _err("spec must be an object")
    try:
        client = _get_client()
        data = client.submit_job(spec)
        _log_tool("cluster_submit_job", args, "allow", data.get("job", {}).get("job_id", "ok"))
        return json.dumps(data)
    except Exception as exc:
        ctrl = _runtime.get("controller")
        if ctrl:
            data = ctrl.submit_job(spec)
            return json.dumps(data)
        _log_tool("cluster_submit_job", args, "reject", str(exc))
        return _err(str(exc))


def handle_cluster_job_status(args: Dict[str, Any], **kwargs: Any) -> str:
    job_id = str(args.get("job_id") or "")
    if not job_id:
        return _err("job_id required")
    try:
        client = _get_client()
        data = client.job_status(job_id)
        _log_tool("cluster_job_status", args, "allow", job_id)
        return json.dumps(data)
    except Exception as exc:
        ctrl = _runtime.get("controller")
        if ctrl:
            return json.dumps(ctrl.job_status(job_id))
        return _err(str(exc))


def handle_cluster_logs(args: Dict[str, Any], **kwargs: Any) -> str:
    params = {
        k: args.get(k)
        for k in ("job_id", "node_id", "limit")
        if args.get(k)
    }
    try:
        client = _get_client()
        data = client.logs(**params)
        _log_tool("cluster_logs", args, "allow", "ok")
        return json.dumps(data)
    except Exception as exc:
        store = _runtime.get("store")
        if store:
            logs = store.query_logs(
                job_id=str(args.get("job_id") or ""),
                node_id=str(args.get("node_id") or ""),
                limit=int(args.get("limit") or 50),
            )
            return _ok(logs)
        return _err(str(exc))


def handle_cluster_stop_job(args: Dict[str, Any], **kwargs: Any) -> str:
    job_id = str(args.get("job_id") or "")
    if not job_id:
        return _err("job_id required")
    try:
        client = _get_client()
        data = client.stop_job(job_id)
        _log_tool("cluster_stop_job", args, "allow", job_id)
        return json.dumps(data)
    except Exception as exc:
        ctrl = _runtime.get("controller")
        if ctrl:
            return json.dumps(ctrl.stop_job(job_id))
        return _err(str(exc))


def handle_cluster_node_action(args: Dict[str, Any], **kwargs: Any) -> str:
    node_id = str(args.get("node_id") or "")
    action = str(args.get("action") or "")
    if not node_id or not action:
        return _err("node_id and action required")
    try:
        client = _get_client()
        data = client.node_action(node_id, action)
        _log_tool("cluster_node_action", args, "allow", action)
        return json.dumps(data)
    except Exception as exc:
        ctrl = _runtime.get("controller")
        if ctrl:
            return json.dumps(ctrl.node_action(node_id, action))
        return _err(str(exc))


def set_runtime(**kwargs: Any) -> None:
    _runtime.update(kwargs)


CLUSTER_STATUS_SCHEMA = {
    "name": "cluster_status",
    "description": "View cluster master status, registered nodes, jobs, and stale nodes.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

CLUSTER_VALIDATE_CONFIG_SCHEMA = {
    "name": "cluster_validate_config",
    "description": "Programmatically validate a distributed training job spec before submission.",
    "parameters": {
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "description": (
                    "Job spec with script, nnodes, nproc_per_node, script_args, env, working_dir. "
                    "Logical fields (env_name, project, release, dataset, output_run_id, "
                    "min_scratch_gb) may appear at top level or in extra for heterogeneous workers."
                ),
            },
        },
        "required": ["spec"],
    },
}

CLUSTER_SUBMIT_JOB_SCHEMA = {
    "name": "cluster_submit_job",
    "description": "Submit a distributed training job to the cluster master; returns job id and rank plan.",
    "parameters": {
        "type": "object",
        "properties": {
            "spec": {
                "type": "object",
                "description": (
                    "Training job spec (script, nnodes, nproc_per_node, framework, env, working_dir). "
                    "Use env_name/project/release/dataset/output_run_id for per-node path resolution."
                ),
            },
        },
        "required": ["spec"],
    },
}

CLUSTER_JOB_STATUS_SCHEMA = {
    "name": "cluster_job_status",
    "description": "Query training job status, assignments, and recent log summaries.",
    "parameters": {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "Job identifier returned by cluster_submit_job."},
        },
        "required": ["job_id"],
    },
}

CLUSTER_LOGS_SCHEMA = {
    "name": "cluster_logs",
    "description": "Query structured cluster logs by job_id, node_id, or time window (paginated summaries).",
    "parameters": {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "node_id": {"type": "string"},
            "limit": {"type": "integer", "description": "Max records (default 50)."},
        },
    },
}

CLUSTER_STOP_JOB_SCHEMA = {
    "name": "cluster_stop_job",
    "description": "Stop or cancel a running cluster training job.",
    "parameters": {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
        },
        "required": ["job_id"],
    },
}

CLUSTER_NODE_ACTION_SCHEMA = {
    "name": "cluster_node_action",
    "description": "Quarantine, restore, or revalidate a cluster worker node.",
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["quarantine", "restore", "revalidate"],
            },
        },
        "required": ["node_id", "action"],
    },
}
