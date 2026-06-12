"""Cluster control-plane plugin — temporary master, worker agents, cluster_* tools."""

from __future__ import annotations

import logging

from plugins.cluster.cli import cluster_command, register_cli
from plugins.cluster.tools import (
    CLUSTER_JOB_STATUS_SCHEMA,
    CLUSTER_LOGS_SCHEMA,
    CLUSTER_NODE_ACTION_SCHEMA,
    CLUSTER_STATUS_SCHEMA,
    CLUSTER_STOP_JOB_SCHEMA,
    CLUSTER_SUBMIT_JOB_SCHEMA,
    CLUSTER_VALIDATE_CONFIG_SCHEMA,
    check_cluster_available,
    handle_cluster_job_status,
    handle_cluster_logs,
    handle_cluster_node_action,
    handle_cluster_status,
    handle_cluster_stop_job,
    handle_cluster_submit_job,
    handle_cluster_validate_config,
)

logger = logging.getLogger(__name__)

_TOOLS = (
    ("cluster_status", CLUSTER_STATUS_SCHEMA, handle_cluster_status, "🖧"),
    ("cluster_validate_config", CLUSTER_VALIDATE_CONFIG_SCHEMA, handle_cluster_validate_config, "✅"),
    ("cluster_submit_job", CLUSTER_SUBMIT_JOB_SCHEMA, handle_cluster_submit_job, "🚀"),
    ("cluster_job_status", CLUSTER_JOB_STATUS_SCHEMA, handle_cluster_job_status, "📊"),
    ("cluster_logs", CLUSTER_LOGS_SCHEMA, handle_cluster_logs, "📜"),
    ("cluster_stop_job", CLUSTER_STOP_JOB_SCHEMA, handle_cluster_stop_job, "🛑"),
    ("cluster_node_action", CLUSTER_NODE_ACTION_SCHEMA, handle_cluster_node_action, "🔧"),
)


def register(ctx) -> None:
    """Register cluster tools and CLI. Enable via plugins.enabled: [cluster]."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="cluster",
            schema=schema,
            handler=handler,
            check_fn=check_cluster_available,
            emoji=emoji,
        )

    ctx.register_cli_command(
        name="cluster",
        help="Temporary master control plane for multi-node training",
        setup_fn=register_cli,
        handler_fn=cluster_command,
    )

    logger.info("cluster plugin registered (%d tools)", len(_TOOLS))
