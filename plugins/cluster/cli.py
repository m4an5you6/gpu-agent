"""CLI entry points: gpucloud cluster serve | worker | status."""

from __future__ import annotations

import argparse
import json
import sys

from plugins.cluster.client import ClusterClient
from plugins.cluster.config import load_cluster_config, resolve_role
from plugins.cluster.controller import ClusterController
from plugins.cluster.events import ClusterEventBridge
from plugins.cluster.cluster_logging import ClusterLogger
from plugins.cluster.node_agent import NodeAgent
from plugins.cluster.server import ClusterHTTPServer
from plugins.cluster.store import open_store
from plugins.cluster.tools import set_runtime


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="cluster_command")

    subs.add_parser("serve", help="Run temporary master control plane HTTP server")
    subs.add_parser("worker", help="Run worker node agent loop (heartbeat + assignments)")
    subs.add_parser("status", help="Print cluster status from master")

    init_p = subs.add_parser("init-db", help="Initialize Postgres schema (master only)")
    init_p.add_argument("--database-url", default="", help="Override cluster.database_url")


def cluster_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "cluster_command", None)
    if cmd == "serve":
        return _cmd_serve()
    if cmd == "worker":
        return _cmd_worker()
    if cmd == "status":
        return _cmd_status()
    if cmd == "init-db":
        return _cmd_init_db(getattr(args, "database_url", ""))
    subparser = getattr(args, "_cluster_subparser", None)
    if subparser:
        subparser.print_help()
    return 1


def _build_runtime():
    cfg = load_cluster_config()
    cfg.enabled = True
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = open_store(cfg.database_url)
    store.ensure_schema()
    logger = ClusterLogger(cfg, store)
    events = ClusterEventBridge(cfg, store)
    controller = ClusterController(cfg, store, logger, events)
    set_runtime(controller=controller, store=store, logger=logger, events=events)
    return cfg, store, logger, events, controller


def _cmd_serve() -> int:
    cfg, store, logger, events, controller = _build_runtime()
    role = resolve_role(cfg)
    if role != "master":
        print(f"cluster.role resolves to {role}; serve expects master", file=sys.stderr)
        return 1
    server = ClusterHTTPServer(cfg, controller, logger)
    print(f"Cluster master on {cfg.bind_host}:{cfg.bind_port} (epoch pending startup)")
    server.start(block=True)
    return 0


def _cmd_worker() -> int:
    cfg, store, logger, _events, _controller = _build_runtime()
    role = resolve_role(cfg)
    if role != "worker":
        print(f"Warning: cluster.role resolves to {role}; worker loop starting anyway", file=sys.stderr)
    agent = NodeAgent(cfg, store, logger)
    try:
        agent.run_loop()
    except KeyboardInterrupt:
        agent.stop()
    return 0


def _cmd_status() -> int:
    cfg = load_cluster_config()
    client = ClusterClient(cfg)
    data = client.status()
    print(json.dumps(data, indent=2))
    return 0


def _cmd_init_db(database_url: str) -> int:
    cfg = load_cluster_config()
    url = database_url or cfg.database_url
    if not url:
        print("Set cluster.database_url or pass --database-url", file=sys.stderr)
        return 1
    store = open_store(url)
    store.ensure_schema()
    print(f"Schema initialized at {url}")
    return 0
