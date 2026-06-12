"""Master controller state machine — registration, jobs, assignments, sweeps."""

from __future__ import annotations

import socket
import time
from typing import Any, Dict, List, Optional, Tuple

from plugins.cluster.config import ClusterConfig, compute_config_hash
from plugins.cluster.events import ClusterEventBridge
from plugins.cluster.cluster_logging import ClusterLogger
from plugins.cluster.models import (
    ClusterEvent,
    GpuInfo,
    HeartbeatPayload,
    JobRecord,
    JobSpec,
    NodeRecord,
    RankAssignment,
    ValidationResult,
    new_id,
)
from plugins.cluster.store import ClusterStore
from plugins.cluster.training import (
    parse_master_addr,
    pick_rendezvous_port,
    validate_job_spec,
)
from plugins.cluster.node_capabilities import (
    LogicalJobRequirements,
    select_nodes_for_job,
)


class ClusterController:
    """Master-side orchestration logic."""

    def __init__(
        self,
        cfg: ClusterConfig,
        store: ClusterStore,
        logger: ClusterLogger,
        events: ClusterEventBridge,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.logger = logger
        self.events = events
        self._master_epoch = store.get_master_epoch()

    def startup(self) -> int:
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self._master_epoch = self.store.bump_master_epoch()
        self.events.emit(
            "master_started",
            {"master_epoch": self._master_epoch, "node_id": self.cfg.node_id},
            route_mode="record",
        )
        return self._master_epoch

    def register_node(
        self,
        *,
        node_id: str,
        advertised_addr: str,
        gpus: List[Dict[str, Any]],
        agent_version: str,
        config_hash: str,
    ) -> NodeRecord:
        gpu_objs = [
            GpuInfo(index=int(g.get("index", i)), name=str(g.get("name", "")), memory_mb=int(g.get("memory_mb", 0)))
            for i, g in enumerate(gpus)
        ]
        node = NodeRecord(
            node_id=node_id,
            advertised_addr=advertised_addr,
            state="ready",
            gpus=gpu_objs,
            agent_version=agent_version,
            config_hash=config_hash,
        )
        self.store.upsert_node(node)
        self.events.emit(
            "node_registered",
            {"node_id": node_id, "advertised_addr": advertised_addr},
            node_id=node_id,
        )
        return node

    def heartbeat(self, hb: HeartbeatPayload) -> Dict[str, Any]:
        self.store.record_heartbeat(hb)
        self.events.emit(
            "heartbeat",
            {"state": hb.state, "metrics": hb.metrics},
            node_id=hb.node_id,
            route_mode="record",
        )
        assignment = self.store.get_assignment_for_node(hb.node_id)
        return {
            "ok": True,
            "master_epoch": self._master_epoch,
            "assignment": assignment.to_dict() if assignment else None,
        }

    def status(self) -> Dict[str, Any]:
        nodes = self.store.list_nodes()
        stale = set(self.store.stale_node_ids(self.cfg.heartbeat_ttl_sec))
        jobs = self.store.list_jobs(limit=20)
        return {
            "master_epoch": self._master_epoch,
            "master_url": self.cfg.master_url,
            "nodes": [
                {
                    **n.to_dict(),
                    "stale": n.node_id in stale,
                }
                for n in nodes
            ],
            "jobs": [j.to_dict() for j in jobs],
            "stale_nodes": list(stale),
        }

    def validate_config(self, raw: Dict[str, Any]) -> ValidationResult:
        return validate_job_spec(raw)

    def submit_job(self, raw: Dict[str, Any], *, request_id: str = "") -> Dict[str, Any]:
        validation = validate_job_spec(raw)
        if not validation.ok:
            return {"success": False, "errors": validation.errors}

        norm = validation.normalized
        req = LogicalJobRequirements.from_spec_dict(norm)
        stale_ids = set(self.store.stale_node_ids(self.cfg.heartbeat_ttl_sec))
        candidates = [n for n in self.store.list_nodes() if n.state in ("ready", "busy")]
        nnodes = int(norm["nnodes"])
        nproc = int(norm["nproc_per_node"])

        selected, rejections = select_nodes_for_job(
            candidates,
            req,
            nnodes=nnodes,
            nproc_per_node=nproc,
            stale_ids=stale_ids,
            metrics_by_node={
                n.node_id: self.store.get_node_metrics(n.node_id) for n in candidates
            },
        )
        if len(selected) < nnodes:
            return {
                "success": False,
                "errors": [
                    f"need {nnodes} eligible nodes matching job requirements, "
                    f"have {len(selected)}",
                    *rejections[:10],
                ],
            }

        master_addr = parse_master_addr(self.cfg.master_url, selected[0].advertised_addr)
        master_port = pick_rendezvous_port(self.cfg)
        world_size = nnodes * nproc

        spec = JobSpec(
            job_id=norm["job_id"],
            script=norm["script"],
            script_args=norm["script_args"],
            nnodes=nnodes,
            nproc_per_node=nproc,
            framework=norm["framework"],
            env=norm["env"],
            working_dir=norm["working_dir"],
            idempotency_key=norm["idempotency_key"],
            extra=norm.get("extra") or {},
        )

        if spec.idempotency_key:
            existing = self.store.get_job_by_idempotency(spec.idempotency_key)
            if existing:
                assignments = self.store.list_assignments_for_job(existing.job_id)
                return {
                    "success": True,
                    "job": existing.to_dict(),
                    "assignments": [a.to_dict() for a in assignments],
                    "idempotent": True,
                }

        job = JobRecord(
            job_id=spec.job_id,
            spec=spec,
            state="assigning",
            master_epoch=self._master_epoch,
            job_generation=1,
            master_addr=master_addr,
            master_port=master_port,
        )

        assignments: List[RankAssignment] = []
        for rank, node in enumerate(selected):
            gpu_ids = [g.index for g in node.gpus[:nproc]] or list(range(nproc))
            partial = RankAssignment(
                assignment_id=new_id("asg-"),
                job_id=job.job_id,
                node_id=node.node_id,
                node_rank=rank,
                nproc_per_node=nproc,
                nnodes=nnodes,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                master_epoch=self._master_epoch,
                job_generation=job.job_generation,
                gpus=gpu_ids,
                working_dir=spec.working_dir,
                job_spec=spec.to_dict(),
            )
            # Launch command is built on the worker after local path/env resolution.
            partial.launch_command = []
            partial.env = dict(spec.env)
            assignments.append(partial)

        self.store.create_job(job, assignments)
        self.store.update_job_state(job.job_id, "running")
        self.events.emit(
            "job_submitted",
            {"job_id": job.job_id, "nnodes": nnodes, "world_size": world_size},
            job_id=job.job_id,
            request_id=request_id,
        )

        return {
            "success": True,
            "job": job.to_dict(),
            "assignments": [a.to_dict() for a in assignments],
        }

    def job_status(self, job_id: str) -> Dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            return {"success": False, "error": "job not found"}
        assignments = self.store.list_assignments_for_job(job_id)
        logs = self.store.query_logs(job_id=job_id, limit=20)
        return {
            "success": True,
            "job": job.to_dict(),
            "assignments": [a.to_dict() for a in assignments],
            "recent_logs": logs,
        }

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            return {"success": False, "error": "job not found"}
        self.store.update_job_state(job_id, "stopped")
        self.events.emit(
            "job_stopped",
            {"job_id": job_id},
            job_id=job_id,
            route_mode="execute_direct",
        )
        return {"success": True, "job_id": job_id, "state": "stopped"}

    def node_action(self, node_id: str, action: str) -> Dict[str, Any]:
        node = self.store.get_node(node_id)
        if not node:
            return {"success": False, "error": "node not found"}

        if action == "quarantine":
            node.state = "quarantined"
        elif action == "restore":
            node.state = "ready"
        elif action == "revalidate":
            node.state = "registering"
        else:
            return {"success": False, "error": f"unknown action: {action}"}

        node.updated_at = time.time()
        self.store.upsert_node(node)
        self.events.emit(
            "node_action",
            {"node_id": node_id, "action": action, "state": node.state},
            node_id=node_id,
        )
        return {"success": True, "node": node.to_dict()}

    def sweep_stale_nodes(self) -> List[str]:
        stale = self.store.stale_node_ids(self.cfg.heartbeat_ttl_sec)
        for node_id in stale:
            node = self.store.get_node(node_id)
            if node and node.state != "lost":
                node.state = "lost"
                self.store.upsert_node(node)
                self.events.emit(
                    "node_lost",
                    {"node_id": node_id},
                    node_id=node_id,
                )
        return stale

    def ack_assignment(
        self, assignment_id: str, node_id: str, job_generation: int, state: str
    ) -> bool:
        return self.store.ack_assignment(assignment_id, node_id, job_generation, state)

    def report_job_outcome(
        self, job_id: str, *, success: bool, summary: str = "", node_id: str = ""
    ) -> None:
        state = "succeeded" if success else "failed"
        self.store.update_job_state(job_id, state, error_summary=summary if not success else "")
        event_type = "job_completed" if success else "job_failed"
        self.events.emit(
            event_type,
            {"summary": summary or state, "job_id": job_id},
            job_id=job_id,
            node_id=node_id,
        )
