"""Worker node agent — register, heartbeat, assignment validation, process launch."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from plugins.cluster.client import ClusterClient
from plugins.cluster.config import ClusterConfig, compute_config_hash
from plugins.cluster.cluster_logging import ClusterLogger
from plugins.cluster.node_capabilities import collect_local_metrics, resolve_local_launch, validate_resolved_launch
from plugins.cluster.models import GpuInfo, HeartbeatPayload, JobSpec, RankAssignment
from plugins.cluster.store import ClusterStore
from plugins.cluster.training import validate_local_assignment

_log = logging.getLogger(__name__)

_ACTIVE_PROCS: Dict[str, subprocess.Popen] = {}
_ACTIVE_LOCK = threading.Lock()


def _detect_gpus() -> List[GpuInfo]:
    """Best-effort GPU detection without nvidia-smi dependency."""
    gpus: List[GpuInfo] = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(GpuInfo(index=i, name=props.name, memory_mb=props.total_memory // (1024 * 1024)))
    except Exception:
        pass
    if not gpus:
        gpus.append(GpuInfo(index=0, name="cpu", memory_mb=0))
    return gpus


def _local_addrs() -> List[str]:
    addrs = {"127.0.0.1", "localhost", "::1"}
    try:
        hostname = socket.gethostname()
        addrs.add(hostname)
        addrs.update(socket.gethostbyname_ex(hostname)[2])
    except OSError:
        pass
    try:
        addrs.add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    return sorted(addrs)


def _advertised_addr(cfg: ClusterConfig) -> str:
    env = os.environ.get("GPUCLOUD_CLUSTER_ADVERTISED_ADDR", "").strip()
    if env:
        return env
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


class NodeAgent:
    def __init__(
        self,
        cfg: ClusterConfig,
        store: ClusterStore,
        logger: ClusterLogger,
        client: Optional[ClusterClient] = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.logger = logger
        self.client = client or ClusterClient(cfg)
        self._stop = threading.Event()
        self._config_hash = compute_config_hash(cfg)
        self._running_job_id: Optional[str] = None

    def register(self) -> Dict[str, Any]:
        return self.client.register(
            node_id=self.cfg.node_id,
            advertised_addr=_advertised_addr(self.cfg),
            gpus=[g.to_dict() for g in _detect_gpus()],
            agent_version=_agent_version(),
            config_hash=self._config_hash,
        )

    def heartbeat_once(self) -> Dict[str, Any]:
        gpus = _detect_gpus()
        metrics = collect_local_metrics(self.cfg, gpus)
        payload = HeartbeatPayload(
            node_id=self.cfg.node_id,
            state="busy" if self._running_job_id else "ready",
            gpus=gpus,
            config_hash=self._config_hash,
            running_job_id=self._running_job_id,
            metrics=metrics,
        )
        return self.client.heartbeat(payload.to_dict())

    def run_loop(self) -> None:
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.register()
        _log.info("cluster worker %s registered with master %s", self.cfg.node_id, self.cfg.master_url)
        while not self._stop.wait(self.cfg.heartbeat_interval_sec):
            try:
                resp = self.heartbeat_once()
                assignment_raw = resp.get("assignment")
                if assignment_raw and not self._running_job_id:
                    assignment = RankAssignment(**assignment_raw)
                    self._maybe_launch(assignment)
            except Exception as exc:
                self.logger.log_error(
                    error_type="node_heartbeat",
                    message=str(exc),
                    node_id=self.cfg.node_id,
                )

    def stop(self) -> None:
        self._stop.set()
        self._stop_all_jobs()

    def _maybe_launch(self, assignment: RankAssignment) -> None:
        validation = validate_local_assignment(
            assignment,
            node_id=self.cfg.node_id,
            config_hash=self._config_hash,
            local_addrs=_local_addrs(),
        )
        if not validation.ok:
            self.logger.log_error(
                error_type="assignment_validation",
                message="; ".join(validation.errors),
                job_id=assignment.job_id,
                node_id=self.cfg.node_id,
            )
            self.client.ack_assignment(
                assignment.assignment_id,
                self.cfg.node_id,
                assignment.job_generation,
                "rejected",
            )
            return

        spec_dict = assignment.job_spec or {}
        if not spec_dict:
            self.logger.log_error(
                error_type="assignment_validation",
                message="missing job_spec on assignment",
                job_id=assignment.job_id,
                node_id=self.cfg.node_id,
            )
            self.client.ack_assignment(
                assignment.assignment_id,
                self.cfg.node_id,
                assignment.job_generation,
                "rejected",
            )
            return

        spec = JobSpec(
            script=str(spec_dict.get("script") or ""),
            script_args=list(spec_dict.get("script_args") or []),
            nnodes=int(spec_dict.get("nnodes") or assignment.nnodes),
            nproc_per_node=int(spec_dict.get("nproc_per_node") or assignment.nproc_per_node),
            framework=str(spec_dict.get("framework") or "torchrun"),
            env=dict(spec_dict.get("env") or {}),
            working_dir=str(spec_dict.get("working_dir") or assignment.working_dir),
            job_id=str(spec_dict.get("job_id") or assignment.job_id),
            idempotency_key=str(spec_dict.get("idempotency_key") or ""),
            extra=dict(spec_dict.get("extra") or {}),
        )

        resolved = resolve_local_launch(self.cfg, spec, assignment)
        resolved_validation = validate_resolved_launch(resolved)
        if not resolved_validation.ok:
            msg = "; ".join(resolved_validation.errors)
            self.logger.log_error(
                error_type="path_resolution",
                message=msg,
                job_id=assignment.job_id,
                node_id=self.cfg.node_id,
            )
            self.client.ack_assignment(
                assignment.assignment_id,
                self.cfg.node_id,
                assignment.job_generation,
                "rejected",
            )
            return

        assignment = RankAssignment(
            assignment_id=assignment.assignment_id,
            job_id=assignment.job_id,
            node_id=assignment.node_id,
            node_rank=assignment.node_rank,
            nproc_per_node=assignment.nproc_per_node,
            nnodes=assignment.nnodes,
            world_size=assignment.world_size,
            master_addr=assignment.master_addr,
            master_port=assignment.master_port,
            master_epoch=assignment.master_epoch,
            job_generation=assignment.job_generation,
            gpus=assignment.gpus,
            launch_command=resolved.launch_command,
            env=resolved.env,
            working_dir=resolved.working_dir,
            job_spec=spec_dict,
        )

        self.client.ack_assignment(
            assignment.assignment_id,
            self.cfg.node_id,
            assignment.job_generation,
            "accepted",
        )
        self._launch_process(assignment)

    def _launch_process(self, assignment: RankAssignment) -> None:
        cwd = Path(assignment.working_dir).expanduser()
        if not cwd.is_absolute():
            cwd = Path.cwd() / cwd
        if not cwd.exists():
            self.logger.log_error(
                error_type="path_resolution",
                message=f"working_dir does not exist: {cwd}",
                job_id=assignment.job_id,
                node_id=self.cfg.node_id,
            )
            self.client.ack_assignment(
                assignment.assignment_id,
                self.cfg.node_id,
                assignment.job_generation,
                "rejected",
            )
            return

        if not assignment.launch_command:
            self.logger.log_error(
                error_type="path_resolution",
                message="empty launch_command after resolution",
                job_id=assignment.job_id,
                node_id=self.cfg.node_id,
            )
            self.client.ack_assignment(
                assignment.assignment_id,
                self.cfg.node_id,
                assignment.job_generation,
                "rejected",
            )
            return

        env = os.environ.copy()
        env.update(assignment.env)

        log_dir = self.cfg.logs_dir / assignment.job_id
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{self.cfg.node_id}.stdout.log"
        stderr_path = log_dir / f"{self.cfg.node_id}.stderr.log"

        run_id = self.logger.start_process(
            job_id=assignment.job_id,
            node_id=self.cfg.node_id,
            command=assignment.launch_command,
            cwd=str(cwd),
            env=env,
        )

        with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
            proc = subprocess.Popen(
                assignment.launch_command,
                cwd=str(cwd),
                env=env,
                stdout=out_f,
                stderr=err_f,
            )

        with _ACTIVE_LOCK:
            _ACTIVE_PROCS[assignment.job_id] = proc

        self._running_job_id = assignment.job_id
        self.client.ack_assignment(
            assignment.assignment_id,
            self.cfg.node_id,
            assignment.job_generation,
            "running",
        )

        watcher = threading.Thread(
            target=self._watch_process,
            args=(proc, assignment, run_id, stdout_path, stderr_path),
            daemon=True,
        )
        watcher.start()

    def _watch_process(
        self,
        proc: subprocess.Popen,
        assignment: RankAssignment,
        run_id: str,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        exit_code = proc.wait()
        self.logger.finish_process(
            run_id,
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        with _ACTIVE_LOCK:
            _ACTIVE_PROCS.pop(assignment.job_id, None)
        self._running_job_id = None
        success = exit_code == 0
        try:
            self.client.report_outcome(
                assignment.job_id,
                success=success,
                summary=f"exit_code={exit_code}",
                node_id=self.cfg.node_id,
            )
        except Exception as exc:
            self.logger.log_error(
                error_type="report_outcome",
                message=str(exc),
                job_id=assignment.job_id,
                node_id=self.cfg.node_id,
            )

    def _stop_all_jobs(self) -> None:
        with _ACTIVE_LOCK:
            procs = list(_ACTIVE_PROCS.items())
        for job_id, proc in procs:
            try:
                proc.terminate()
            except OSError:
                pass


def _agent_version() -> str:
    try:
        from gpucloud_cli import __version__
        return __version__
    except Exception:
        return "dev"
