"""Cluster control-plane state store — Postgres (production) or in-memory (tests)."""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from plugins.cluster.models import (
    AgentActionLog,
    ClusterEvent,
    GpuInfo,
    HeartbeatPayload,
    JobRecord,
    JobSpec,
    NodeRecord,
    ProcessRunLog,
    RankAssignment,
    new_id,
)

_log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cluster_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_nodes (
    node_id TEXT PRIMARY KEY,
    advertised_addr TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'registering',
    gpus JSONB NOT NULL DEFAULT '[]',
    agent_version TEXT NOT NULL DEFAULT '',
    config_hash TEXT NOT NULL DEFAULT '',
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_heartbeats (
    node_id TEXT PRIMARY KEY REFERENCES cluster_nodes(node_id) ON DELETE CASCADE,
    last_seen_at DOUBLE PRECISION NOT NULL,
    state TEXT NOT NULL DEFAULT 'ready',
    gpus JSONB NOT NULL DEFAULT '[]',
    config_hash TEXT NOT NULL DEFAULT '',
    running_job_id TEXT,
    metrics JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS cluster_jobs (
    job_id TEXT PRIMARY KEY,
    spec JSONB NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    master_epoch INTEGER NOT NULL DEFAULT 0,
    job_generation INTEGER NOT NULL DEFAULT 1,
    master_addr TEXT NOT NULL DEFAULT '',
    master_port INTEGER NOT NULL DEFAULT 0,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    error_summary TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS cluster_assignments (
    assignment_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES cluster_jobs(job_id) ON DELETE CASCADE,
    node_id TEXT NOT NULL REFERENCES cluster_nodes(node_id),
    node_rank INTEGER NOT NULL,
    nproc_per_node INTEGER NOT NULL,
    nnodes INTEGER NOT NULL,
    world_size INTEGER NOT NULL,
    master_addr TEXT NOT NULL,
    master_port INTEGER NOT NULL,
    master_epoch INTEGER NOT NULL,
    job_generation INTEGER NOT NULL,
    gpus JSONB NOT NULL DEFAULT '[]',
    launch_command JSONB NOT NULL DEFAULT '[]',
    env JSONB NOT NULL DEFAULT '{}',
    working_dir TEXT NOT NULL DEFAULT '.',
    state TEXT NOT NULL DEFAULT 'pending',
    validation_errors JSONB NOT NULL DEFAULT '[]',
    UNIQUE (job_id, node_id)
);

CREATE TABLE IF NOT EXISTS cluster_leases (
    lease_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    ref_id TEXT NOT NULL,
    fencing_token INTEGER NOT NULL DEFAULT 0,
    expires_at DOUBLE PRECISION NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_events (
    event_id TEXT PRIMARY KEY,
    ts DOUBLE PRECISION NOT NULL,
    request_id TEXT NOT NULL DEFAULT '',
    job_id TEXT NOT NULL DEFAULT '',
    node_id TEXT NOT NULL DEFAULT '',
    event_type TEXT NOT NULL,
    route_mode TEXT NOT NULL DEFAULT 'record',
    payload JSONB NOT NULL DEFAULT '{}',
    payload_hash TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS agent_actions (
    action_id TEXT PRIMARY KEY,
    ts DOUBLE PRECISION NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    turn_id TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL,
    tool_args JSONB NOT NULL DEFAULT '{}',
    decision TEXT NOT NULL DEFAULT '',
    result_summary TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS process_runs (
    run_id TEXT PRIMARY KEY,
    ts_start DOUBLE PRECISION NOT NULL,
    ts_end DOUBLE PRECISION,
    job_id TEXT NOT NULL DEFAULT '',
    node_id TEXT NOT NULL DEFAULT '',
    pid INTEGER,
    command JSONB NOT NULL DEFAULT '[]',
    cwd TEXT NOT NULL DEFAULT '',
    env_keys JSONB NOT NULL DEFAULT '[]',
    exit_code INTEGER
);

CREATE TABLE IF NOT EXISTS process_log_refs (
    ref_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES process_runs(run_id) ON DELETE CASCADE,
    stream TEXT NOT NULL,
    path TEXT NOT NULL,
    tail TEXT NOT NULL DEFAULT '',
    size_bytes BIGINT NOT NULL DEFAULT 0,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_errors (
    error_id TEXT PRIMARY KEY,
    ts DOUBLE PRECISION NOT NULL,
    error_type TEXT NOT NULL,
    message TEXT NOT NULL,
    traceback TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL DEFAULT '',
    job_id TEXT NOT NULL DEFAULT '',
    node_id TEXT NOT NULL DEFAULT '',
    action_id TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_cluster_events_job ON cluster_events(job_id);
CREATE INDEX IF NOT EXISTS idx_cluster_events_ts ON cluster_events(ts);
CREATE INDEX IF NOT EXISTS idx_agent_actions_ts ON agent_actions(ts);
CREATE INDEX IF NOT EXISTS idx_process_runs_job ON process_runs(job_id);
"""


def _gpus_from_json(raw: Any) -> List[GpuInfo]:
    out: List[GpuInfo] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            out.append(GpuInfo(
                index=int(item.get("index", 0)),
                name=str(item.get("name", "")),
                memory_mb=int(item.get("memory_mb", 0)),
            ))
    return out


def _gpus_to_json(gpus: List[GpuInfo]) -> List[Dict[str, Any]]:
    return [g.to_dict() for g in gpus]


class ClusterStore(ABC):
    @abstractmethod
    def ensure_schema(self) -> None: ...

    @abstractmethod
    def get_master_epoch(self) -> int: ...

    @abstractmethod
    def bump_master_epoch(self) -> int: ...

    @abstractmethod
    def upsert_node(self, node: NodeRecord) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[NodeRecord]: ...

    @abstractmethod
    def list_nodes(self) -> List[NodeRecord]: ...

    @abstractmethod
    def record_heartbeat(self, hb: HeartbeatPayload) -> None: ...

    @abstractmethod
    def get_node_metrics(self, node_id: str) -> Dict[str, Any]: ...

    @abstractmethod
    def stale_node_ids(self, ttl_sec: float, now: Optional[float] = None) -> List[str]: ...

    @abstractmethod
    def create_job(self, job: JobRecord, assignments: List[RankAssignment]) -> JobRecord: ...

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[JobRecord]: ...

    @abstractmethod
    def get_job_by_idempotency(self, key: str) -> Optional[JobRecord]: ...

    @abstractmethod
    def update_job_state(
        self, job_id: str, state: str, *, error_summary: str = "", expected_generation: Optional[int] = None
    ) -> bool: ...

    @abstractmethod
    def list_jobs(self, limit: int = 50) -> List[JobRecord]: ...

    @abstractmethod
    def get_assignment(self, assignment_id: str) -> Optional[RankAssignment]: ...

    @abstractmethod
    def get_assignment_for_node(self, node_id: str) -> Optional[RankAssignment]: ...

    @abstractmethod
    def list_assignments_for_job(self, job_id: str) -> List[RankAssignment]: ...

    @abstractmethod
    def ack_assignment(
        self, assignment_id: str, node_id: str, job_generation: int, state: str
    ) -> bool: ...

    @abstractmethod
    def append_event(self, event: ClusterEvent) -> None: ...

    @abstractmethod
    def list_events(
        self,
        *,
        job_id: str = "",
        node_id: str = "",
        limit: int = 100,
        since_ts: float = 0,
    ) -> List[ClusterEvent]: ...

    @abstractmethod
    def log_agent_action(self, action: AgentActionLog) -> None: ...

    @abstractmethod
    def log_process_run(self, run: ProcessRunLog) -> None: ...

    @abstractmethod
    def update_process_run(self, run_id: str, *, exit_code: int, ts_end: float) -> None: ...

    @abstractmethod
    def log_process_ref(
        self, run_id: str, stream: str, path: str, tail: str, size_bytes: int
    ) -> None: ...

    @abstractmethod
    def log_error(
        self,
        *,
        error_type: str,
        message: str,
        traceback: str = "",
        request_id: str = "",
        job_id: str = "",
        node_id: str = "",
        action_id: str = "",
    ) -> str: ...

    @abstractmethod
    def query_logs(
        self,
        *,
        job_id: str = "",
        node_id: str = "",
        limit: int = 50,
    ) -> Dict[str, Any]: ...


class MemoryClusterStore(ClusterStore):
    """In-memory store for tests and dev without Postgres."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._master_epoch = 1
        self._nodes: Dict[str, NodeRecord] = {}
        self._heartbeats: Dict[str, Dict[str, Any]] = {}
        self._jobs: Dict[str, JobRecord] = {}
        self._idempotency: Dict[str, str] = {}
        self._assignments: Dict[str, RankAssignment] = {}
        self._events: List[ClusterEvent] = []
        self._actions: List[AgentActionLog] = []
        self._runs: Dict[str, ProcessRunLog] = {}
        self._log_refs: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []

    def ensure_schema(self) -> None:
        return

    def get_master_epoch(self) -> int:
        with self._lock:
            return self._master_epoch

    def bump_master_epoch(self) -> int:
        with self._lock:
            self._master_epoch += 1
            return self._master_epoch

    def upsert_node(self, node: NodeRecord) -> None:
        with self._lock:
            self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[NodeRecord]:
        with self._lock:
            return self._nodes.get(node_id)

    def list_nodes(self) -> List[NodeRecord]:
        with self._lock:
            return list(self._nodes.values())

    def record_heartbeat(self, hb: HeartbeatPayload) -> None:
        with self._lock:
            self._heartbeats[hb.node_id] = {
                "last_seen_at": time.time(),
                "state": hb.state,
                "gpus": hb.gpus,
                "config_hash": hb.config_hash,
                "running_job_id": hb.running_job_id,
                "metrics": hb.metrics,
            }
            node = self._nodes.get(hb.node_id)
            if node:
                node.state = hb.state  # type: ignore[assignment]
                node.capabilities = dict(hb.metrics or {})
                node.updated_at = time.time()

    def get_node_metrics(self, node_id: str) -> Dict[str, Any]:
        with self._lock:
            hb = self._heartbeats.get(node_id) or {}
            metrics = hb.get("metrics") or {}
            return dict(metrics) if isinstance(metrics, dict) else {}

    def stale_node_ids(self, ttl_sec: float, now: Optional[float] = None) -> List[str]:
        now = now or time.time()
        with self._lock:
            stale = []
            for node_id, hb in self._heartbeats.items():
                if now - float(hb.get("last_seen_at", 0)) > ttl_sec:
                    stale.append(node_id)
            return stale

    def create_job(self, job: JobRecord, assignments: List[RankAssignment]) -> JobRecord:
        with self._lock:
            if job.spec.idempotency_key:
                existing = self._idempotency.get(job.spec.idempotency_key)
                if existing and existing in self._jobs:
                    return self._jobs[existing]
                self._idempotency[job.spec.idempotency_key] = job.job_id
            self._jobs[job.job_id] = job
            for a in assignments:
                self._assignments[a.assignment_id] = a
            return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_job_by_idempotency(self, key: str) -> Optional[JobRecord]:
        with self._lock:
            jid = self._idempotency.get(key)
            return self._jobs.get(jid) if jid else None

    def update_job_state(
        self, job_id: str, state: str, *, error_summary: str = "", expected_generation: Optional[int] = None
    ) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if expected_generation is not None and job.job_generation != expected_generation:
                return False
            job.state = state  # type: ignore[assignment]
            job.updated_at = time.time()
            if error_summary:
                job.error_summary = error_summary
            return True

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]

    def get_assignment(self, assignment_id: str) -> Optional[RankAssignment]:
        with self._lock:
            return self._attach_job_spec(self._assignments.get(assignment_id))

    def get_assignment_for_node(self, node_id: str) -> Optional[RankAssignment]:
        with self._lock:
            for a in self._assignments.values():
                if a.node_id == node_id and a.state in ("pending", "accepted", "running"):
                    return self._attach_job_spec(a)
            return None

    def _attach_job_spec(self, assignment: Optional[RankAssignment]) -> Optional[RankAssignment]:
        if not assignment or assignment.job_spec:
            return assignment
        job = self.get_job(assignment.job_id)
        if job:
            assignment.job_spec = job.spec.to_dict()
        return assignment

    def list_assignments_for_job(self, job_id: str) -> List[RankAssignment]:
        with self._lock:
            return [a for a in self._assignments.values() if a.job_id == job_id]

    def ack_assignment(
        self, assignment_id: str, node_id: str, job_generation: int, state: str
    ) -> bool:
        with self._lock:
            a = self._assignments.get(assignment_id)
            if not a or a.node_id != node_id or a.job_generation != job_generation:
                return False
            a.state = state
            return True

    def append_event(self, event: ClusterEvent) -> None:
        with self._lock:
            self._events.append(event)

    def list_events(
        self,
        *,
        job_id: str = "",
        node_id: str = "",
        limit: int = 100,
        since_ts: float = 0,
    ) -> List[ClusterEvent]:
        with self._lock:
            out = []
            for ev in reversed(self._events):
                if since_ts and ev.ts < since_ts:
                    continue
                if job_id and ev.job_id != job_id:
                    continue
                if node_id and ev.node_id != node_id:
                    continue
                out.append(ev)
                if len(out) >= limit:
                    break
            return out

    def log_agent_action(self, action: AgentActionLog) -> None:
        with self._lock:
            self._actions.append(action)

    def log_process_run(self, run: ProcessRunLog) -> None:
        with self._lock:
            self._runs[run.run_id] = run

    def update_process_run(self, run_id: str, *, exit_code: int, ts_end: float) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run:
                run.exit_code = exit_code
                run.ts_end = ts_end

    def log_process_ref(
        self, run_id: str, stream: str, path: str, tail: str, size_bytes: int
    ) -> None:
        with self._lock:
            self._log_refs.append({
                "ref_id": new_id("log-"),
                "run_id": run_id,
                "stream": stream,
                "path": path,
                "tail": tail,
                "size_bytes": size_bytes,
                "updated_at": time.time(),
            })

    def log_error(
        self,
        *,
        error_type: str,
        message: str,
        traceback: str = "",
        request_id: str = "",
        job_id: str = "",
        node_id: str = "",
        action_id: str = "",
    ) -> str:
        eid = new_id("err-")
        with self._lock:
            self._errors.append({
                "error_id": eid,
                "ts": time.time(),
                "error_type": error_type,
                "message": message,
                "traceback": traceback,
                "request_id": request_id,
                "job_id": job_id,
                "node_id": node_id,
                "action_id": action_id,
            })
        return eid

    def query_logs(
        self,
        *,
        job_id: str = "",
        node_id: str = "",
        limit: int = 50,
    ) -> Dict[str, Any]:
        with self._lock:
            events = self.list_events(job_id=job_id, node_id=node_id, limit=limit)
            runs = [
                r.to_dict() for r in self._runs.values()
                if (not job_id or r.job_id == job_id)
                and (not node_id or r.node_id == node_id)
            ][:limit]
            refs = [
                r for r in self._log_refs
                if any(pr["run_id"] == r["run_id"] for pr in runs) or not runs
            ][:limit]
            errors = [
                e for e in self._errors
                if (not job_id or e.get("job_id") == job_id)
                and (not node_id or e.get("node_id") == node_id)
            ][:limit]
            return {
                "events": [e.to_dict() for e in events],
                "process_runs": runs,
                "process_log_refs": refs,
                "errors": errors,
                "agent_actions": [a.to_dict() for a in self._actions[-limit:]],
            }


class PostgresClusterStore(ClusterStore):
    """Postgres-backed store using psycopg3."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._conn = None

    def _connect(self):
        import psycopg

        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._url, autocommit=False)
        return self._conn

    @contextmanager
    def _tx(self) -> Iterator[Any]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def ensure_schema(self) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()

    def get_master_epoch(self) -> int:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM cluster_meta WHERE key = 'master_epoch'"
                )
                row = cur.fetchone()
                if row:
                    return int(row[0])
                cur.execute(
                    "INSERT INTO cluster_meta (key, value) VALUES ('master_epoch', '1')"
                )
                return 1

    def bump_master_epoch(self) -> int:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cluster_meta (key, value) VALUES ('master_epoch', '1')
                    ON CONFLICT (key) DO UPDATE
                    SET value = (cluster_meta.value::int + 1)::text
                    RETURNING value
                    """
                )
                return int(cur.fetchone()[0])

    def upsert_node(self, node: NodeRecord) -> None:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cluster_nodes
                    (node_id, advertised_addr, state, gpus, agent_version, config_hash, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (node_id) DO UPDATE SET
                      advertised_addr=EXCLUDED.advertised_addr,
                      state=EXCLUDED.state,
                      gpus=EXCLUDED.gpus,
                      agent_version=EXCLUDED.agent_version,
                      config_hash=EXCLUDED.config_hash,
                      updated_at=EXCLUDED.updated_at
                    """,
                    (
                        node.node_id, node.advertised_addr, node.state,
                        json.dumps(_gpus_to_json(node.gpus)),
                        node.agent_version, node.config_hash,
                        node.created_at, node.updated_at,
                    ),
                )

    def get_node(self, node_id: str) -> Optional[NodeRecord]:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM cluster_nodes WHERE node_id = %s", (node_id,))
                row = cur.fetchone()
                if not row:
                    return None
                cols = [d.name for d in cur.description]
                data = dict(zip(cols, row))
                return NodeRecord(
                    node_id=data["node_id"],
                    advertised_addr=data["advertised_addr"],
                    state=data["state"],
                    gpus=_gpus_from_json(data["gpus"]),
                    agent_version=data["agent_version"],
                    config_hash=data["config_hash"],
                    created_at=float(data["created_at"]),
                    updated_at=float(data["updated_at"]),
                )

    def list_nodes(self) -> List[NodeRecord]:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT node_id FROM cluster_nodes ORDER BY node_id")
                return [self.get_node(r[0]) for r in cur.fetchall() if r[0]]  # type: ignore[misc]

    def record_heartbeat(self, hb: HeartbeatPayload) -> None:
        now = time.time()
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cluster_heartbeats
                    (node_id, last_seen_at, state, gpus, config_hash, running_job_id, metrics)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (node_id) DO UPDATE SET
                      last_seen_at=EXCLUDED.last_seen_at,
                      state=EXCLUDED.state,
                      gpus=EXCLUDED.gpus,
                      config_hash=EXCLUDED.config_hash,
                      running_job_id=EXCLUDED.running_job_id,
                      metrics=EXCLUDED.metrics
                    """,
                    (
                        hb.node_id, now, hb.state,
                        json.dumps(_gpus_to_json(hb.gpus)),
                        hb.config_hash, hb.running_job_id,
                        json.dumps(hb.metrics),
                    ),
                )
                cur.execute(
                    "UPDATE cluster_nodes SET state=%s, updated_at=%s WHERE node_id=%s",
                    (hb.state, now, hb.node_id),
                )

    def get_node_metrics(self, node_id: str) -> Dict[str, Any]:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT metrics FROM cluster_heartbeats WHERE node_id = %s",
                    (node_id,),
                )
                row = cur.fetchone()
                if not row or not row[0]:
                    return {}
                raw = row[0]
                if isinstance(raw, dict):
                    return dict(raw)
                return json.loads(raw)

    def stale_node_ids(self, ttl_sec: float, now: Optional[float] = None) -> List[str]:
        now = now or time.time()
        cutoff = now - ttl_sec
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT node_id FROM cluster_heartbeats WHERE last_seen_at < %s",
                    (cutoff,),
                )
                return [r[0] for r in cur.fetchall()]

    def create_job(self, job: JobRecord, assignments: List[RankAssignment]) -> JobRecord:
        with self._tx() as conn:
            with conn.cursor() as cur:
                if job.spec.idempotency_key:
                    cur.execute(
                        "SELECT job_id FROM cluster_jobs WHERE idempotency_key = %s",
                        (job.spec.idempotency_key,),
                    )
                    row = cur.fetchone()
                    if row:
                        existing = self.get_job(row[0])
                        if existing:
                            return existing
                cur.execute(
                    """
                    INSERT INTO cluster_jobs
                    (job_id, spec, state, master_epoch, job_generation, master_addr, master_port,
                     created_at, updated_at, error_summary, idempotency_key)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        job.job_id, json.dumps(job.spec.to_dict()), job.state,
                        job.master_epoch, job.job_generation,
                        job.master_addr, job.master_port,
                        job.created_at, job.updated_at, job.error_summary,
                        job.spec.idempotency_key or None,
                    ),
                )
                for a in assignments:
                    cur.execute(
                        """
                        INSERT INTO cluster_assignments
                        (assignment_id, job_id, node_id, node_rank, nproc_per_node, nnodes,
                         world_size, master_addr, master_port, master_epoch, job_generation,
                         gpus, launch_command, env, working_dir, state, validation_errors)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            a.assignment_id, a.job_id, a.node_id, a.node_rank,
                            a.nproc_per_node, a.nnodes, a.world_size,
                            a.master_addr, a.master_port, a.master_epoch, a.job_generation,
                            json.dumps(a.gpus), json.dumps(a.launch_command),
                            json.dumps(a.env), a.working_dir, a.state,
                            json.dumps(a.validation_errors),
                        ),
                    )
        return job

    def _row_to_job(self, data: Dict[str, Any]) -> JobRecord:
        spec_raw = data["spec"]
        if isinstance(spec_raw, str):
            spec_raw = json.loads(spec_raw)
        spec = JobSpec(**spec_raw)
        return JobRecord(
            job_id=data["job_id"],
            spec=spec,
            state=data["state"],
            master_epoch=int(data["master_epoch"]),
            job_generation=int(data["job_generation"]),
            master_addr=data["master_addr"],
            master_port=int(data["master_port"]),
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
            error_summary=data.get("error_summary") or "",
        )

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM cluster_jobs WHERE job_id = %s", (job_id,))
                row = cur.fetchone()
                if not row:
                    return None
                cols = [d.name for d in cur.description]
                return self._row_to_job(dict(zip(cols, row)))

    def get_job_by_idempotency(self, key: str) -> Optional[JobRecord]:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT job_id FROM cluster_jobs WHERE idempotency_key = %s", (key,)
                )
                row = cur.fetchone()
                return self.get_job(row[0]) if row else None

    def update_job_state(
        self, job_id: str, state: str, *, error_summary: str = "", expected_generation: Optional[int] = None
    ) -> bool:
        with self._tx() as conn:
            with conn.cursor() as cur:
                if expected_generation is not None:
                    cur.execute(
                        """
                        UPDATE cluster_jobs SET state=%s, updated_at=%s, error_summary=%s
                        WHERE job_id=%s AND job_generation=%s
                        """,
                        (state, time.time(), error_summary, job_id, expected_generation),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE cluster_jobs SET state=%s, updated_at=%s, error_summary=%s
                        WHERE job_id=%s
                        """,
                        (state, time.time(), error_summary, job_id),
                    )
                return cur.rowcount > 0

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT job_id FROM cluster_jobs ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
                return [self.get_job(r[0]) for r in cur.fetchall() if r[0]]  # type: ignore[misc]

    def _row_to_assignment(self, data: Dict[str, Any]) -> RankAssignment:
        def _loads(v, default):
            if v is None:
                return default
            if isinstance(v, (list, dict)):
                return v
            return json.loads(v)

        return RankAssignment(
            assignment_id=data["assignment_id"],
            job_id=data["job_id"],
            node_id=data["node_id"],
            node_rank=int(data["node_rank"]),
            nproc_per_node=int(data["nproc_per_node"]),
            nnodes=int(data["nnodes"]),
            world_size=int(data["world_size"]),
            master_addr=data["master_addr"],
            master_port=int(data["master_port"]),
            master_epoch=int(data["master_epoch"]),
            job_generation=int(data["job_generation"]),
            gpus=_loads(data["gpus"], []),
            launch_command=_loads(data["launch_command"], []),
            env=_loads(data["env"], {}),
            working_dir=data["working_dir"],
            state=data["state"],
            validation_errors=_loads(data["validation_errors"], []),
        )

    def get_assignment(self, assignment_id: str) -> Optional[RankAssignment]:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM cluster_assignments WHERE assignment_id = %s",
                    (assignment_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                cols = [d.name for d in cur.description]
                return self._attach_job_spec(self._row_to_assignment(dict(zip(cols, row))))

    def _attach_job_spec(self, assignment: Optional[RankAssignment]) -> Optional[RankAssignment]:
        if not assignment or assignment.job_spec:
            return assignment
        job = self.get_job(assignment.job_id)
        if job:
            assignment.job_spec = job.spec.to_dict()
        return assignment

    def get_assignment_for_node(self, node_id: str) -> Optional[RankAssignment]:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT assignment_id FROM cluster_assignments
                    WHERE node_id = %s AND state IN ('pending','accepted','running')
                    ORDER BY assignment_id DESC LIMIT 1
                    """,
                    (node_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return self.get_assignment(row[0])

    def list_assignments_for_job(self, job_id: str) -> List[RankAssignment]:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT assignment_id FROM cluster_assignments WHERE job_id = %s",
                    (job_id,),
                )
                return [self.get_assignment(r[0]) for r in cur.fetchall() if r[0]]  # type: ignore[misc]

    def ack_assignment(
        self, assignment_id: str, node_id: str, job_generation: int, state: str
    ) -> bool:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE cluster_assignments SET state=%s
                    WHERE assignment_id=%s AND node_id=%s AND job_generation=%s
                    """,
                    (state, assignment_id, node_id, job_generation),
                )
                return cur.rowcount > 0

    def append_event(self, event: ClusterEvent) -> None:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cluster_events
                    (event_id, ts, request_id, job_id, node_id, event_type, route_mode, payload, payload_hash)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (
                        event.event_id, event.ts, event.request_id,
                        event.job_id, event.node_id, event.event_type,
                        event.route_mode, json.dumps(event.payload), "",
                    ),
                )

    def list_events(
        self,
        *,
        job_id: str = "",
        node_id: str = "",
        limit: int = 100,
        since_ts: float = 0,
    ) -> List[ClusterEvent]:
        clauses = ["1=1"]
        params: List[Any] = []
        if job_id:
            clauses.append("job_id = %s")
            params.append(job_id)
        if node_id:
            clauses.append("node_id = %s")
            params.append(node_id)
        if since_ts:
            clauses.append("ts >= %s")
            params.append(since_ts)
        params.append(limit)
        sql = f"""
            SELECT * FROM cluster_events
            WHERE {' AND '.join(clauses)}
            ORDER BY ts DESC LIMIT %s
        """
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                cols = [d.name for d in cur.description]
                out = []
                for row in rows:
                    data = dict(zip(cols, row))
                    payload = data["payload"]
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    out.append(ClusterEvent(
                        event_id=data["event_id"],
                        event_type=data["event_type"],
                        payload=payload,
                        route_mode=data["route_mode"],
                        job_id=data["job_id"],
                        node_id=data["node_id"],
                        request_id=data["request_id"],
                        ts=float(data["ts"]),
                    ))
                return out

    def log_agent_action(self, action: AgentActionLog) -> None:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_actions
                    (action_id, ts, session_id, turn_id, tool_name, tool_args, decision, result_summary)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        action.action_id, action.ts, action.session_id, action.turn_id,
                        action.tool_name, json.dumps(action.tool_args),
                        action.decision, action.result_summary,
                    ),
                )

    def log_process_run(self, run: ProcessRunLog) -> None:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO process_runs
                    (run_id, ts_start, ts_end, job_id, node_id, pid, command, cwd, env_keys, exit_code)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        run.run_id, run.ts_start, run.ts_end, run.job_id, run.node_id,
                        run.pid, json.dumps(run.command), run.cwd,
                        json.dumps(run.env_keys), run.exit_code,
                    ),
                )

    def update_process_run(self, run_id: str, *, exit_code: int, ts_end: float) -> None:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE process_runs SET exit_code=%s, ts_end=%s WHERE run_id=%s",
                    (exit_code, ts_end, run_id),
                )

    def log_process_ref(
        self, run_id: str, stream: str, path: str, tail: str, size_bytes: int
    ) -> None:
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO process_log_refs
                    (ref_id, run_id, stream, path, tail, size_bytes, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (new_id("log-"), run_id, stream, path, tail, size_bytes, time.time()),
                )

    def log_error(
        self,
        *,
        error_type: str,
        message: str,
        traceback: str = "",
        request_id: str = "",
        job_id: str = "",
        node_id: str = "",
        action_id: str = "",
    ) -> str:
        eid = new_id("err-")
        with self._tx() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cluster_errors
                    (error_id, ts, error_type, message, traceback, request_id, job_id, node_id, action_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (eid, time.time(), error_type, message, traceback,
                     request_id, job_id, node_id, action_id),
                )
        return eid

    def query_logs(
        self,
        *,
        job_id: str = "",
        node_id: str = "",
        limit: int = 50,
    ) -> Dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.list_events(job_id=job_id, node_id=node_id, limit=limit)],
            "process_runs": [],
            "process_log_refs": [],
            "errors": [],
            "agent_actions": [],
        }


def open_store(database_url: str = "") -> ClusterStore:
    """Open the configured store backend."""
    if database_url:
        try:
            store = PostgresClusterStore(database_url)
            store.ensure_schema()
            return store
        except ImportError:
            _log.warning("psycopg not installed; falling back to in-memory store")
        except Exception as exc:
            _log.error("Postgres store failed (%s); falling back to in-memory store", exc)
    return MemoryClusterStore()
