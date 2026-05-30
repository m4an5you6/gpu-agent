"""GPUCLOUD training job persistence (phase 6)."""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

JOB_STATUSES = frozenset(
    {"pending", "running", "failed", "stopped", "completed"}
)


@dataclass
class TrainingJob:
    job_id: str
    job_type: str = "train"
    cluster: str = ""
    status: str = "pending"
    launch_command: str = ""
    workdir: str = ""
    log_path: str = ""
    checkpoint_path: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    last_error: Optional[str] = None
    node_index: int = 0
    host: str = ""
    remote_pid: Optional[str] = None
    dataset: str = ""
    model: str = ""
    port: Optional[int] = None
    service_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TrainingJob":
        data = dict(row)
        return cls(
            job_id=data["job_id"],
            job_type=data.get("job_type") or "train",
            cluster=data.get("cluster") or "",
            status=data.get("status") or "pending",
            launch_command=data.get("launch_command") or "",
            workdir=data.get("workdir") or "",
            log_path=data.get("log_path") or "",
            checkpoint_path=data.get("checkpoint_path") or "",
            created_at=float(data.get("created_at") or 0),
            updated_at=float(data.get("updated_at") or 0),
            last_error=data.get("last_error"),
            node_index=int(data.get("node_index") or 0),
            host=data.get("host") or "",
            remote_pid=data.get("remote_pid"),
            dataset=data.get("dataset") or "",
            model=data.get("model") or "",
            port=int(data["port"]) if data.get("port") is not None else None,
            service_url=data.get("service_url") or "",
        )


def jobs_db_path() -> Path:
    path = get_hermes_home() / "gpucloud"
    path.mkdir(parents=True, exist_ok=True)
    return path / "jobs.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(jobs_db_path()), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_jobs_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL DEFAULT 'train',
                cluster TEXT,
                status TEXT NOT NULL,
                launch_command TEXT,
                workdir TEXT,
                log_path TEXT,
                checkpoint_path TEXT,
                created_at REAL,
                updated_at REAL,
                last_error TEXT,
                node_index INTEGER DEFAULT 0,
                host TEXT,
                remote_pid TEXT,
                dataset TEXT,
                model TEXT,
                port INTEGER,
                service_url TEXT
            )
            """
        )
        existing = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(training_jobs)").fetchall()
        }
        if "port" not in existing:
            conn.execute("ALTER TABLE training_jobs ADD COLUMN port INTEGER")
        if "service_url" not in existing:
            conn.execute("ALTER TABLE training_jobs ADD COLUMN service_url TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_jobs_updated "
            "ON training_jobs(updated_at DESC)"
        )
        conn.commit()


def save_job(job: TrainingJob) -> None:
    init_jobs_db()
    now = time.time()
    if not job.created_at:
        job.created_at = now
    job.updated_at = now
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO training_jobs (
                job_id, job_type, cluster, status, launch_command,
                workdir, log_path, checkpoint_path, created_at, updated_at,
                last_error, node_index, host, remote_pid, dataset, model,
                port, service_url
            ) VALUES (
                :job_id, :job_type, :cluster, :status, :launch_command,
                :workdir, :log_path, :checkpoint_path, :created_at, :updated_at,
                :last_error, :node_index, :host, :remote_pid, :dataset, :model,
                :port, :service_url
            )
            """,
            job.to_dict(),
        )
        conn.commit()


def get_job(job_id: str) -> Optional[TrainingJob]:
    init_jobs_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM training_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    return TrainingJob.from_row(row) if row else None


def update_job_status(
    job_id: str,
    status: str,
    *,
    last_error: Optional[str] = None,
    remote_pid: Optional[str] = None,
) -> Optional[TrainingJob]:
    if status not in JOB_STATUSES:
        raise ValueError(f"invalid status: {status}")
    job = get_job(job_id)
    if not job:
        return None
    job.status = status
    job.updated_at = time.time()
    if last_error is not None:
        job.last_error = last_error
    if remote_pid is not None:
        job.remote_pid = remote_pid
    save_job(job)
    return job


def list_recent_jobs(limit: int = 10, *, job_type: Optional[str] = None) -> List[TrainingJob]:
    init_jobs_db()
    with _connect() as conn:
        if job_type:
            rows = conn.execute(
                "SELECT * FROM training_jobs WHERE job_type = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (job_type, max(1, int(limit))),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM training_jobs ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
    return [TrainingJob.from_row(r) for r in rows]


def new_job_id(prefix: str = "train") -> str:
    ts = int(time.time())
    safe_prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in "-_") or "job"
    return f"{safe_prefix}-{ts}-{uuid.uuid4().hex[:8]}"
