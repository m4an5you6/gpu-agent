"""Structured logging for cluster control plane."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from plugins.cluster.config import ClusterConfig
from plugins.cluster.models import AgentActionLog, ProcessRunLog, new_id
from plugins.cluster.store import ClusterStore

_log = logging.getLogger(__name__)

_SECRET_PATTERN = re.compile(
    r"(api[_-]?key|token|secret|password|credential)",
    re.IGNORECASE,
)


def redact_env(env: Dict[str, str]) -> List[str]:
    """Return env var keys only — never log secret values."""
    return sorted(k for k in env.keys())


def tail_lines(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return "".join(deque(fh, maxlen=max_lines))
    except OSError:
        return ""


class ClusterLogger:
    """Writes audit records to JSONL, Postgres tables, and Python logging."""

    def __init__(self, cfg: ClusterConfig, store: ClusterStore) -> None:
        self.cfg = cfg
        self.store = store
        self._lock = threading.Lock()
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        level = str(cfg.logging.get("level", "info")).upper()
        _log.setLevel(getattr(logging, level, logging.INFO))

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        if not self.cfg.logging.get("jsonl", True):
            return
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            with self.cfg.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def log_http(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        status: int,
        job_id: str = "",
        node_id: str = "",
    ) -> None:
        rec = {
            "ts": time.time(),
            "kind": "http",
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "job_id": job_id,
            "node_id": node_id,
        }
        self._append_jsonl(rec)
        _log.info("cluster http %s %s -> %s", method, path, status)

    def log_tool_action(
        self,
        *,
        tool_name: str,
        tool_args: Dict[str, Any],
        decision: str,
        result_summary: str,
        session_id: str = "",
        turn_id: str = "",
    ) -> None:
        action = AgentActionLog(
            action_id=new_id("act-"),
            tool_name=tool_name,
            tool_args=tool_args,
            decision=decision,
            result_summary=result_summary,
            session_id=session_id,
            turn_id=turn_id,
        )
        self.store.log_agent_action(action)
        self._append_jsonl({"ts": action.ts, "kind": "agent_action", **action.to_dict()})

    def start_process(
        self,
        *,
        job_id: str,
        node_id: str,
        command: List[str],
        cwd: str,
        env: Dict[str, str],
        pid: Optional[int] = None,
    ) -> str:
        run = ProcessRunLog(
            run_id=new_id("run-"),
            job_id=job_id,
            node_id=node_id,
            command=command,
            cwd=cwd,
            env_keys=redact_env(env),
            pid=pid,
        )
        self.store.log_process_run(run)
        self._append_jsonl({"ts": run.ts_start, "kind": "process_start", **run.to_dict()})
        return run.run_id

    def finish_process(
        self,
        run_id: str,
        *,
        exit_code: int,
        stdout_path: Optional[Path] = None,
        stderr_path: Optional[Path] = None,
    ) -> None:
        ts_end = time.time()
        self.store.update_process_run(run_id, exit_code=exit_code, ts_end=ts_end)
        stdout_tail = ""
        stderr_tail = ""
        max_out = int(self.cfg.logging.get("capture_stdout_tail_lines", 200))
        max_err = int(self.cfg.logging.get("capture_stderr_tail_lines", 200))
        if stdout_path:
            stdout_tail = tail_lines(stdout_path, max_out)
            self.store.log_process_ref(
                run_id, "stdout", str(stdout_path), stdout_tail,
                stdout_path.stat().st_size if stdout_path.exists() else 0,
            )
        if stderr_path:
            stderr_tail = tail_lines(stderr_path, max_err)
            self.store.log_process_ref(
                run_id, "stderr", str(stderr_path), stderr_tail,
                stderr_path.stat().st_size if stderr_path.exists() else 0,
            )
        self._append_jsonl({
            "ts": ts_end,
            "kind": "process_end",
            "run_id": run_id,
            "exit_code": exit_code,
            "stdout_tail": stdout_tail[-2000:],
            "stderr_tail": stderr_tail[-2000:],
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
        eid = self.store.log_error(
            error_type=error_type,
            message=message,
            traceback=traceback,
            request_id=request_id,
            job_id=job_id,
            node_id=node_id,
            action_id=action_id,
        )
        self._append_jsonl({
            "ts": time.time(),
            "kind": "error",
            "error_id": eid,
            "error_type": error_type,
            "message": message,
            "request_id": request_id,
            "job_id": job_id,
            "node_id": node_id,
        })
        _log.error("cluster %s: %s", error_type, message)
        return eid
