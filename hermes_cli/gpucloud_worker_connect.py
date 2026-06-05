"""WSS control-plane client for GPUCLOUD worker agents."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import ssl
from typing import Any, Callable, Dict, Optional
from uuid import uuid4
from urllib.parse import urlparse

import yaml

from hermes_constants import get_hermes_home
from hermes_cli import __version__
from hermes_cli.gpucloud_probe import probe_local_gpu
from hermes_cli.gpucloud_worker import run_worker_logs, run_worker_stop
from hermes_cli.gpucloud_worker_goal import run_worker_goal_run


TERMINAL_STAGES = {"training_failed", "conversion_failed", "completed"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    safe = "".join(ch for ch in str(value) if ch.isalnum() or ch in "-_")
    return safe or "worker-task"


def default_worker_task_dir() -> Path:
    return get_hermes_home() / "gpucloud" / "wss_worker_tasks"


def build_ssl_context(
    server_url: str,
    *,
    ca_path: Optional[str] = None,
    cert_path: Optional[str] = None,
    key_path: Optional[str] = None,
) -> Optional[ssl.SSLContext]:
    scheme = urlparse(server_url).scheme.lower()
    if scheme != "wss":
        return None
    context = ssl.create_default_context(cafile=os.path.expanduser(ca_path)) if ca_path else ssl.create_default_context()
    if cert_path:
        context.load_cert_chain(
            certfile=os.path.expanduser(cert_path),
            keyfile=os.path.expanduser(key_path) if key_path else None,
        )
    return context


def extract_exit_code(result: Dict[str, Any]) -> Optional[int]:
    candidates = [
        result.get("exit_code"),
        ((result.get("train") or {}).get("status") or {}).get("job", {}).get("exit_code"),
        ((result.get("train") or {}).get("start") or {}).get("job", {}).get("exit_code"),
        (result.get("conversion") or {}).get("exit_code"),
        (result.get("inference") or {}).get("exit_code"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


class WorkerControlClient:
    def __init__(
        self,
        *,
        server_url: str,
        node_id: str,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        ca_path: Optional[str] = None,
        cert_fingerprint: Optional[str] = None,
        task_dir: Optional[Path] = None,
        poll_interval_sec: float = 5.0,
        heartbeat_sec: float = 10.0,
        goal_runner: Callable[..., Dict[str, Any]] = run_worker_goal_run,
        stopper: Callable[..., Dict[str, Any]] = run_worker_stop,
        logs_reader: Callable[..., Dict[str, Any]] = run_worker_logs,
    ) -> None:
        self.server_url = server_url
        self.node_id = node_id
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path
        self.cert_fingerprint = cert_fingerprint
        self.task_dir = task_dir or default_worker_task_dir()
        self.poll_interval_sec = max(0.1, float(poll_interval_sec))
        self.heartbeat_sec = max(1.0, float(heartbeat_sec))
        self.goal_runner = goal_runner
        self.stopper = stopper
        self.logs_reader = logs_reader
        self.websocket: Any = None
        self.active_jobs: Dict[str, str] = {}

    def message(
        self,
        message_type: str,
        *,
        task_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "type": message_type,
            "message_id": str(uuid4()),
            "node_id": self.node_id,
            "task_id": task_id,
            "timestamp": _utc_now(),
            "payload": payload or {},
        }

    def register_payload(self) -> Dict[str, Any]:
        gpu = probe_local_gpu()
        return {
            "node_id": self.node_id,
            "hostname": socket.gethostname(),
            "gpu_info": gpu,
            "agent_version": __version__,
            "capabilities": ["gpucloud.worker_goal", "gpucloud.worker_status", "gpucloud.worker_logs", "gpucloud.worker_stop"],
            "cert_fingerprint": self.cert_fingerprint,
        }

    async def run_forever(self, *, reconnect_sec: float = 5.0) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(max(1.0, reconnect_sec))

    async def run_once(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("websockets package is required for gpucloud worker connect") from exc

        ssl_context = build_ssl_context(
            self.server_url,
            ca_path=self.ca_path,
            cert_path=self.cert_path,
            key_path=self.key_path,
        )
        async with websockets.connect(self.server_url, ssl=ssl_context, ping_interval=20) as websocket:
            self.websocket = websocket
            await self.send_message("worker.register", payload=self.register_payload())
            heartbeat = asyncio.create_task(self._heartbeat_loop())
            try:
                async for raw in websocket:
                    await self.handle_raw_message(raw)
            finally:
                heartbeat.cancel()
                self.websocket = None

    async def handle_raw_message(self, raw: str | bytes | Dict[str, Any]) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        message = json.loads(raw) if isinstance(raw, str) else raw
        message_type = str(message.get("type") or "")
        task_id = str(message.get("task_id") or "")
        if message.get("node_id") not in {None, "", self.node_id}:
            return
        if message_type == "task.submit":
            asyncio.create_task(self.run_task(message))
        elif message_type == "task.cancel" and task_id:
            await self.cancel_task(task_id, message.get("payload") or {})
        elif message_type == "task.logs_request" and task_id:
            await self.send_task_logs(task_id)
        elif message_type == "task.status_request" and task_id:
            await self.send_message("task.status", task_id=task_id, payload={"status": "known" if task_id in self.active_jobs else "unknown"})

    async def run_task(self, message: Dict[str, Any]) -> None:
        task_id = str(message.get("task_id") or uuid4())
        payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
        task_payload = payload.get("task") if isinstance(payload.get("task"), dict) else payload
        goal = str(payload.get("goal") or "")
        mode = payload.get("mode")
        job_id = str(task_payload.get("job_id") or task_id)
        self.active_jobs[task_id] = job_id
        task_file = self.write_task_file(task_id, task_payload)

        await self.send_message("task.accepted", task_id=task_id, payload={"job_id": job_id, "task_file": str(task_file)})
        while True:
            try:
                result = self.goal_runner(task_file=task_file, goal=goal, mode=mode)
            except Exception as exc:
                await self.send_message("task.error", task_id=task_id, payload={"error": str(exc), "job_id": job_id})
                await self.send_message("task.exit", task_id=task_id, payload={"exit_code": 1, "last_error": str(exc), "job_id": job_id})
                return

            stage = str(result.get("stage") or "")
            status = str(result.get("status") or "")
            await self.send_message(
                "task.status",
                task_id=task_id,
                payload={
                    "job_id": job_id,
                    "stage": stage,
                    "status": status,
                    "ok": result.get("ok"),
                    "last_error": result.get("last_error"),
                    "next_action": result.get("next_action"),
                },
            )
            logs = result.get("logs") if isinstance(result.get("logs"), dict) else {}
            if logs:
                await self.send_message("task.logs", task_id=task_id, payload={"job_id": job_id, "logs": logs})
            if stage in TERMINAL_STAGES or status in {"completed", "failed"}:
                exit_code = extract_exit_code(result)
                if exit_code is None and (not result.get("ok") or status == "failed"):
                    exit_code = 1
                await self.send_message(
                    "task.exit",
                    task_id=task_id,
                    payload={
                        "job_id": job_id,
                        "stage": stage,
                        "status": status,
                        "exit_code": exit_code,
                        "last_error": result.get("last_error"),
                    },
                )
                return
            await asyncio.sleep(self.poll_interval_sec)

    async def cancel_task(self, task_id: str, payload: Dict[str, Any]) -> None:
        job_id = self.active_jobs.get(task_id) or str(payload.get("job_id") or task_id)
        result = self.stopper(job_id=job_id, confirm_stop=True)
        await self.send_message("task.status", task_id=task_id, payload={"job_id": job_id, "status": "canceled", "stop": result})

    async def send_task_logs(self, task_id: str) -> None:
        job_id = self.active_jobs.get(task_id)
        if not job_id:
            await self.send_message("task.logs", task_id=task_id, payload={"error": "unknown task"})
            return
        result = self.logs_reader(job_id=job_id, lines=50)
        await self.send_message("task.logs", task_id=task_id, payload={"job_id": job_id, "logs": result})

    def write_task_file(self, task_id: str, task_payload: Dict[str, Any]) -> Path:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        path = self.task_dir / f"{_safe_id(task_id)}.yaml"
        path.write_text(yaml.safe_dump(task_payload, sort_keys=False), encoding="utf-8")
        return path

    async def send_message(
        self,
        message_type: str,
        *,
        task_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.websocket is None:
            raise RuntimeError("worker control websocket is not connected")
        await self.websocket.send(json.dumps(self.message(message_type, task_id=task_id, payload=payload), ensure_ascii=False))

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_sec)
            await self.send_message(
                "worker.heartbeat",
                payload={"active_tasks": sorted(self.active_jobs), "active_jobs": dict(self.active_jobs)},
            )


def run_worker_connect(
    *,
    server_url: str,
    node_id: str,
    cert_path: Optional[str] = None,
    key_path: Optional[str] = None,
    ca_path: Optional[str] = None,
    cert_fingerprint: Optional[str] = None,
    task_dir: Optional[str] = None,
    poll_interval_sec: float = 5.0,
    heartbeat_sec: float = 10.0,
    reconnect_sec: float = 5.0,
) -> int:
    client = WorkerControlClient(
        server_url=server_url,
        node_id=node_id,
        cert_path=cert_path,
        key_path=key_path,
        ca_path=ca_path,
        cert_fingerprint=cert_fingerprint,
        task_dir=Path(task_dir).expanduser() if task_dir else None,
        poll_interval_sec=poll_interval_sec,
        heartbeat_sec=heartbeat_sec,
    )
    try:
        asyncio.run(client.run_forever(reconnect_sec=reconnect_sec))
    except KeyboardInterrupt:
        return 0
    return 0


__all__ = [
    "WorkerControlClient",
    "build_ssl_context",
    "default_worker_task_dir",
    "extract_exit_code",
    "run_worker_connect",
]
