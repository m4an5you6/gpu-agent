"""Threading HTTP server for cluster master control plane."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from plugins.cluster.config import ClusterConfig
from plugins.cluster.controller import ClusterController
from plugins.cluster.cluster_logging import ClusterLogger
from plugins.cluster.models import HeartbeatPayload, GpuInfo

_log = logging.getLogger(__name__)


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: Dict[str, Any]) -> None:
    raw = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _auth_ok(cfg: ClusterConfig, handler: BaseHTTPRequestHandler) -> bool:
    secret = cfg.secret
    if not secret:
        return True
    auth = handler.headers.get("Authorization", "")
    if auth == f"Bearer {secret}":
        return True
    return False


class ClusterHTTPHandler(BaseHTTPRequestHandler):
    controller: ClusterController = None  # type: ignore[assignment]
    cfg: ClusterConfig = None  # type: ignore[assignment]
    logger: ClusterLogger = None  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:
        _log.debug(fmt, *args)

    def _handle(self, method: str) -> None:
        request_id = self.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health" and method == "GET":
            _json_response(self, 200, {"ok": True, "role": "master"})
            self.logger.log_http(request_id=request_id, method=method, path=path, status=200)
            return

        if not _auth_ok(self.cfg, self):
            _json_response(self, 401, {"success": False, "error": "unauthorized"})
            self.logger.log_http(request_id=request_id, method=method, path=path, status=401)
            return

        try:
            status, body = self._dispatch(method, path, parse_qs(parsed.query), _read_json(self))
            _json_response(self, status, body)
            self.logger.log_http(
                request_id=request_id,
                method=method,
                path=path,
                status=status,
                job_id=str(body.get("job_id") or body.get("job", {}).get("job_id") or ""),
            )
        except json.JSONDecodeError:
            _json_response(self, 400, {"success": False, "error": "invalid json"})
            self.logger.log_http(request_id=request_id, method=method, path=path, status=400)
        except Exception as exc:
            self.logger.log_error(error_type="http_handler", message=str(exc), request_id=request_id)
            _json_response(self, 500, {"success": False, "error": str(exc)})
            self.logger.log_http(request_id=request_id, method=method, path=path, status=500)

    def _dispatch(
        self, method: str, path: str, query: Dict[str, list], body: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        ctrl = self.controller

        if path == "/api/status" and method == "GET":
            return 200, ctrl.status()

        if path == "/api/validate" and method == "POST":
            result = ctrl.validate_config(body)
            return 200, result.to_dict()

        if path == "/api/jobs/submit" and method == "POST":
            return 200, ctrl.submit_job(body, request_id=body.get("request_id", ""))

        if path.startswith("/api/jobs/") and method == "GET":
            job_id = path.split("/")[-1]
            return 200, ctrl.job_status(job_id)

        if path.startswith("/api/jobs/") and path.endswith("/stop") and method == "POST":
            job_id = path.split("/")[-2]
            return 200, ctrl.stop_job(job_id)

        if path.startswith("/api/jobs/") and path.endswith("/outcome") and method == "POST":
            job_id = path.split("/")[-2]
            ctrl.report_job_outcome(
                job_id,
                success=bool(body.get("success")),
                summary=str(body.get("summary") or ""),
                node_id=str(body.get("node_id") or ""),
            )
            return 200, {"success": True}

        if path == "/api/nodes/register" and method == "POST":
            node = ctrl.register_node(
                node_id=str(body.get("node_id") or ""),
                advertised_addr=str(body.get("advertised_addr") or ""),
                gpus=body.get("gpus") or [],
                agent_version=str(body.get("agent_version") or ""),
                config_hash=str(body.get("config_hash") or ""),
            )
            return 200, {"success": True, "node": node.to_dict()}

        if path.startswith("/api/nodes/") and path.endswith("/heartbeat") and method == "POST":
            node_id = path.split("/")[-2]
            gpus = [
                GpuInfo(index=int(g.get("index", i)), name=str(g.get("name", "")), memory_mb=int(g.get("memory_mb", 0)))
                for i, g in enumerate(body.get("gpus") or [])
            ]
            hb = HeartbeatPayload(
                node_id=node_id,
                state=body.get("state") or "ready",
                gpus=gpus,
                config_hash=str(body.get("config_hash") or ""),
                running_job_id=body.get("running_job_id"),
                metrics=body.get("metrics") or {},
            )
            return 200, ctrl.heartbeat(hb)

        if path.startswith("/api/nodes/") and path.endswith("/action") and method == "POST":
            node_id = path.split("/")[-2]
            return 200, ctrl.node_action(node_id, str(body.get("action") or ""))

        if path.startswith("/api/assignments/") and path.endswith("/ack") and method == "POST":
            assignment_id = path.split("/")[-2]
            ok = ctrl.ack_assignment(
                assignment_id,
                str(body.get("node_id") or ""),
                int(body.get("job_generation") or 0),
                str(body.get("state") or "accepted"),
            )
            return 200, {"success": ok}

        if path == "/api/logs" and method == "GET":
            job_id = (query.get("job_id") or [""])[0]
            node_id = (query.get("node_id") or [""])[0]
            limit = int((query.get("limit") or ["50"])[0])
            logs = ctrl.store.query_logs(job_id=job_id, node_id=node_id, limit=limit)
            return 200, {"success": True, "logs": logs}

        if path == "/api/events/drain" and method == "GET":
            queued = [e.to_dict() for e in ctrl.events.drain_queue()]
            return 200, {"success": True, "events": queued}

        return 404, {"success": False, "error": f"not found: {method} {path}"}

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")


class ClusterHTTPServer:
    def __init__(
        self,
        cfg: ClusterConfig,
        controller: ClusterController,
        logger: ClusterLogger,
    ) -> None:
        self.cfg = cfg
        self.controller = controller
        self.logger = logger
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._sweep_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self, *, block: bool = True) -> None:
        handler_cls = ClusterHTTPHandler
        handler_cls.controller = self.controller
        handler_cls.cfg = self.cfg
        handler_cls.logger = self.logger

        self._httpd = ThreadingHTTPServer((self.cfg.bind_host, self.cfg.bind_port), handler_cls)
        self.controller.startup()
        _log.info(
            "cluster master listening on %s:%s",
            self.cfg.bind_host,
            self.cfg.bind_port,
        )

        self._sweep_thread = threading.Thread(target=self._sweep_loop, daemon=True)
        self._sweep_thread.start()

        if block:
            try:
                self._httpd.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            t.start()

    def _sweep_loop(self) -> None:
        while not self._stop.wait(self.cfg.heartbeat_interval_sec):
            try:
                self.controller.sweep_stale_nodes()
            except Exception as exc:
                self.logger.log_error(error_type="sweep", message=str(exc))

    def stop(self) -> None:
        self._stop.set()
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
