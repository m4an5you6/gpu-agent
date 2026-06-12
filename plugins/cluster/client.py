"""HTTP client for cluster control-plane API."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx

from plugins.cluster.config import ClusterConfig


class ClusterClient:
    def __init__(self, cfg: ClusterConfig, *, base_url: Optional[str] = None, timeout: float = 30.0):
        self.cfg = cfg
        self.base_url = (base_url or cfg.master_url).rstrip("/")
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        secret = self.cfg.secret
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        return headers

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.request(method, url, headers=self._headers(), json=body)
            resp.raise_for_status()
            if resp.content:
                return resp.json()
            return {"success": True}

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def register(
        self,
        *,
        node_id: str,
        advertised_addr: str,
        gpus: list,
        agent_version: str,
        config_hash: str,
    ) -> Dict[str, Any]:
        return self._request("POST", "/api/nodes/register", {
            "node_id": node_id,
            "advertised_addr": advertised_addr,
            "gpus": gpus,
            "agent_version": agent_version,
            "config_hash": config_hash,
        })

    def heartbeat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload["node_id"]
        return self._request("POST", f"/api/nodes/{node_id}/heartbeat", payload)

    def status(self) -> Dict[str, Any]:
        return self._request("GET", "/api/status")

    def validate_config(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/validate", spec)

    def submit_job(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/api/jobs/submit", spec)

    def job_status(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/jobs/{job_id}")

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/jobs/{job_id}/stop", {})

    def logs(self, **params: Any) -> Dict[str, Any]:
        q = "&".join(f"{k}={v}" for k, v in params.items() if v)
        path = f"/api/logs?{q}" if q else "/api/logs"
        return self._request("GET", path)

    def node_action(self, node_id: str, action: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/nodes/{node_id}/action", {"action": action})

    def ack_assignment(
        self, assignment_id: str, node_id: str, job_generation: int, state: str
    ) -> Dict[str, Any]:
        return self._request("POST", f"/api/assignments/{assignment_id}/ack", {
            "node_id": node_id,
            "job_generation": job_generation,
            "state": state,
        })

    def report_outcome(
        self, job_id: str, *, success: bool, summary: str = "", node_id: str = ""
    ) -> Dict[str, Any]:
        return self._request("POST", f"/api/jobs/{job_id}/outcome", {
            "success": success,
            "summary": summary,
            "node_id": node_id,
        })
