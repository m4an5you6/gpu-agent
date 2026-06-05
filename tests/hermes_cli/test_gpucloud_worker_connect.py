from __future__ import annotations

import asyncio
import json

import yaml

from hermes_cli import gpucloud_worker_connect as connect


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, text: str):
        self.sent.append(json.loads(text))


def test_register_payload_reports_worker_goal_capability(monkeypatch):
    monkeypatch.setattr(connect, "probe_local_gpu", lambda: {"available": True, "gpu_count": 1})
    client = connect.WorkerControlClient(server_url="ws://main/ws", node_id="node-1", cert_fingerprint="fp")

    payload = client.register_payload()

    assert payload["node_id"] == "node-1"
    assert payload["cert_fingerprint"] == "fp"
    assert "gpucloud.worker_goal" in payload["capabilities"]


def test_task_submit_writes_yaml_and_runs_worker_goal(tmp_path):
    websocket = FakeWebSocket()
    calls = []
    results = [
        {"ok": True, "stage": "training_running", "status": "running", "logs": {"train_tail": "step 1"}},
        {
            "ok": True,
            "stage": "completed",
            "status": "completed",
            "logs": {"train_tail": "done"},
            "train": {"status": {"job": {"exit_code": 0}}},
        },
    ]

    def fake_goal_runner(**kwargs):
        calls.append(kwargs)
        return results.pop(0)

    client = connect.WorkerControlClient(
        server_url="ws://main/ws",
        node_id="node-1",
        task_dir=tmp_path,
        poll_interval_sec=0.1,
        goal_runner=fake_goal_runner,
    )
    client.websocket = websocket
    message = {
        "type": "task.submit",
        "node_id": "node-1",
        "task_id": "task-1",
        "payload": {
            "mode": "train_and_infer",
            "goal": "train tiny gpt",
            "task": {
                "job_id": "job-1",
                "framework": "megatron-lm",
                "role": "worker",
                "distributed": {
                    "nnodes": 1,
                    "node_rank": 0,
                    "nproc_per_node": 1,
                    "master_addr": "127.0.0.1",
                    "master_port": 29500,
                },
                "runtime": {"workdir": "/tmp/job", "megatron_lm_dir": "/tmp/Megatron-LM"},
                "training": {
                    "entrypoint": "pretrain_gpt.py",
                    "data_path": "/tmp/data",
                    "checkpoint_dir": "/tmp/ckpt",
                    "log_dir": "/tmp/logs",
                },
            },
        },
    }

    asyncio.run(client.run_task(message))

    task_file = tmp_path / "task-1.yaml"
    assert yaml.safe_load(task_file.read_text(encoding="utf-8"))["job_id"] == "job-1"
    assert calls[0]["task_file"] == task_file
    assert calls[0]["mode"] == "train_and_infer"
    assert [msg["type"] for msg in websocket.sent] == [
        "task.accepted",
        "task.status",
        "task.logs",
        "task.status",
        "task.logs",
        "task.exit",
    ]
    assert websocket.sent[-1]["payload"]["exit_code"] == 0


def test_cancel_task_stops_local_job_only():
    websocket = FakeWebSocket()
    calls = []

    def fake_stopper(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "stop_result": "terminated"}

    client = connect.WorkerControlClient(
        server_url="ws://main/ws",
        node_id="node-1",
        stopper=fake_stopper,
    )
    client.websocket = websocket
    client.active_jobs["task-1"] = "job-1"

    asyncio.run(client.cancel_task("task-1", {"reason": "user requested"}))

    assert calls == [{"job_id": "job-1", "confirm_stop": True}]
    assert websocket.sent[-1]["type"] == "task.status"
    assert websocket.sent[-1]["payload"]["status"] == "canceled"


def test_build_ssl_context_returns_none_for_plain_ws():
    assert connect.build_ssl_context("ws://localhost/api/v0/workers/ws") is None
