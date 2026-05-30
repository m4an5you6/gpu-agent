"""Phase 8 vLLM inference service tests."""

from __future__ import annotations

import textwrap

from hermes_cli.gpucloud_config import prepare_gpucloud_config
from hermes_cli.gpucloud_inference import (
    build_health_command,
    plan_inference_service,
    run_infer_health,
    run_infer_start,
    run_infer_stop,
)
from hermes_cli.gpucloud_jobs import get_job, init_jobs_db
from hermes_cli.gpucloud_ssh import SSHResult


MINIMAL = textwrap.dedent(
    """
    clusters:
      - name: prod
        nodes:
          - host: 10.0.0.1
            port: 22
            user: ubuntu
            ssh_key: ~/.ssh/id_rsa
    dataset_name: my-dataset
    model_name: llama-3-8b
    """
).strip()


def test_plan_vllm_dry_run_defaults(tmp_path):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    prepared = prepare_gpucloud_config(path)

    plan = plan_inference_service(prepared)

    assert plan["ok"]
    assert plan["engine"] == "vllm"
    assert plan["port"] == 8000
    assert plan["tensor_parallel"] == 1
    assert "vllm serve" in plan["launch_command"]
    assert "llama-3-8b" in plan["model_path"]
    assert plan["service_url"] == "http://10.0.0.1:8000"
    assert plan["log_path"].endswith(".log")


def test_unsupported_inference_engine_is_rejected(tmp_path):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(
        MINIMAL + "\ninference:\n  engine: tgi\n",
        encoding="utf-8",
    )
    prepared = prepare_gpucloud_config(path)

    plan = plan_inference_service(prepared)

    assert not plan["ok"]
    assert "inference.engine=vllm only" in plan["error"]


def test_run_infer_start_defaults_to_dry_run(tmp_path, monkeypatch):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    out = run_infer_start(
        dry_run=True,
        allow_discover_without_goal=True,
    )

    assert out["ok"]
    assert out["dry_run"] is True
    assert "vllm serve" in out["launch_command"]


def test_start_health_and_stop_vllm_service(tmp_path, monkeypatch):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    db = tmp_path / "jobs.db"
    monkeypatch.setattr("hermes_cli.gpucloud_jobs.jobs_db_path", lambda: db)
    init_jobs_db()

    calls = []

    def fake_ssh(**kwargs):
        calls.append(kwargs["remote_command"])
        command = kwargs["remote_command"]
        if "nohup bash -lc" in command:
            return SSHResult(ok=True, exit_code=0, stdout="4242\n", stderr="")
        if "/health" in command:
            return SSHResult(ok=True, exit_code=0, stdout="200\nOK\n", stderr="")
        if "kill 4242" in command:
            return SSHResult(ok=True, exit_code=0, stdout="stopped 4242\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("hermes_cli.gpucloud_inference.run_ssh_command", fake_ssh)

    started = run_infer_start(
        config_file=str(path),
        dry_run=False,
        confirm_execute=True,
    )

    assert started["ok"]
    job_id = started["job"]["job_id"]
    job = get_job(job_id)
    assert job is not None
    assert job.job_type == "inference"
    assert job.status == "running"
    assert job.remote_pid == "4242"
    assert job.port == 8000

    health = run_infer_health(
        job_id,
        config_file=str(path),
    )
    assert health["ok"]
    assert health["healthy"] is True

    stopped = run_infer_stop(
        job_id,
        config_file=str(path),
        dry_run=False,
        confirm_stop=True,
    )
    assert stopped["ok"]
    assert stopped["job"]["status"] == "stopped"
    assert any("vllm serve" in command for command in calls)


def test_health_command_uses_local_vllm_endpoint():
    command = build_health_command(9000)
    assert "127.0.0.1:9000/health" in command
    assert command.startswith("python -c ")
