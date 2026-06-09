"""Phase 6 training dry-run and job store tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

from gpucloud_cli.gpucloud_config import prepare_gpucloud_config
from gpucloud_cli.gpucloud_jobs import TrainingJob, get_job, init_jobs_db, list_recent_jobs, save_job
from gpucloud_cli.gpucloud_train import (
    plan_training_job,
    run_train_start,
    validate_training_command,
)


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


def test_validate_megatron_torchrun_only():
    assert validate_training_command("torchrun --nproc_per_node=2 pretrain_gpt.py") is None
    assert validate_training_command("python train.py") is not None


def test_plan_dry_run(tmp_path, monkeypatch):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    prepared = prepare_gpucloud_config(path)
    plan = plan_training_job(prepared)
    assert plan["ok"]
    assert "torchrun" in plan["launch_command"]
    assert "pretrain_gpt.py" in plan["launch_command"]
    assert "my-dataset" in plan["launch_command"]
    assert plan["log_path"].endswith(".log")
    assert "checkpoints" in plan["checkpoint_path"]


def test_run_train_start_dry_run(tmp_path, monkeypatch):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out = run_train_start(
        dry_run=True,
        allow_discover_without_goal=True,
    )
    assert out["ok"]
    assert out.get("dry_run") is True
    assert "torchrun" in out["launch_command"]
    assert "pretrain_gpt.py" in out["launch_command"]


def test_job_persistence(tmp_path, monkeypatch):
    db = tmp_path / "jobs.db"
    monkeypatch.setattr(
        "gpucloud_cli.gpucloud_jobs.jobs_db_path",
        lambda: db,
    )
    init_jobs_db()
    job = TrainingJob(
        job_id="train-test-1",
        cluster="prod",
        status="running",
        launch_command="torchrun ...",
        workdir="~/gpucloud",
        log_path="~/gpucloud/logs/train-test-1.log",
        checkpoint_path="~/gpucloud/checkpoints/llama",
    )
    save_job(job)
    loaded = get_job("train-test-1")
    assert loaded and loaded.status == "running"
    recent = list_recent_jobs(5)
    assert recent[0].job_id == "train-test-1"
