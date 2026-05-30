"""Distributed GPUCLOUD worker task parsing tests."""

from __future__ import annotations

import textwrap

from hermes_cli.gpucloud_worker_task import (
    WorkerTaskError,
    load_worker_task,
    merge_worker_task_defaults,
    validate_worker_task,
)


TASK = textwrap.dedent(
    """
    job_id: gpt-pretrain-001
    framework: megatron-lm
    role: worker
    distributed:
      nnodes: 4
      nproc_per_node: 1
      node_rank: 2
      master_addr: 10.0.0.10
      master_port: 29500
    runtime:
      workdir: /tmp/gpucloud/job
      megatron_lm_dir: /opt/Megatron-LM
      env:
        NCCL_DEBUG: INFO
    training:
      data_path: /data/tokens
      checkpoint_dir: /data/checkpoints/job
      log_dir: /data/logs
      extra_args:
        - --tensor-model-parallel-size=1
        - --pipeline-model-parallel-size=1
    preflight:
      require_gpu_count: 1
      min_vram_gb: 16
      heterogeneous_policy: warn
    """
).strip()


def test_load_worker_task(tmp_path):
    path = tmp_path / "gpucloud-worker-task.yaml"
    path.write_text(TASK, encoding="utf-8")

    task = load_worker_task(path)

    assert task.job_id == "gpt-pretrain-001"
    assert task.nnodes == 4
    assert task.node_rank == 2
    assert task.nproc_per_node == 1
    assert task.master_addr == "10.0.0.10"
    assert task.master_port == 29500
    assert task.preflight["heterogeneous_policy"] == "warn"


def test_worker_task_defaults_and_required_rank_fields():
    data = merge_worker_task_defaults(
        {
            "job_id": "job-1",
            "distributed": {
                "nnodes": 2,
                "node_rank": 1,
                "master_addr": "127.0.0.1",
                "master_port": 29500,
            },
            "training": {"data_path": "/data/tokens"},
        }
    )

    assert data["distributed"]["nproc_per_node"] == 1
    assert data["runtime"]["workdir"].endswith("/job-1")
    assert data["training"]["entrypoint"] == "pretrain_gpt.py"
    assert validate_worker_task(data) == []

    missing = merge_worker_task_defaults({"job_id": "broken", "training": {"data_path": "/data"}})
    errors = validate_worker_task(missing)
    assert "distributed.nnodes" in errors
    assert "distributed.node_rank" in errors
    assert "distributed.master_addr" in errors
    assert "distributed.master_port" in errors


def test_load_worker_task_reports_validation_errors(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("job_id: bad\ntraining:\n  data_path: /data\n", encoding="utf-8")

    try:
        load_worker_task(path)
    except WorkerTaskError as exc:
        assert "distributed.nnodes" in exc.errors
        assert "distributed.node_rank" in exc.errors
    else:
        raise AssertionError("expected WorkerTaskError")
