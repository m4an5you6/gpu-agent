"""Distributed Megatron command rendering tests."""

from __future__ import annotations

import textwrap

from hermes_cli.gpucloud_distributed import (
    build_megatron_worker_command,
    build_worker_plan,
)
from hermes_cli.gpucloud_worker_task import load_worker_task


def _task_text(rank: int) -> str:
    return textwrap.dedent(
        f"""
        job_id: gpt-pretrain-001
        framework: megatron-lm
        role: worker
        distributed:
          nnodes: 4
          nproc_per_node: 1
          node_rank: {rank}
          master_addr: 10.0.0.10
          master_port: 29500
        runtime:
          workdir: /tmp/gpucloud/job
          megatron_lm_dir: /opt/Megatron-LM
          env:
            NCCL_DEBUG: INFO
            API_TOKEN: secret-token
        training:
          data_path: /data/tokens
          checkpoint_dir: /data/checkpoints/job
          log_dir: /data/logs
          extra_args:
            - --tensor-model-parallel-size=1
            - --pipeline-model-parallel-size=1
        """
    ).strip()


def test_four_worker_tasks_render_consistent_rendezvous(tmp_path):
    commands = []
    for rank in range(4):
        path = tmp_path / f"task-{rank}.yaml"
        path.write_text(_task_text(rank), encoding="utf-8")
        task = load_worker_task(path)
        command = build_megatron_worker_command(task)
        commands.append(command)

        assert "--nnodes=4" in command
        assert "--nproc-per-node=1" in command
        assert f"--node-rank={rank}" in command
        assert "--master-addr=10.0.0.10" in command
        assert "--master-port=29500" in command
        assert "/opt/Megatron-LM/pretrain_gpt.py" in command

    assert len(set(commands)) == 4
    assert all("--master-addr=10.0.0.10" in command for command in commands)


def test_worker_plan_redacts_sensitive_env(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(_task_text(0), encoding="utf-8")
    task = load_worker_task(path)

    plan = build_worker_plan(task)

    assert plan["ok"]
    assert plan["env"]["API_TOKEN"] == "***"
    assert "Megatron-LM/PyTorch distributed/NCCL" in plan["communication"]
    assert plan["log_path"].endswith("gpt-pretrain-001.rank0.log")


def test_command_template_can_override_default(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(
        _task_text(1)
        + "\ntraining:\n"
        + "  data_path: /data/tokens\n"
        + "  checkpoint_dir: /data/checkpoints/job\n"
        + "  log_dir: /data/logs\n"
        + "  command_template: torchrun --node-rank={node_rank} train.py {extra_args}\n",
        encoding="utf-8",
    )

    command = build_megatron_worker_command(load_worker_task(path))

    assert command.startswith("torchrun --node-rank=1 train.py")
