"""Distributed Megatron command rendering tests."""

from __future__ import annotations

import textwrap

from gpucloud_cli.gpucloud_distributed import (
    build_megatron_worker_command,
    build_swift_megatron_config,
    build_swift_megatron_worker_command,
    build_worker_plan,
)
from gpucloud_cli.gpucloud_worker_task import load_worker_task


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


def test_swift_megatron_runner_renders_structured_megatron_sft(tmp_path):
    path = tmp_path / "swift-task.yaml"
    path.write_text(
        textwrap.dedent(
            """
            job_id: qwen-swift-001
            framework: megatron-lm
            role: worker
            environment:
              hf_endpoint: https://hf-mirror.com
            distributed:
              nnodes: 2
              nproc_per_node: 1
              node_rank: 0
              master_addr: 10.0.22.72
              master_port: 29500
            runtime:
              workdir: /tmp/gpucloud/qwen
              megatron_lm_dir: /opt/Megatron-LM
              env:
                MODELSCOPE_CACHE: /shared/modelscope
            training:
              runner: swift_megatron
              training_type: sft
              batch_size: 2
              learning_rate: 5.0e-5
              max_steps: 50
              checkpoint_dir: /tmp/gpucloud/qwen/checkpoints
              log_dir: /tmp/gpucloud/qwen/logs
              swift:
                train_type: lora
                model: Qwen2.5-Coder-7B
                dataset: swift/sharegpt:common-zh
                max_length: 2048
                target_modules: all-linear
                lora_rank: 8
                lora_alpha: 32
                lora_dropout: 0.05
                attention_backend: unfused
                padding_free: false
                packing: false
            """
        ).strip(),
        encoding="utf-8",
    )
    task = load_worker_task(path)

    command = build_swift_megatron_worker_command(task)
    config = build_swift_megatron_config(task)
    plan = build_worker_plan(task)

    assert command.startswith("megatron sft")
    assert "--model Qwen2.5-Coder-7B" in command
    assert "--dataset swift/sharegpt:common-zh" in command
    assert "--lora_rank 8" in command
    assert "--save /tmp/gpucloud/qwen/checkpoints" in command
    assert config["micro_batch_size"] == 2
    assert config["lr"] == 5.0e-5
    assert plan["training_runner"] == "swift_megatron"
    assert plan["env"]["MODELSCOPE_CACHE"] == "/shared/modelscope"
    assert "Megatron-SWIFT" in plan["communication"]
