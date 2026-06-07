"""Local /goal worker state machine tests."""

from __future__ import annotations

import time
import textwrap

from hermes_cli.gpucloud_worker_goal import (
    run_worker_goal_run,
    run_worker_goal_status,
)


def _write_goal_task(tmp_path, *, conversion: str | None = None, auto_discover: bool = True):
    workdir = tmp_path / "work"
    megatron = tmp_path / "Megatron-LM"
    data = tmp_path / "tokens"
    ckpt = tmp_path / "checkpoints"
    logs = tmp_path / "logs"
    model = tmp_path / "model"
    megatron.mkdir()
    data.mkdir()
    (megatron / "pretrain_gpt.py").write_text("print('train')\n", encoding="utf-8")
    if conversion is None:
        conversion = (
            "python -c \"from pathlib import Path; "
            "Path('{conversion_output_dir}').mkdir(parents=True, exist_ok=True); "
            "Path('{conversion_output_dir}/config.json').write_text('x')\""
        )
    path = tmp_path / "gpucloud-worker-task.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            job_id: worker-goal-1
            framework: megatron-lm
            role: worker
            distributed:
              nnodes: 1
              nproc_per_node: 1
              node_rank: 0
              master_addr: 127.0.0.1
              master_port: 29621
            runtime:
              workdir: {workdir}
              megatron_lm_dir: {megatron}
            training:
              data_path: {data}
              checkpoint_dir: {ckpt}
              log_dir: {logs}
              command_template: python -c "print('trained')"
            conversion:
              output_dir: {model}
              command_template: >-
                {conversion}
              auto_discover: {str(auto_discover).lower()}
            inference:
              engine: vllm
              model_path: {model}
              port: 8123
              command_template: python -c "print('vllm')"
            preflight:
              require_gpu_count: 1
              min_vram_gb: 0
            """
        ).strip(),
        encoding="utf-8",
    )
    return path


def _wait_for_exit(path, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _advance_to_training(task):
    first = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    assert first["stage"] == "data_preparing"
    second = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    assert second["stage"] == "preflight"
    third = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    assert third["stage"] == "training_running"
    return third


def test_worker_goal_runs_training_conversion_and_inference(tmp_path, monkeypatch):
    task = _write_goal_task(tmp_path)
    monkeypatch.setenv("GPUCLOUD_WORKER_STATE_DIR", str(tmp_path / "worker-state"))
    monkeypatch.setenv("GPUCLOUD_WORKER_GOAL_STATE_DIR", str(tmp_path / "goal-state"))
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker.run_worker_preflight",
        lambda **kwargs: {"ok": True, "checks": []},
    )
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker_goal.run_worker_preflight",
        lambda **kwargs: {"ok": True, "checks": []},
    )
    monkeypatch.setattr("hermes_cli.gpucloud_worker_goal._health_ok", lambda port: True)

    first = _advance_to_training(task)
    assert first["ok"]
    assert first["backend"] == "worker_local"
    assert first["stage"] == "training_running"

    train_exit = tmp_path / "logs" / "worker-goal-1.rank0.exitcode"
    _wait_for_exit(train_exit)
    fourth = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    assert fourth["stage"] == "training_completed"

    fifth = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    assert fifth["stage"] == "conversion_running"

    conversion_exit = tmp_path / "logs" / "worker-goal-1.conversion.exitcode"
    _wait_for_exit(conversion_exit)
    sixth = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    assert sixth["stage"] == "conversion_completed"

    seventh = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    assert seventh["stage"] == "inference_running"

    final = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    assert final["stage"] == "completed"
    assert final["status"] == "completed"
    assert final["inference"]["healthy"] is True

    status = run_worker_goal_status(task_file=task)
    assert status["ok"]
    assert status["stage"] == "completed"


def test_worker_goal_preflight_failure_does_not_start_training(tmp_path, monkeypatch):
    task = _write_goal_task(tmp_path)
    monkeypatch.setenv("GPUCLOUD_WORKER_GOAL_STATE_DIR", str(tmp_path / "goal-state"))
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker_goal.run_worker_preflight",
        lambda **kwargs: {"ok": False, "checks": [{"name": "gpu_count", "ok": False}]},
    )

    run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    out = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")

    assert out["ok"] is False
    assert out["stage"] == "training_failed"
    assert "preflight" in out["last_error"]


def test_worker_goal_conversion_failure_does_not_start_inference(tmp_path, monkeypatch):
    task = _write_goal_task(tmp_path, conversion="", auto_discover=False)
    monkeypatch.setenv("GPUCLOUD_WORKER_STATE_DIR", str(tmp_path / "worker-state"))
    monkeypatch.setenv("GPUCLOUD_WORKER_GOAL_STATE_DIR", str(tmp_path / "goal-state"))
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker.run_worker_preflight",
        lambda **kwargs: {"ok": True, "checks": []},
    )
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker_goal.run_worker_preflight",
        lambda **kwargs: {"ok": True, "checks": []},
    )

    first = _advance_to_training(task)
    assert first["stage"] == "training_running"
    _wait_for_exit(tmp_path / "logs" / "worker-goal-1.rank0.exitcode")

    second = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")
    assert second["stage"] == "training_completed"
    third = run_worker_goal_run(task_file=task, goal="训练并推理 gpt2")

    assert third["ok"] is False
    assert third["stage"] == "conversion_failed"
    assert not third["inference"]
