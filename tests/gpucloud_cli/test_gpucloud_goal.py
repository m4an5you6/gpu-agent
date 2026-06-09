"""Phase 9 GPUCLOUD /goal workflow tests."""

from __future__ import annotations

import textwrap

from gpucloud_cli.gpucloud_config import prepare_gpucloud_config
from gpucloud_cli.gpucloud_goal import (
    build_goal_context_block,
    infer_goal_intent,
    megatron_communication_notes,
    run_goal_prepare,
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


def test_goal_intent_inference_keywords():
    assert infer_goal_intent("部署 vllm 推理服务") == "infer"
    assert infer_goal_intent("train and then deploy endpoint") == "train_and_infer"
    assert infer_goal_intent("训练模型") == "train"


def test_goal_context_names_mandatory_prepare_tool(tmp_path):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    prepared = prepare_gpucloud_config(path)

    block = build_goal_context_block(prepared, goal="部署推理服务")

    assert "gpucloud_goal_prepare first" in block
    assert "gpucloud_cluster_check" not in block
    assert "confirm_execute=true" in block
    assert "MASTER_ADDR" in block
    assert "vLLM inference service" in block
    assert "Do not read or print SSH private key contents" in block


def test_goal_prepare_stops_on_cluster_failure(tmp_path, monkeypatch):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")

    monkeypatch.setattr(
        "gpucloud_cli.gpucloud_goal.run_cluster_check",
        lambda **kwargs: {
            "ok": False,
            "nodes": [{"status": "error", "error": "ssh failed"}],
        },
    )

    called = {"train": False}

    def fake_train(**kwargs):
        called["train"] = True
        return {"ok": True}

    monkeypatch.setattr("gpucloud_cli.gpucloud_goal.run_train_start", fake_train)

    out = run_goal_prepare(
        goal="训练模型",
        config_file=str(path),
        allow_discover_without_goal=True,
    )

    assert out["ok"] is False
    assert out["stage"] == "cluster_check"
    assert called["train"] is False
    assert "stopped before" in out["error"]
    assert "No training or inference command was started" in out["plan_summary"]


def test_goal_prepare_reaches_train_and_infer_dry_run(tmp_path, monkeypatch):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")

    monkeypatch.setattr(
        "gpucloud_cli.gpucloud_goal.run_cluster_check",
        lambda **kwargs: {"ok": True, "nodes_checked": 1, "nodes_ok": 1},
    )

    out = run_goal_prepare(
        goal="训练并部署推理服务",
        config_file=str(path),
        allow_discover_without_goal=True,
    )

    assert out["ok"] is True
    assert out["stage"] == "dry_run"
    assert out["intent"] == "train_and_infer"
    assert out["dry_runs"]["train"]["dry_run"] is True
    assert out["dry_runs"]["infer"]["dry_run"] is True
    assert "torchrun" in out["dry_runs"]["train"]["launch_command"]
    assert "vllm serve" in out["dry_runs"]["infer"]["launch_command"]
    assert "GPUCLOUD Goal Plan" in out["plan_summary"]
    assert "Train dry-run" in out["plan_summary"]
    assert "Inference dry-run" in out["plan_summary"]
    assert "Megatron communication" in out["plan_summary"]
    assert "MASTER_ADDR" in out["communication"]["multi_node_requirement"]
    assert "confirm_execute=true" in out["plan_summary"]


def test_megatron_communication_notes_define_boundary():
    notes = megatron_communication_notes()
    assert notes["default_scope"] == "single_node_torchrun"
    assert "NCCL" in notes["default_communication"]
    assert "external launcher" in notes["multi_node_requirement"]
    assert "Heterogeneous GPUs" in notes["heterogeneous_gpu_warning"]


def test_generic_goal_does_not_require_gpucloud_yaml(tmp_path, monkeypatch):
    from gpucloud_cli.goals import GoalManager

    monkeypatch.chdir(tmp_path)
    mgr = GoalManager(session_id="generic-goal-no-yaml")

    state = mgr.set("write a short project note")

    assert state.gpucloud_goal is False
    assert mgr.initial_user_message() == "write a short project note"


def test_gpucloud_goal_prefers_worker_task_without_gpucloud_yaml(tmp_path, monkeypatch):
    from gpucloud_cli.goals import GoalManager

    task = tmp_path / "gpucloud-worker-task.yaml"
    workdir = tmp_path / "work"
    megatron = tmp_path / "Megatron-LM"
    data = tmp_path / "tokens"
    ckpt = tmp_path / "checkpoints"
    logs = tmp_path / "logs"
    megatron.mkdir()
    data.mkdir()
    (megatron / "pretrain_gpt.py").write_text("print('train')\n", encoding="utf-8")
    task.write_text(
        textwrap.dedent(
            f"""
            job_id: worker-goal-test
            distributed:
              nnodes: 1
              nproc_per_node: 1
              node_rank: 0
              master_addr: 127.0.0.1
              master_port: 29611
            runtime:
              workdir: {workdir}
              megatron_lm_dir: {megatron}
            training:
              data_path: {data}
              checkpoint_dir: {ckpt}
              log_dir: {logs}
            """
        ).strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    mgr = GoalManager(session_id="worker-goal-no-yaml")
    state = mgr.set("训练并推理 gpt2")
    message = mgr.initial_user_message()

    assert state.gpucloud_goal is True
    assert message is not None
    assert "gpucloud_worker_goal_run first" in message
    assert "Do not run SSH cluster checks" in message
    assert "gpucloud_goal_prepare first" not in message
