"""Phase 9 GPUCLOUD /goal workflow tests."""

from __future__ import annotations

import textwrap

from hermes_cli.gpucloud_config import prepare_gpucloud_config
from hermes_cli.gpucloud_goal import (
    build_goal_context_block,
    infer_goal_intent,
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
    assert "vLLM inference service" in block
    assert "Do not read or print SSH private key contents" in block


def test_goal_prepare_stops_on_cluster_failure(tmp_path, monkeypatch):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")

    monkeypatch.setattr(
        "hermes_cli.gpucloud_goal.run_cluster_check",
        lambda **kwargs: {
            "ok": False,
            "nodes": [{"status": "error", "error": "ssh failed"}],
        },
    )

    called = {"train": False}

    def fake_train(**kwargs):
        called["train"] = True
        return {"ok": True}

    monkeypatch.setattr("hermes_cli.gpucloud_goal.run_train_start", fake_train)

    out = run_goal_prepare(
        goal="训练模型",
        config_file=str(path),
        allow_discover_without_goal=True,
    )

    assert out["ok"] is False
    assert out["stage"] == "cluster_check"
    assert called["train"] is False
    assert "stopped before" in out["error"]


def test_goal_prepare_reaches_train_and_infer_dry_run(tmp_path, monkeypatch):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")

    monkeypatch.setattr(
        "hermes_cli.gpucloud_goal.run_cluster_check",
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


def test_generic_goal_does_not_require_gpucloud_yaml(tmp_path, monkeypatch):
    from hermes_cli.goals import GoalManager

    monkeypatch.chdir(tmp_path)
    mgr = GoalManager(session_id="generic-goal-no-yaml")

    state = mgr.set("write a short project note")

    assert state.gpucloud_goal is False
    assert mgr.initial_user_message() == "write a short project note"
