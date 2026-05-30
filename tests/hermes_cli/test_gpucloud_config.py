"""Tests for GPUCLOUD phase-4 yaml config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hermes_cli.gpucloud_config import (
    GpucloudConfigError,
    generate_training_command,
    merge_gpucloud_defaults,
    prepare_gpucloud_config,
    validate_required,
)
from hermes_cli.goals import GoalManager


MINIMAL_YAML = textwrap.dedent(
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


def test_validate_required_minimal_ok(tmp_path):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL_YAML, encoding="utf-8")
    raw = __import__("yaml").safe_load(path.read_text())
    assert validate_required(raw) == []


def test_validate_required_missing_port(tmp_path):
    yaml_text = MINIMAL_YAML.replace("        port: 22\n", "")
    data = __import__("yaml").safe_load(yaml_text)
    errors = validate_required(data)
    assert any("port" in e for e in errors)


def test_merge_defaults_and_generated_command(tmp_path):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL_YAML, encoding="utf-8")
    prepared = prepare_gpucloud_config(path)
    assert prepared.merged["training"]["framework"] == "megatron-lm"
    assert "my-dataset" in prepared.merged["training"]["log_dir"]
    assert "pretrain_gpt.py" in prepared.training_command
    assert "torchrun" in prepared.training_command
    summary = "\n".join(prepared.summary_lines())
    assert "BEGIN" not in summary
    assert "~/.ssh/id_rsa" in summary


def test_training_command_override(tmp_path):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(
        MINIMAL_YAML
        + "\ntraining:\n  command: custom train cmd\n",
        encoding="utf-8",
    )
    prepared = prepare_gpucloud_config(path)
    assert prepared.training_command == "custom train cmd"


def test_missing_workdir_does_not_fail_validate(tmp_path):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL_YAML, encoding="utf-8")
    prepared = prepare_gpucloud_config(path)
    node = prepared.merged["clusters"][0]["nodes"][0]
    assert node["workdir"] == "~/gpucloud"


def test_effective_dataset_override():
    data = __import__("yaml").safe_load(
        MINIMAL_YAML + "\ntraining:\n  dataset_name: v2\n"
    )
    merged = merge_gpucloud_defaults(data)
    assert generate_training_command(merged).count("--data-path v2")


def test_inline_ssh_key_rejected():
    data = __import__("yaml").safe_load(MINIMAL_YAML)
    data["clusters"][0]["nodes"][0]["ssh_key"] = "-----BEGIN PRIVATE KEY-----\nabc"
    errors = validate_required(data)
    assert any("ssh_key" in e for e in errors)


def test_goal_set_loads_context(tmp_path, monkeypatch):
    cfg = tmp_path / "gpucloud.yaml"
    cfg.write_text(MINIMAL_YAML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    mgr = GoalManager(session_id="test-goal-gpucloud")
    mgr.set("train the model")
    msg = mgr.initial_user_message()
    assert msg is not None
    assert "[GPUCLOUD goal context" in msg
    assert "gpucloud_goal_prepare" in msg
    assert "my-dataset" in msg
    assert "train the model" in msg


def test_goal_set_fails_on_missing_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mgr = GoalManager(session_id="test-goal-no-yaml")
    with pytest.raises(ValueError, match="gpucloud.yaml|no gpucloud"):
        mgr.set("run training")
