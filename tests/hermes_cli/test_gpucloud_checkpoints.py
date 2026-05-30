"""Phase 7 checkpoint list, validation, resume, and cleanup tests."""

from __future__ import annotations

import textwrap

from hermes_cli.gpucloud_checkpoints import (
    build_resume_training_command,
    run_checkpoint_cleanup,
    run_checkpoint_list,
    run_checkpoint_validate,
    run_train_resume,
)
from hermes_cli.gpucloud_config import prepare_gpucloud_config
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


def _write_config(tmp_path):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    return path


def test_checkpoint_list_identifies_latest(tmp_path, monkeypatch):
    path = _write_config(tmp_path)

    def fake_ssh(**kwargs):
        return SSHResult(
            ok=True,
            exit_code=0,
            stdout=(
                "__GPUCLOUD_ROOT__|/home/ubuntu/gpucloud/checkpoints/llama-3-8b\n"
                "200|/home/ubuntu/gpucloud/checkpoints/llama-3-8b/checkpoint-200\n"
                "100|/home/ubuntu/gpucloud/checkpoints/llama-3-8b/checkpoint-100\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("hermes_cli.gpucloud_checkpoints.run_ssh_command", fake_ssh)
    out = run_checkpoint_list(
        config_file=str(path),
        allow_discover_without_goal=True,
    )
    assert out["ok"]
    assert out["latest"]["name"] == "checkpoint-200"
    assert out["count"] == 2


def test_checkpoint_validate_reports_damaged_checkpoint(tmp_path, monkeypatch):
    path = _write_config(tmp_path)

    def fake_ssh(**kwargs):
        return SSHResult(
            ok=False,
            exit_code=3,
            stdout=(
                "__GPUCLOUD_CHECKPOINT__|/home/ubuntu/checkpoint-bad\n"
                "__GPUCLOUD_NO_MARKERS__|/home/ubuntu/checkpoint-bad\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("hermes_cli.gpucloud_checkpoints.run_ssh_command", fake_ssh)
    out = run_checkpoint_validate(
        config_file=str(path),
        checkpoint_path="/home/ubuntu/checkpoint-bad",
        allow_discover_without_goal=True,
    )
    assert not out["ok"]
    assert "damaged" in out["error"]
    assert out["required_any"]


def test_resume_command_appends_checkpoint_arg(tmp_path):
    path = _write_config(tmp_path)
    prepared = prepare_gpucloud_config(path)
    command = build_resume_training_command(
        prepared.merged,
        "/home/ubuntu/gpucloud/checkpoints/llama-3-8b/checkpoint-200",
    )
    assert command.startswith("torchrun ")
    assert "--resume_from_checkpoint" in command
    assert "checkpoint-200" in command


def test_train_resume_dry_run_uses_validated_checkpoint(tmp_path, monkeypatch):
    path = _write_config(tmp_path)

    def fake_validate(**kwargs):
        return {
            "ok": True,
            "checkpoint_path": "/home/ubuntu/checkpoints/checkpoint-200",
            "markers": ["trainer_state.json"],
        }

    monkeypatch.setattr(
        "hermes_cli.gpucloud_checkpoints.run_checkpoint_validate",
        fake_validate,
    )
    out = run_train_resume(
        config_file=str(path),
        checkpoint_path="/home/ubuntu/checkpoints/checkpoint-200",
        dry_run=True,
        allow_discover_without_goal=True,
    )
    assert out["ok"]
    assert out["dry_run"] is True
    assert "--resume_from_checkpoint" in out["launch_command"]
    assert out["source_checkpoint"].endswith("checkpoint-200")


def test_checkpoint_cleanup_defaults_to_dry_run(tmp_path, monkeypatch):
    path = _write_config(tmp_path)

    def fake_list(**kwargs):
        return {
            "ok": True,
            "checkpoint_root": "/home/ubuntu/gpucloud/checkpoints/llama-3-8b",
            "checkpoints": [
                {"path": "/home/ubuntu/gpucloud/checkpoints/llama-3-8b/checkpoint-300"},
                {"path": "/home/ubuntu/gpucloud/checkpoints/llama-3-8b/checkpoint-200"},
                {"path": "/home/ubuntu/gpucloud/checkpoints/llama-3-8b/checkpoint-100"},
            ],
        }

    monkeypatch.setattr(
        "hermes_cli.gpucloud_checkpoints.run_checkpoint_list",
        fake_list,
    )
    out = run_checkpoint_cleanup(
        config_file=str(path),
        keep=1,
        allow_discover_without_goal=True,
    )
    assert out["ok"]
    assert out["dry_run"] is True
    assert out["delete_count"] == 2
    assert "rm -rf" in out["delete_command"]
