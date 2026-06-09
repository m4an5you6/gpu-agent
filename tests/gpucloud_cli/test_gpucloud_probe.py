"""Phase 5 probe unit tests (no live SSH)."""

from __future__ import annotations

import textwrap

from gpucloud_cli.gpucloud_probe import (
    command_allowed,
    parse_nvidia_smi_output,
    probe_local_gpu,
    run_cluster_check,
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
    dataset_name: ds
    model_name: m
    """
).strip()


def test_parse_nvidia_smi_csv():
    text = "0, NVIDIA A100, 535.54, 81920, 1024"
    parsed = parse_nvidia_smi_output(text)
    assert parsed["available"] is True
    assert parsed["gpu_count"] == 1


def test_command_allowed_prefixes():
    assert command_allowed("nvidia-smi -L", ["nvidia-smi", "echo"])
    assert not command_allowed("rm -rf /", ["echo", "nvidia-smi"])


def test_local_gpu_skip_without_driver():
    out = probe_local_gpu()
    assert "target" in out


def test_cluster_check_requires_goal_or_discover(tmp_path, monkeypatch):
    path = tmp_path / "gpucloud.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # Without goal, discover allowed for CLI-style call
    result = run_cluster_check(allow_discover_without_goal=True)
    assert "nodes" in result
    assert result["nodes_checked"] >= 1
