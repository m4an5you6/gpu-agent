"""Local distributed worker runtime tests."""

from __future__ import annotations

import textwrap

from hermes_cli.gpucloud_worker import (
    run_worker_dry_run,
    run_worker_logs,
    run_worker_preflight,
    run_worker_start,
    run_worker_status,
    run_worker_stop,
    run_worker_wait,
)


def _write_task(tmp_path):
    workdir = tmp_path / "work"
    megatron = tmp_path / "Megatron-LM"
    data = tmp_path / "tokens"
    ckpt = tmp_path / "checkpoints"
    logs = tmp_path / "logs"
    megatron.mkdir()
    data.mkdir()
    (megatron / "pretrain_gpt.py").write_text("print('train')\n", encoding="utf-8")
    path = tmp_path / "gpucloud-worker-task.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            job_id: worker-test-1
            framework: megatron-lm
            role: worker
            distributed:
              nnodes: 1
              nproc_per_node: 1
              node_rank: 0
              master_addr: 127.0.0.1
              master_port: 29601
            runtime:
              workdir: {workdir}
              megatron_lm_dir: {megatron}
              env:
                NCCL_DEBUG: INFO
            training:
              data_path: {data}
              checkpoint_dir: {ckpt}
              log_dir: {logs}
              extra_args:
                - --micro-batch-size=1
            preflight:
              require_gpu_count: 1
              min_vram_gb: 0
              heterogeneous_policy: warn
            """
        ).strip(),
        encoding="utf-8",
    )
    return path


def test_worker_wait_and_dry_run(tmp_path):
    task_file = _write_task(tmp_path)

    waited = run_worker_wait(task_file=task_file, timeout_sec=0)
    assert waited["ok"]
    assert waited["task"]["job_id"] == "worker-test-1"

    dry = run_worker_dry_run(task_file=task_file)
    assert dry["ok"]
    assert dry["dry_run"] is True
    assert "--nnodes=1" in dry["launch_command"]
    assert "--node-rank=0" in dry["launch_command"]


def test_worker_preflight_reports_missing_gpu(tmp_path, monkeypatch):
    task_file = _write_task(tmp_path)

    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker.probe_local_gpu",
        lambda: {
            "target": "local",
            "available": False,
            "skipped": True,
            "reason": "nvidia-smi not in PATH",
            "gpus": [],
            "gpu_count": 0,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker._torch_probe",
        lambda python: {
            "ok": True,
            "cuda_available": True,
            "distributed_available": True,
            "nccl_available": True,
        },
    )

    out = run_worker_preflight(task_file=task_file, check_network=False)

    assert not out["ok"]
    failed = {c["name"] for c in out["checks"] if not c["ok"]}
    assert "gpu_count" in failed


def test_heterogeneous_policy_warn_does_not_fail_on_vram_warning(tmp_path, monkeypatch):
    task_file = _write_task(tmp_path)
    text = task_file.read_text(encoding="utf-8")
    task_file.write_text(text.replace("min_vram_gb: 0", "min_vram_gb: 80"), encoding="utf-8")

    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker.probe_local_gpu",
        lambda: {
            "target": "local",
            "available": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA RTX 4090",
                    "memory_total_mib": "24576",
                }
            ],
            "gpu_count": 1,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker._torch_probe",
        lambda python: {
            "ok": True,
            "cuda_available": True,
            "distributed_available": True,
            "nccl_available": True,
        },
    )

    out = run_worker_preflight(task_file=task_file, check_network=False)

    assert out["ok"]
    vram = [c for c in out["checks"] if c["name"] == "gpu_vram"][0]
    assert not vram["ok"]
    assert vram["severity"] == "warning"


def test_worker_start_status_logs_and_stop(tmp_path, monkeypatch):
    task_file = _write_task(tmp_path)
    monkeypatch.setenv("GPUCLOUD_WORKER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker.run_worker_preflight",
        lambda **kwargs: {"ok": True, "checks": []},
    )

    class DummyPopen:
        pid = 12345

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr("hermes_cli.gpucloud_worker.subprocess.Popen", DummyPopen)
    monkeypatch.setattr("hermes_cli.gpucloud_worker._pid_running", lambda pid: True)

    refused = run_worker_start(task_file=task_file, confirm_execute=False)
    assert not refused["ok"]

    started = run_worker_start(task_file=task_file, confirm_execute=True)
    assert started["ok"]
    assert started["job"]["status"] == "running"
    assert started["pid"] == 12345

    status = run_worker_status(job_id="worker-test-1")
    assert status["ok"]
    assert status["running"] is True

    log_path = tmp_path / "logs" / "worker-test-1.rank0.log"
    log_path.write_text("line1\nline2\n", encoding="utf-8")
    logs = run_worker_logs(job_id="worker-test-1", lines=1)
    assert logs["ok"]
    assert logs["tail"] == "line2"

    monkeypatch.setattr("hermes_cli.gpucloud_worker._terminate_pid", lambda pid: "terminated")
    stopped = run_worker_stop(job_id="worker-test-1", confirm_stop=True)
    assert stopped["ok"]
    assert stopped["job"]["status"] == "stopped"


def test_worker_stop_requires_confirmation(tmp_path, monkeypatch):
    task_file = _write_task(tmp_path)
    monkeypatch.setenv("GPUCLOUD_WORKER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker.run_worker_preflight",
        lambda **kwargs: {"ok": True, "checks": []},
    )

    class DummyPopen:
        pid = 222

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("hermes_cli.gpucloud_worker.subprocess.Popen", DummyPopen)
    run_worker_start(task_file=task_file, confirm_execute=True)

    stopped = run_worker_stop(job_id="worker-test-1", confirm_stop=False)
    assert not stopped["ok"]


def test_swift_megatron_multinode_requires_shared_modelscope_cache(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    logs = tmp_path / "logs"
    ckpt = tmp_path / "checkpoints"
    path = tmp_path / "swift-task.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            job_id: swift-cache-check
            framework: megatron-lm
            role: worker
            distributed:
              nnodes: 2
              nproc_per_node: 1
              node_rank: 0
              master_addr: 127.0.0.1
              master_port: 29631
            runtime:
              workdir: {workdir}
              megatron_lm_dir: {tmp_path / "Megatron-LM"}
            training:
              runner: swift_megatron
              training_type: sft
              checkpoint_dir: {ckpt}
              log_dir: {logs}
              swift:
                model: Qwen2.5-Coder-7B
                dataset: swift/sharegpt:common-zh
            """
        ).strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker.probe_local_gpu",
        lambda: {
            "target": "local",
            "available": True,
            "gpus": [{"index": 0, "name": "A10", "memory_total_mib": "24576"}],
            "gpu_count": 1,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.gpucloud_worker._torch_probe",
        lambda python: {
            "ok": True,
            "cuda_available": True,
            "distributed_available": True,
            "nccl_available": True,
        },
    )
    monkeypatch.setattr("hermes_cli.gpucloud_worker.shutil.which", lambda name: "/usr/bin/megatron")

    out = run_worker_preflight(task_file=path, check_network=False)

    assert not out["ok"]
    failed = {c["name"] for c in out["checks"] if not c["ok"]}
    assert "modelscope_cache_shared" in failed
