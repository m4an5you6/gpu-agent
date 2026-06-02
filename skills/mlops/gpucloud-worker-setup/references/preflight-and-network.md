# GPUCLOUD Preflight Check Source Reference

## Source File
`/usr/local/lib/hermes-agent/hermes_cli/gpucloud_worker.py` — `run_worker_preflight()` (line 220-350)

## Workflow State Machine

Source: `/usr/local/lib/hermes-agent/hermes_cli/gpucloud_worker_goal.py`

```
preflight → train_dry_run → training_running → training_completed
    ↓                                            ↓
training_failed                          conversion_resolving → conversion_running
                                              ↓                        ↓
                                        conversion_completed    conversion_failed
                                              ↓
                                        inference_starting → inference_running
                                                                  ↓        ↓
                                                             completed  conversion_failed
```

Terminal stages: `training_failed`, `conversion_failed`, `completed`

**Important:** Once in a terminal stage, `gpucloud_worker_goal_run` returns immediately without re-running. Delete the state file to reset.

## Key Code Paths

### Preflight (gpucloud_worker_goal.py:339-357)
```python
if state["stage"] == "preflight":
    preflight = run_worker_preflight(task_file=task.path)
    if not preflight.get("ok"):
        state.update(stage="training_failed", ...)
    # If auto_execute: immediately start training
    if auto_execute:
        started = run_worker_start(task_file=task.path, confirm_execute=True, skip_preflight=True)
```

### Torch Probe (gpucloud_worker.py:171-202)
Runs: `<python> -c "import json, torch; print(json.dumps({...}))"`
Checks: torch_version, cuda_version, cuda_available, distributed_available, nccl_available

### State Persistence
- Worker goal state: `~/.hermes/gpucloud/worker_goal_runs/<job_id>.json`
- Worker job state: `~/.hermes/gpucloud/worker_jobs/<job_id>.json`
- Delete to reset a stuck workflow.

## 算力自由 (Suànlì Zìyóu) Instance Details

From the instance banner:
- CPU: 8 cores
- Memory: 42 GB
- GPU: Tesla T4, 1
- System disk: small (root)
- Data disk: /root/gpufree-data (fast, NOT preserved on image save)
- pip.conf: `index-url = https://pypi.tuna.tsinghua.edu.cn/simple`
- CUDA 12.6 pre-installed at /usr/local/cuda-12.6/
- cuDNN 9, NCCL 2, cublas, cufft, curand all pre-installed

## Network Test Results (2026-06-01)

| Endpoint | Status | Notes |
|----------|--------|-------|
| pypi.org | 200 | Via tuna.tsinghua mirror |
| pypi.tuna.tsinghua.edu.cn | 200 | Pre-configured in pip.conf |
| download.pytorch.org | 403 (root) / 200 (whl/) | Works, ~1.5MB/s |
| huggingface.co | timeout | Blocked |
| hf-mirror.com | 200 | Chinese HF mirror, serves API + file downloads |
| github.com | timeout/TLS error | Blocked |
| gitee.com | 200 | Chinese GitHub mirror, Megatron-LM clone works |
| research.metamind.io S3 | PermanentRedirect + SSL cert mismatch | Broken, use HF mirror |
| s3.amazonaws.com | 301 → SSL error | Bucket redirect endpoint has wrong cert |

## System CUDA Libraries (common on GPU cloud)

Most GPU cloud instances have CUDA pre-installed. Check with:
```bash
ldconfig -p | grep -E "cudnn|nccl|cublas|cufft|curand|nvrtc"
nvcc --version 2>/dev/null || echo "no nvcc (OK, runtime libs may still exist)"
nvidia-smi  # driver + CUDA compat version
```

Typical pre-installed libs: libcudart, libcudnn_ops, libcublas, libcufft, libcurand, libnccl, libnvrtc.
This means `pip install --no-deps torch` works and avoids downloading ~1.5GB of bundled NVIDIA packages.

## Torch Wheel Filename Convention

pip requires wheel filenames to follow PEP 427 format:
```
{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
```

The PyTorch download server returns files named like `torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl`.
If you curl with a short name (`-o torch.whl`), pip rejects it with "Invalid wheel filename (wrong number of parts)".
Always use the full name or rename after download.

## HF Mirror API for Dataset Discovery

List files in a dataset directory:
```bash
curl -s "https://hf-mirror.com/api/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1"
```

Returns JSON with file paths, sizes, and LFS hashes. Use `resolve/main/` URLs to download:
```bash
curl -L -o file.parquet "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet"
```
