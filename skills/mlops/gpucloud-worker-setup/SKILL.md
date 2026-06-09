---
name: gpucloud-worker-setup
description: Prepare and repair a GPUCLOUD worker-local runtime for training and inference. Use for GPUCLOUD worker preflight failures, Megatron-LM or Megatron-SWIFT environment setup, CUDA/PyTorch/vLLM compatibility, data disk layout, mirrors, and local process readiness before /goal worker runs.
version: 1.0.0
author: GPUCLOUD
platforms: [linux]
metadata:
  gpucloud:
    tags: [gpucloud, worker, setup, preflight, cuda, pytorch, megatron, swift, vllm, mlops]
    related_skills: [gpucloud-sft-training, gpucloud-inference-deployment]
    triggers:
      - setup gpucloud worker
      - worker preflight failed
      - prepare megatron environment
      - prepare swift megatron environment
      - train_and_infer worker goal
---

# GPUCLOUD Worker Setup

Use this skill before calling `gpucloud_worker_goal_run` or when worker preflight fails. A GPUCLOUD child agent manages only local host processes; main-node SSH deployment and multi-host task distribution are outside this skill.

## Core Rule

Prepare the local environment first. Worker preflight expects these to already work:

- `runtime.python` imports `torch`.
- `torch.cuda.is_available()` is true.
- `training.data_path` or dataset paths exist and are readable when required.
- Megatron-LM entrypoint or Megatron-SWIFT runner dependencies are installed.
- Rendezvous ports are free.
- `runtime.env.PATH` includes the selected venv or conda `bin`.

If preflight has reached a terminal stage, inspect or remove the worker goal state under `~/.gpucloud/gpucloud/worker_goal_runs/` before rerunning.

## Setup Checklist

Keep runtime files on the data disk, normally `/root/gpufree-data`, not the small root disk.

1. Create or select the Python runtime declared by the task.
2. Install a CUDA-compatible PyTorch stack and verify `python -c "import torch; print(torch.cuda.is_available())"`.
3. Apply mirror settings from the task, such as `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, `PIP_TRUSTED_HOST`, and `HF_ENDPOINT`.
4. Install Megatron-LM or Megatron-SWIFT dependencies according to `training.runner`.
5. Prepare tokenizer, dataset, cache, and output directories before starting training.
6. Ensure local logs and exit-code files are writable by the worker process wrapper.
7. Run worker preflight again; start training only after preflight is clean.

## Variant Selection

- **Megatron-LM**: use for direct GPT-style pretrain/fine-tune tasks where GPUCLOUD renders a torchrun command from structured YAML.
- **Megatron-SWIFT**: use when `megatron.backend=swift` or `training.runner=swift_megatron`. Treat SWIFT as a Megatron runner, not a separate framework.
- **vLLM**: install or repair only when inference is requested or conversion has produced a vLLM/HF-loadable model directory.

## References

- Read `references/runtime-pitfalls.md` for known Megatron-LM, PyTorch, data preprocessing, checkpoint conversion, and vLLM compatibility traps.
- Read `references/deployment-runtime.md` when translating one-click deployment backend fields such as `docker`, `auto_install`, mirrors, `setup_command`, workdir, dataset upload, or runtime repair into worker-local setup.
