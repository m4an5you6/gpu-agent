---
name: gpucloud-inference-deployment
description: Deploy GPUCLOUD inference from trained checkpoints using local conversion and vLLM worker runtime. Use for Megatron-to-HF conversion, validating vLLM-loadable model directories, starting local vLLM, health checks, logs, exit codes, and understanding that ComputingPlatform Deployment-master inference routes are currently disabled stubs.
version: 1.0.0
author: GPUCLOUD
platforms: [linux]
metadata:
  gpucloud:
    tags: [gpucloud, inference, deployment, vllm, checkpoint-conversion, hf, megatron, serving]
    related_skills: [gpucloud-worker-setup, gpucloud-sft-training, serving-llms-vllm]
    triggers:
      - deploy trained model
      - vllm from training
      - megatron checkpoint conversion
      - gpucloud inference deployment
      - model_path not loadable
---

# GPUCLOUD Inference Deployment

Use this skill after training completes or when a task asks to expose a trained model through vLLM.

## Core Rule

The worker starts only local inference processes. It does not SSH to other machines and does not invent conversion commands. Prefer structured `conversion` and `inference` fields; run commands only when provided by the task or safely discovered by GPUCLOUD code.

## Model Resolution

Before starting vLLM:

1. If `inference.model_path` exists and is HF/vLLM loadable, start vLLM directly.
2. Else if `conversion.command_template` is provided, run the local conversion wrapper and validate output.
3. Else allow GPUCLOUD auto-discovery to look for compatible Megatron conversion tools.
4. If no deterministic conversion path exists, return `conversion_failed` with a clear reason and do not start vLLM.

## Task Shape

Use structured inference config:

```yaml
conversion:
  output_dir: /root/gpufree-data/models/job-123-hf
  auto_discover: true
inference:
  engine: vllm
  model_path: /root/gpufree-data/models/job-123-hf
  host: 0.0.0.0
  port: 8000
  tensor_parallel: 1
  extra_args: []
```

Keep host, port, tensor parallelism, and extra args explicit. Do not hide them in a shell string.

## vLLM Start Contract

- Launch through the local process wrapper.
- Write stdout/stderr to an inference log.
- Write exit code to a sidecar file.
- Poll local health before marking deployment ready.
- Return tail logs on failure.
- Stop only the local pid or process group.

## References

- Read `references/vllm-runtime-and-model-readiness.md` for vLLM package compatibility and HF model directory checks.
- Read `references/deployment-master-inference-status.md` before assuming the one-click deployment backend has usable inference APIs.
