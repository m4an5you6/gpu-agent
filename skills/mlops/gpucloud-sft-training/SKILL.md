---
name: gpucloud-sft-training
description: Build structured GPUCLOUD worker task YAML for SFT or pretraining without generating shell training commands. Use for GPT single-GPU or multi-node Megatron-LM, Qwen LoRA SFT, Megatron-SWIFT, backend training field mapping, GPU allocation, HRL3D/distributed settings, and /goal worker training handoff.
version: 1.0.0
author: GPUCLOUD
platforms: [linux]
metadata:
  gpucloud:
    tags: [gpucloud, sft, training, pretraining, megatron-lm, megatron-swift, qwen, lora, gpt2, distributed, hrl3d]
    related_skills: [gpucloud-worker-setup, gpucloud-inference-deployment]
    triggers:
      - qwen sft
      - megatron swift training
      - gpt2 multi gpu training
      - gpucloud training yaml
      - backend training config
---

# GPUCLOUD SFT and Training

Use this skill when turning user intent or backend fields into GPUCLOUD worker tasks.

## Non-Negotiable Rule

Do not let the LLM invent a training shell command. Produce structured YAML fields only. GPUCLOUD deterministic code renders Megatron-LM or Megatron-SWIFT launch commands from those fields. Use `command_template` only when the user or backend explicitly provides one.

## Runner Selection

- `training.runner: megatron_lm` for direct Megatron-LM GPT-style tasks.
- `training.runner: swift_megatron` when backend config says `megatron.backend=swift`; this means Megatron-SWIFT runner, not a separate framework.
- `training.training_type: pretrain` for continuation/pretraining.
- `training.training_type: sft` for supervised fine-tuning.

## Worker Task Shape

Prefer this structured shape:

```yaml
environment:
  auto_install: true
  pip_index_url: https://pypi.tuna.tsinghua.edu.cn/simple
  hf_endpoint: https://hf-mirror.com
training:
  framework: megatron-lm
  runner: megatron_lm
  training_type: pretrain
  batch_size: 1
  learning_rate: 5.0e-5
  max_steps: 50
  distributed: false
  megatron: {}
  swift: {}
backend:
  training_job_id: 0
  gpu_id: 0
  node_id: "0"
```

Only include fields that are known. If required fields such as model, dataset, tokenizer, GPU IDs, or output path are missing, ask for clarification instead of fabricating values.

## Single-GPU Megatron-LM

For one GPU on one node:

- `distributed: false`
- `nproc_per_node: 1`
- `tensor_parallel: 1` unless explicitly requested otherwise
- dataset/tokenizer/checkpoint fields must come from the task or backend config
- training hyperparameters must come from the task or user; GPUCLOUD does not guess them

## Multi-Node or Multi-GPU

Use backend GPU allocation data to produce one worker task per node. When each child node has one GPU:

- `nnodes` equals the number of selected GPU IDs/nodes.
- `nproc_per_node: 1`.
- `node_rank` is deterministic from the ordered selected GPU list.
- `MASTER_ADDR` and `MASTER_PORT` are identical for every worker.
- rank 0 aligns with `master_node_id` or the selected master GPU.
- communication is owned by Megatron/PyTorch/NCCL or Megatron-SWIFT; GPUCLOUD only starts and monitors local worker processes.

## References

- Read `references/backend-training-contract.md` for backend field names and endpoint lifecycle.
- Read `references/distributed-gpu-mapping.md` for GPU ID to node/rank mapping rules.
- Read `references/megatron-swift-qwen-sft.md` for Qwen LoRA SFT preset and Megatron-SWIFT details.
