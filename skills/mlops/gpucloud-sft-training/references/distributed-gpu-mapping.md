# Distributed GPU Mapping

Use backend GPU and node records to produce deterministic per-worker tasks.

## Inputs

- Selected GPU IDs from allocation or user config.
- `/api/gpus/list` records with `GpuOut.id`, `GpuOut.node_id`, and `GpuOut.node_gpu_index`.
- `/api/nodes/list` records with `NodeOut.id`, `host`, `internal_ip`, and status.
- Optional `hrl3d.master_node_id`, `hrl3d.master_addr`, or backend-selected master GPU.

## Mapping Rules

1. Resolve every selected GPU ID to its `GpuOut`.
2. Resolve each GPU `node_id` to a `NodeOut`.
3. Use `str(NodeOut.id)` as worker `node_id`.
4. Pick rank 0 from `master_node_id` if present; otherwise use the first selected GPU after backend validation.
5. Prefer `NodeOut.internal_ip` for `MASTER_ADDR`; fall back to `host`.
6. Use one `MASTER_PORT` for all workers.
7. When each worker has one GPU, set `nproc_per_node: 1` and `CUDA_VISIBLE_DEVICES` or local GPU index to `node_gpu_index`.

## Per-Worker Fields

Every worker task should include:

```yaml
training:
  distributed: true
  nnodes: 2
  nproc_per_node: 1
  node_rank: 0
  master_addr: 10.0.0.10
  master_port: 29500
backend:
  training_job_id: 123
  gpu_id: 5
  node_id: "2"
  node_gpu_index: 0
```

Do not ask a child agent to schedule other machines. The main agent distributes one task per child. The child starts only its local rank; Megatron/PyTorch/NCCL perform inter-node communication after all ranks are launched.

## Validation Failures

Fail before dispatch if:

- any selected GPU ID is missing from `/api/gpus/list`
- any GPU has no node mapping
- rank 0 cannot resolve `MASTER_ADDR`
- selected GPUs imply different shared-data assumptions that the task cannot satisfy
- multi-node Megatron-SWIFT lacks a shared model or dataset cache path when required
