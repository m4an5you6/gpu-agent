---
name: cluster-control
description: Operate the temporary master training control plane.
version: 1.0.0
author: NousResearch
platforms: [linux, macos]
metadata:
  gpucloud:
    category: devops
    tags: [cluster, training, torchrun, distributed]
---

# Cluster Control Skill

GPUCLOUD can coordinate multi-node distributed training through a temporary
master control plane. The master assigns ranks and launches `torchrun` on each
worker; NCCL/Gloo carries the data plane. Workers may be **heterogeneous** —
each node maps logical project, dataset, and conda env keys to local paths.

## When to Use

- Submit or monitor multi-node `torchrun` / DeepSpeed jobs across workers with
  different code paths, data mounts, or conda environments.
- Inspect node health, stale workers, or structured training logs after failures.
- Validate a job spec programmatically before launching expensive GPU time.

## Prerequisites

1. Enable the plugin: `plugins.enabled: [cluster]` in `config.yaml`.
2. Set `cluster.enabled: true` and configure `cluster.master_url`, `cluster.node_id`.
3. Set `GPUCLOUD_CLUSTER_SECRET` in `~/.gpucloud/.env` (shared bearer token).
4. Optional Postgres: `cluster.database_url` (otherwise in-memory for dev/tests).
5. Install deps: `pip install 'gpucloud-agent[cluster]'` for Postgres support.
6. Configure per-node paths and conda envs (see below).
7. Start master: `gpucloud cluster serve` on the designated host.
8. Start workers: `gpucloud cluster worker` on each GPU node.

## Heterogeneous Worker Setup

Each worker declares **logical keys → local paths** in `config.yaml`. The master
schedules by capability; each worker resolves paths before launch.

```yaml
cluster:
  enabled: true
  node_paths:
    code_roots:
      my-model: /srv/gpucloud/projects/my-model/releases
    data_roots:
      redpajama-v1: /mnt/datasets/redpajama/v1
    checkpoint_roots:
      my-model: /mnt/checkpoints/my-model
    scratch_roots:
      default: /local_nvme/gpucloud/scratch
  conda:
    envs:
      my-model-cu124: /srv/gpucloud/conda/envs/my-model-cu124/bin/python
  training:
    python_executable: python   # fallback when env_name omitted
```

Workers heartbeat these capabilities to the master. Jobs reference logical names
instead of hardcoded absolute paths shared across nodes.

## How to Run

| Task | Command / Tool |
|------|----------------|
| Cluster snapshot | `cluster_status` |
| Validate spec | `cluster_validate_config` with `spec` object |
| Submit training | `cluster_submit_job` with `spec` (see below) |
| Job progress | `cluster_job_status` with `job_id` |
| Logs / errors | `cluster_logs` with `job_id` and/or `node_id` |
| Cancel job | `cluster_stop_job` |
| Quarantine node | `cluster_node_action` action=`quarantine` |

### Logical job spec (heterogeneous)

```json
{
  "script": "train.py",
  "nnodes": 4,
  "nproc_per_node": 8,
  "framework": "torchrun",
  "env_name": "my-model-cu124",
  "project": "my-model",
  "release": "a1b2c3d",
  "dataset": "redpajama-v1",
  "output_run_id": "run-20260612-001",
  "min_scratch_gb": 500,
  "script_args": ["--epochs", "3"]
}
```

Each worker resolves locally:

- `working_dir` → `{code_roots[project]}/{release}`
- `DATA_DIR` env → `{data_roots[dataset]}`
- `OUTPUT_DIR` env → `{checkpoint_roots[project]}/{output_run_id}`
- `TMPDIR` / `SCRATCH_DIR` → best `{scratch_roots[*]}` with enough free space
- Python → `{conda.envs[env_name]}`

Legacy absolute `working_dir` still works when logical fields are omitted.

Use `framework: placeholder` for dry-run rank planning without launching real training.

## Quick Reference

```yaml
cluster:
  enabled: true
  role: auto
  master_url: http://10.0.0.1:8765
  database_url: postgresql+psycopg://user:pass@127.0.0.1:5432/gpucloud_cluster
  event_routing:
    job_failed: queue
    node_lost: guide
    config_mismatch: interrupt
```

Event routing modes: `record` (DB only), `queue` (next turn message), `guide`
(steer into running agent), `interrupt` (stop current turn), `execute_direct`
(deterministic action, no LLM).

## Procedure

1. Call `cluster_status` — confirm enough `ready` nodes and none in `stale_nodes`.
2. Ensure each worker has code, data, conda env, and scratch configured locally.
3. Call `cluster_validate_config` with the intended training spec.
4. Fix any validation errors before submit (script, nnodes, env keys).
5. Call `cluster_submit_job`; record returned `job_id` and per-node assignments.
6. Poll `cluster_job_status` until state is `succeeded`, `failed`, or `stopped`.
7. On failure, call `cluster_logs` for process tail summaries and error rows.
8. If a node misbehaves, `cluster_node_action` with `quarantine`, then `restore`.

## Pitfalls

- `role: auto` does not elect a leader — configure one explicit master host.
- Master filters nodes by heartbeat capabilities; register workers before submit.
- Launch commands are built on each worker — do not assume shared absolute paths.
- Workers reject assignments when paths, conda env, or scratch space are missing.
- Idempotent submits require a stable `idempotency_key` in the job spec.
- Full stdout/stderr live on disk under `cluster.data_dir/logs/`; Postgres stores tails only.

## Verification

- `cluster_status` shows all nodes `ready`, none in `stale_nodes`.
- `cluster_job_status` reports assignments in `running` then terminal state.
- `cluster_logs` returns matching `process_runs` with exit codes for each node.
