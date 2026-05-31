# GPUCLOUD Agent

GPUCLOUD Agent is a CLI-first ML operations agent for GPU clusters. It keeps the
core agent loop, tools, memory, skills, cron, and delegation features, then
narrows the product surface around GPU probing, Megatron-LM training,
checkpoint management, vLLM inference, and goal-driven ML workflows.

This repository now exposes `gpucloud` as the primary command. The legacy
general-purpose messaging, dashboard, entertainment, and IDE surfaces are not
part of the default GPUCLOUD workflow.

## What Changed

- User-visible product identity is GPUCLOUD.
- `gpucloud.yaml` is the ML cluster configuration file.
- `/goal` is the only Agent path that implicitly loads GPUCLOUD ML config; a
  worker task file is preferred, with `gpucloud.yaml` as the SSH fallback.
- Explicit CLI commands can load `gpucloud.yaml` for validation, probes,
  training, checkpoints, and inference.
- Training framework is Megatron-LM.
- Inference engine is vLLM.
- Distributed Megatron worker runtime is supported through per-node task files.
- GPUCLOUD starts, checks, and monitors worker processes; Megatron-LM, PyTorch
  distributed, and NCCL perform model training communication.

## Architecture

GPUCLOUD separates three responsibilities:

| Layer | Responsibility |
| --- | --- |
| Coordinator or user script | Decides which machines participate and distributes per-node task files. This repository does not require Kubernetes, Slurm, or SSH fan-out for the first worker runtime. |
| GPUCLOUD Agent / CLI | Validates configs, probes GPUs, renders dry-run plans, starts local or remote processes, records job state, tails logs, and exposes Agent tools. |
| ML runtime | Megatron-LM, PyTorch distributed, NCCL, and vLLM run the actual training or inference processes. |

The important distributed-training distinction is that a multi-node
Megatron-LM job is not launched only from the master node. Every training node
runs a local command with the same rendezvous address and a different
`node_rank`.

For 4 machines with 1 GPU each, each node runs a command shaped like:

```bash
torchrun \
  --nnodes=4 \
  --nproc-per-node=1 \
  --node-rank=<0|1|2|3> \
  --master-addr=<rank0_host_or_ip> \
  --master-port=<port> \
  /opt/Megatron-LM/pretrain_gpt.py \
  ...
```

GPUCLOUD does not proxy gradients or replace NCCL. It prepares and supervises
the local process that joins the PyTorch distributed group.

## Core Commands

Validate the cluster config:

```bash
gpucloud config validate --file gpucloud.yaml
```

Check SSH connectivity, workdirs, and GPUs:

```bash
gpucloud cluster check --file gpucloud.yaml
```

Dry-run or start single-node Megatron-LM training through the SSH path:

```bash
gpucloud train dry-run --file gpucloud.yaml
gpucloud train start --file gpucloud.yaml --yes
gpucloud train status --limit 10
gpucloud train logs <job-id> --lines 100
```

Manage checkpoints:

```bash
gpucloud checkpoint list --file gpucloud.yaml
gpucloud checkpoint latest --file gpucloud.yaml
gpucloud checkpoint validate --file gpucloud.yaml
gpucloud checkpoint resume --file gpucloud.yaml --yes
gpucloud checkpoint cleanup --file gpucloud.yaml --keep 3 --yes
```

Dry-run, start, health-check, and stop vLLM inference:

```bash
gpucloud infer dry-run --file gpucloud.yaml
gpucloud infer start --file gpucloud.yaml --yes
gpucloud infer status --limit 10
gpucloud infer health <job-id> --file gpucloud.yaml
gpucloud infer stop <job-id> --file gpucloud.yaml --yes
```

Run a distributed Megatron worker on each participating machine:

```bash
gpucloud worker wait --task-file /data/gpucloud/task.yaml
gpucloud worker preflight --task-file /data/gpucloud/task.yaml
gpucloud worker dry-run --task-file /data/gpucloud/task.yaml
gpucloud worker start --task-file /data/gpucloud/task.yaml --yes
gpucloud worker status --job-id gpt-pretrain-001
gpucloud worker logs --job-id gpt-pretrain-001 --lines 100
gpucloud worker stop --job-id gpt-pretrain-001 --yes
```

`worker start` and `worker stop` require explicit confirmation with `--yes`.
`worker dry-run` never starts a process.

In child-agent mode, `/goal` prefers a local worker task file over
`gpucloud.yaml`. Discovery order is `GPUCLOUD_WORKER_TASK`, then
`./gpucloud-worker-task.yaml`, then `~/.gpucloud/worker-task.yaml`. When a
task file exists, `/goal` uses `gpucloud_worker_goal_run` and never performs
SSH cluster checks; the child agent only manages local Megatron, conversion,
and vLLM processes.

## `gpucloud.yaml`

Minimal cluster config:

```yaml
clusters:
  - name: prod
    nodes:
      - host: 10.0.0.1
        port: 22
        user: ubuntu
        ssh_key: ~/.ssh/id_rsa

dataset_name: my-dataset
model_name: llama-3-8b
```

Default behavior:

- `training.framework` defaults to `megatron-lm`.
- `training.command` is generated as a Megatron-LM `torchrun` command unless
  explicitly overridden.
- `inference.engine` defaults to `vllm`.
- `inference.port` defaults to `8000`.
- log and checkpoint paths are derived from the node workdir when omitted.
- dry-run is required by default before remote execution.

## Distributed Worker Task File

The coordinator or user script creates one `gpucloud-worker-task.yaml` per
machine. The task files share `job_id`, `nnodes`, `master_addr`, and
`master_port`, but each file has a different `node_rank`.

See [gpucloud-worker-task.yaml.example](gpucloud-worker-task.yaml.example) for
a complete starter file.

Example for rank 2 of a 4-node job:

```yaml
job_id: gpt-pretrain-001
framework: megatron-lm
role: worker

distributed:
  nnodes: 4
  nproc_per_node: 1
  node_rank: 2
  master_addr: 10.0.0.10
  master_port: 29500
  start_timeout_sec: 900

runtime:
  workdir: /data/gpucloud/jobs/gpt-pretrain-001
  megatron_lm_dir: /opt/Megatron-LM
  python: python
  env:
    NCCL_DEBUG: INFO
    NCCL_SOCKET_IFNAME: eth0

training:
  data_path: /data/datasets/tokens
  checkpoint_dir: /data/checkpoints/gpt-pretrain-001
  log_dir: /data/logs/gpucloud
  command_template: ""
  extra_args:
    - --tensor-model-parallel-size=1
    - --pipeline-model-parallel-size=1
    - --micro-batch-size=1
    - --global-batch-size=4
    - --seq-length=2048

preflight:
  require_gpu_count: 1
  min_vram_gb: 16
  heterogeneous_policy: warn

goal:
  mode: train_and_infer
  auto_execute: true

conversion:
  output_dir: /data/models/gpt2-finetuned-hf
  command_template: ""
  auto_discover: true

inference:
  engine: vllm
  model_path: /data/models/gpt2-finetuned-hf
  host: 0.0.0.0
  port: 8000
  tensor_parallel: 1
  extra_args: []
```

Heterogeneous GPU policy:

- `reject`: fail preflight when expected GPU type or VRAM requirements are not met.
- `warn`: report the mismatch, but allow the worker to start if other hard checks pass.
- `allow`: record the environment without blocking on heterogeneity checks.

For heterogeneous multi-node training, Megatron parallelism must still be chosen
explicitly through `training.extra_args` or `training.command_template`.
GPUCLOUD does not auto-balance tensor or pipeline parallelism across different
GPU models.

## Agent Tools

The GPUCLOUD toolset includes:

- `gpucloud_cluster_check`
- `gpucloud_ssh_exec`
- `gpucloud_gpu_probe`
- `gpucloud_train_start`
- `gpucloud_train_status`
- `gpucloud_train_logs`
- `gpucloud_checkpoint_list`
- `gpucloud_checkpoint_latest`
- `gpucloud_checkpoint_validate`
- `gpucloud_train_resume`
- `gpucloud_checkpoint_cleanup`
- `gpucloud_infer_start`
- `gpucloud_infer_status`
- `gpucloud_infer_health`
- `gpucloud_infer_stop`
- `gpucloud_goal_prepare`
- `gpucloud_worker_wait`
- `gpucloud_worker_preflight`
- `gpucloud_worker_dry_run`
- `gpucloud_worker_start`
- `gpucloud_worker_status`
- `gpucloud_worker_logs`
- `gpucloud_worker_stop`
- `gpucloud_worker_goal_run`
- `gpucloud_worker_goal_status`

Worker goal tools use the fixed task discovery order above when `task_file` is
omitted. They do not dispatch commands to other machines or use SSH.

## Development

Create or use the local virtual environment, then install development
dependencies:

```bash
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
```

Run focused GPUCLOUD tests:

```bash
venv/bin/python -m pytest \
  tests/hermes_cli/test_gpucloud_worker_task.py \
  tests/hermes_cli/test_gpucloud_distributed.py \
  tests/hermes_cli/test_gpucloud_worker.py \
  tests/hermes_cli/test_gpucloud_train.py \
  tests/hermes_cli/test_gpucloud_inference.py -q
```

Run the full isolated test suite:

```bash
scripts/run_tests.sh
```

## Safety Boundaries

- Remote SSH training and inference default to dry-run.
- Local worker start/stop requires `--yes`.
- task files must not contain private SSH keys or plaintext tokens.
- secret-like environment variables are redacted from worker plans.
- long logs are tailed and truncated.
- `training.command_template` is treated as trusted input from the coordinator
  or the user.

## License

MIT. See [LICENSE](LICENSE).
