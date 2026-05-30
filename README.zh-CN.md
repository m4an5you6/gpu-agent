# GPUCLOUD Agent

GPUCLOUD Agent 是一个面向 GPU 集群和机器学习任务的 CLI 优先 Agent。它保留核心 Agent 循环、工具、记忆、技能、cron 和委派能力，同时把默认产品面收敛到 GPU 探测、Megatron-LM 训练、checkpoint 管理、vLLM 推理和 `/goal` 驱动的 ML 流程。

当前主入口是 `gpucloud`。通用消息平台、Dashboard、娱乐插件和 IDE 入口不属于 GPUCLOUD 默认工作流。

## 更新内容

- 用户可见产品名改为 GPUCLOUD。
- `gpucloud.yaml` 作为 ML 集群配置文件。
- 只有 `/goal` 会在 Agent 流程中隐式加载 `gpucloud.yaml`。
- 显式 CLI 命令可以加载 `gpucloud.yaml` 做校验、探测、训练、checkpoint 和推理管理。
- 训练框架固定为 Megatron-LM。
- 推理引擎固定为 vLLM。
- 新增 Distributed Megatron Worker Runtime，支持每台子机读取本机 task 文件并启动本机 rank。
- GPUCLOUD 负责检查、dry-run、启动和监控；Megatron-LM、PyTorch distributed、NCCL 负责训练通信。

## 架构差异

GPUCLOUD 把职责分为三层：

| 层 | 职责 |
| --- | --- |
| 主节点调度器或用户脚本 | 决定哪些机器参与训练，并把每台机器的 task 文件分发过去。第一版 worker runtime 不要求 Kubernetes、Slurm 或 GPUCLOUD 自己做 SSH fan-out。 |
| GPUCLOUD Agent / CLI | 校验配置、探测 GPU、生成 dry-run、启动本机或远程进程、记录 job 状态、查看日志、暴露 Agent 工具。 |
| ML Runtime | Megatron-LM、PyTorch distributed、NCCL 和 vLLM 执行真实训练或推理。 |

多机 Megatron-LM 训练的关键点是：不是只在主节点执行一个命令。每台训练节点都要启动本机命令，并使用相同 rendezvous 地址、不同 `node_rank` 加入同一个 distributed 组。

4 台机器、每台 1 张 GPU 时，每台机器的命令形态类似：

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

GPUCLOUD 不代理梯度通信，也不替代 NCCL；它只负责让本机训练进程可靠地加入 Megatron/PyTorch distributed 通信组。

## 常用命令

校验集群配置：

```bash
gpucloud config validate --file gpucloud.yaml
```

检查 SSH、工作目录和 GPU：

```bash
gpucloud cluster check --file gpucloud.yaml
```

单节点 Megatron-LM 训练 SSH 路径：

```bash
gpucloud train dry-run --file gpucloud.yaml
gpucloud train start --file gpucloud.yaml --yes
gpucloud train status --limit 10
gpucloud train logs <job-id> --lines 100
```

checkpoint 管理：

```bash
gpucloud checkpoint list --file gpucloud.yaml
gpucloud checkpoint latest --file gpucloud.yaml
gpucloud checkpoint validate --file gpucloud.yaml
gpucloud checkpoint resume --file gpucloud.yaml --yes
gpucloud checkpoint cleanup --file gpucloud.yaml --keep 3 --yes
```

vLLM 推理服务：

```bash
gpucloud infer dry-run --file gpucloud.yaml
gpucloud infer start --file gpucloud.yaml --yes
gpucloud infer status --limit 10
gpucloud infer health <job-id> --file gpucloud.yaml
gpucloud infer stop <job-id> --file gpucloud.yaml --yes
```

每台训练子机执行本机 worker：

```bash
gpucloud worker wait --task-file /data/gpucloud/task.yaml
gpucloud worker preflight --task-file /data/gpucloud/task.yaml
gpucloud worker dry-run --task-file /data/gpucloud/task.yaml
gpucloud worker start --task-file /data/gpucloud/task.yaml --yes
gpucloud worker status --job-id gpt-pretrain-001
gpucloud worker logs --job-id gpt-pretrain-001 --lines 100
gpucloud worker stop --job-id gpt-pretrain-001 --yes
```

`worker start` 和 `worker stop` 必须显式传 `--yes`。`worker dry-run` 永远不会启动训练进程。

## `gpucloud.yaml`

最小集群配置：

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

默认规则：

- `training.framework` 默认为 `megatron-lm`。
- 未显式配置 `training.command` 时，会生成 Megatron-LM `torchrun` 命令。
- `inference.engine` 默认为 `vllm`。
- `inference.port` 默认为 `8000`。
- 日志和 checkpoint 路径会从节点 workdir 推导。
- 远程执行默认要求先 dry-run。

## 分布式 Worker Task 文件

主节点调度器或用户脚本为每台机器生成一份 `gpucloud-worker-task.yaml`。所有文件共享 `job_id`、`nnodes`、`master_addr`、`master_port`，但每台机器的 `node_rank` 不同。

完整起步文件见 [gpucloud-worker-task.yaml.example](gpucloud-worker-task.yaml.example)。

rank 2 示例：

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
```

异构 GPU 策略：

- `reject`：GPU 型号或显存不满足要求时 preflight 失败。
- `warn`：报告风险，但其他硬检查通过时允许启动。
- `allow`：只记录环境差异，不阻止。

异构 GPU 下的 tensor parallel / pipeline parallel 拓扑需要通过 `training.extra_args` 或 `training.command_template` 显式配置。GPUCLOUD 第一版不会自动平衡不同 GPU 型号。

## Agent 工具

GPUCLOUD toolset 包含：

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

worker 工具必须显式提供 `task_file` 或 `job_id`，不会自动扫描任意配置文件，也不会调度其他机器。

## 开发

准备开发环境：

```bash
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
```

运行 GPUCLOUD 相关测试：

```bash
venv/bin/python -m pytest \
  tests/hermes_cli/test_gpucloud_worker_task.py \
  tests/hermes_cli/test_gpucloud_distributed.py \
  tests/hermes_cli/test_gpucloud_worker.py \
  tests/hermes_cli/test_gpucloud_train.py \
  tests/hermes_cli/test_gpucloud_inference.py -q
```

运行完整隔离测试：

```bash
scripts/run_tests.sh
```

## 安全边界

- 远程 SSH 训练和推理默认 dry-run。
- 本机 worker start/stop 必须传 `--yes`。
- task 文件不要写入 SSH 私钥或明文 token。
- worker plan 会脱敏疑似 secret 的环境变量。
- 日志只返回 tail，避免把完整日志塞进 Agent 上下文。
- `training.command_template` 视为来自可信主节点或用户的高权限输入。

## License

MIT。详见 [LICENSE](LICENSE)。
