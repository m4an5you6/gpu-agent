# GPUCLOUD Agent

面向 GPU 集群的 CLI 优先 MLOps Agent：Megatron-LM 训练、checkpoint 管理、vLLM 推理、YAML 分布式 worker。

**v0.1 文档：** [release-v0.1.md](release-v0.1.md)

本 fork **已移除** 上游 Hermes 根目录的 `RELEASE_v*.md`，发布说明以 `release-v0.1.md` 为准。

```bash
gpucloud --help
gpucloud config validate --file gpucloud.yaml
```

Worker 示例：[gpucloud-worker-task.yaml.example](gpucloud-worker-task.yaml.example)

**许可：** [MIT](LICENSE)。引擎 fork 自 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)。

### 为什么仓库有数千个 commit？

`gpu-agent` 是从 `hermes-agent` **完整 fork/克隆** 出来的，Git **默认保留全部上游历史**（约 9.8k 条 commit）。真正属于 GPUCLOUD 的改动只在最近少量 commit（rebrand、worker、skills 等），不代表 v0.1 有上千次发布。

查看 GPUCLOUD 相关提交：`git log --oneline --grep=GPUCLOUD -i` 或 `git log --oneline 06fca998a..HEAD`。

若想要**无 Hermes 历史的干净仓库**，需要新建 orphan 分支或做 `git filter-repo` / 压成单次导入 — 可选，默认未做。
