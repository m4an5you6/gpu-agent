# GPUCLOUD Agent

CLI-first ML operations agent for GPU clusters: Megatron-LM training, checkpoint management, vLLM inference, and YAML-driven distributed workers.

**v0.1 documentation:** [release-v0.1.md](release-v0.1.md)

GPUCLOUD upstream release files (`RELEASE_v*.md`) are **not** used in this fork.

```bash
gpucloud --help
gpucloud config validate --file gpucloud.yaml
```

Worker task example: [gpucloud-worker-task.yaml.example](gpucloud-worker-task.yaml.example)

**License:** [MIT](LICENSE). Engine forked from [NousResearch/gpucloud-agent](https://github.com/NousResearch/gpucloud-agent).

### Why does this repo show thousands of commits?

`gpu-agent` was created from a **full git clone/fork** of `gpucloud-agent`, so **all upstream commit history is preserved** (~9.8k commits). Only a handful of commits at the tip are GPUCLOUD-specific (rebrand, worker runtime, skills). That is normal for a fork; it does not mean v0.1 shipped 9k changes.

To see only GPUCLOUD-era commits: `git log --oneline --grep=GPUCLOUD -i` or `git log --oneline 06fca998a..HEAD`.

To start a **clean history** repo (one commit, no GPUCLOUD log), you would need a new orphan branch or `git filter-repo` / squash import — optional, not done by default.
