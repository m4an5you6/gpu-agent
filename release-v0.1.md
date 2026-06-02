# GPUCLOUD Agent — Release v0.1

**Date:** 2026-06-02  
**Repository:** [m4an5you6/gpu-agent](https://github.com/m4an5you6/gpu-agent)  
**Base engine:** Hermes Agent (MIT), product surface rebranded to GPUCLOUD

---

## Highlights

- **Product identity:** CLI entry `gpucloud`; README and docs oriented to ML cluster / worker workflows.
- **Training:** Megatron-LM via `gpucloud train` (SSH cluster path) and `gpucloud worker` (per-node task file).
- **Inference:** vLLM lifecycle (`infer dry-run | start | status | health | stop`).
- **Goal workflow:** `/goal` routes to local worker backend when `gpucloud-worker-task.yaml` is present (`gpucloud_worker_goal_run`).
- **Distributed worker:** Per-machine YAML with shared `job_id` / rendezvous; GPUCLOUD supervises local processes only (NCCL/Megatron handle training comms).
- **Skills:** `gpucloud-worker-setup`, `gpucloud-megatron-local`, updates to `huggingface-hub` under `skills/mlops/`.

---

## Removed (documentation cleanup)

This fork drops upstream Hermes **root / plan markdown** that does not apply to GPUCLOUD. Runtime code (`hermes_*.py`, `hermes_cli/`, package name `hermes-agent`) is unchanged for compatibility.

**Replacement for releases:** this file (`release-v0.1.md`).  
**Replacement for product overview:** [README.md](README.md) / [README.zh-CN.md](README.zh-CN.md).

### Hermes release notes (14 files)

| Removed file | Notes |
| --- | --- |
| `RELEASE_v0.2.0.md` | Upstream Hermes v0.2.0 notes |
| `RELEASE_v0.3.0.md` | v0.3.0 |
| `RELEASE_v0.4.0.md` | v0.4.0 |
| `RELEASE_v0.5.0.md` | v0.5.0 |
| `RELEASE_v0.6.0.md` | v0.6.0 |
| `RELEASE_v0.7.0.md` | v0.7.0 |
| `RELEASE_v0.8.0.md` | v0.8.0 |
| `RELEASE_v0.9.0.md` | v0.9.0 |
| `RELEASE_v0.10.0.md` | v0.10.0 |
| `RELEASE_v0.11.0.md` | v0.11.0 |
| `RELEASE_v0.12.0.md` | v0.12.0 |
| `RELEASE_v0.13.0.md` | v0.13.0 |
| `RELEASE_v0.14.0.md` | v0.14.0 |
| `RELEASE_v0.15.0.md` | v0.15.0 |
| `RELEASE_v0.15.1.md` | v0.15.1 patch notes |

Future GPUCLOUD versions: add `release-v0.2.md`, etc., at repo root (same pattern as this file).

### Upstream contributor / policy docs (3 files)

| Removed file | Former role | Where to look now |
| --- | --- | --- |
| `AGENTS.md` | Hermes developer guide for coding agents | GPUCLOUD scope in README; upstream copy still on [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent/blob/main/AGENTS.md) |
| `CONTRIBUTING.md` | Nous dependency-pinning & PR policy | Not maintained in this fork |
| `SECURITY.md` | Upstream security disclosure policy | Report engine issues upstream; follow GPUCLOUD safety notes in README / worker task examples |

**Note:** Projects may still use a **local** `AGENTS.md` in their own repo; Hermes/GPUCLOUD can auto-inject that at runtime. Only the **fork’s root** `AGENTS.md` was removed.

### Marketing / internal plans (6 files)

| Removed file |
| --- |
| `hermes-already-has-routines.md` |
| `.plans/openai-api-server.md` |
| `.plans/streaming-support.md` |
| `plans/gemini-oauth-provider.md` |
| `docs/plans/2026-05-02-telegram-dm-user-managed-multisession-topics.md` |
| `docs/plans/2026-05-07-s6-overlay-dynamic-subagent-gateways.md` |
| `docs/plans/2026-05-15-acp-zed-edit-approval-diffs.md` |

### README changes

- Root README rewritten for GPUCLOUD (no longer a full duplicate of upstream Hermes README).
- Links only to `release-v0.1.md` for v0.1 docs (no separate `gpucloud-v0.1.md` in the release tree).

### Not removed in v0.1

- `website/` — Hermes documentation site (large; may be trimmed in a later release).
- Python modules and CLI named `hermes_*` / `hermes-agent` package.
- `skills/`, `optional-skills/`, GPUCLOUD-specific skills under `skills/mlops/gpucloud-*`.

---

## Breaking / product vs upstream Hermes

- Default workflow no longer centers on messaging gateway, dashboard, or general-purpose `hermes` UX.
- Use **`gpucloud`** as the primary CLI; `hermes` entry may still exist but is outside the default GPUCLOUD path.

---

## Added

| Area | Change |
| --- | --- |
| CLI | `gpucloud` command; cluster check, train, checkpoint, infer, worker subcommands |
| Config | `gpucloud.yaml`, `gpucloud-worker-task.yaml.example` |
| Worker runtime | Preflight, dry-run, start/stop with `--yes`, goal state under `~/.hermes/gpucloud/worker_goal_runs/` |
| Agent tools | GPUCLOUD toolset (`gpucloud_worker_goal_run`, `gpucloud_goal_prepare`, …) |
| Tests | `tests/hermes_cli/test_gpucloud_*.py` |
| Docs | `release-v0.1.md` (this file); `README.md` / `README.zh-CN.md` updated; **24** upstream markdown files removed (see **Removed** above) |

---

## Changed

- Phase 1 rebrand from Hermes product surface to GPUCLOUD (`feat(cli): rebrand phase 1 to GPUCLOUD`).
- Goal dry-run plans clarified for GPUCLOUD workflows.
- GitHub Actions deploy/docs triggers disabled on this fork where appropriate.

---

## Known limitations (v0.1)

- Worker preflight expects environment (venv, PyTorch, data, Megatron) to exist before `gpucloud_worker_goal_run`; agent often must bootstrap via skills first.
- Coordinator must distribute task files; GPUCLOUD does not fan out SSH to all nodes in v0.1 worker runtime.
- `training.command_template` and conversion templates are trusted coordinator input.
- Python package name remains `hermes-agent` for module-path compatibility with upstream.

---

## Upgrade / install

```bash
git clone https://github.com/m4an5you6/gpu-agent.git
cd gpu-agent
uv venv venv --python 3.11 && source venv/bin/activate
uv pip install -e ".[all,dev]"
gpucloud --help
```

See [README.md](README.md) and `gpucloud --help` for CLI entry points; worker example in [gpucloud-worker-task.yaml.example](gpucloud-worker-task.yaml.example).

---

## Commits in this release line (GPUCLOUD fork)

Approximate fork-specific history on `main` since rebrand:

- `06fca998a` — rebrand phase 1 to GPUCLOUD
- `4873cf24e` — phase 9 goal workflow
- `ae75b688e` — phase 8 inference service
- `fb93d7bb4` — goal dry-run plan clarity
- `aef5c0462` — distributed Megatron worker runtime
- `ee289aa3f` — route goals through local worker backend
- `059834234` — disable automatic GitHub Actions triggers
- `1078fd49c` — GPT-2 worker skills / session artifacts

(Full `git log` still includes upstream Hermes history — see README FAQ below.)
