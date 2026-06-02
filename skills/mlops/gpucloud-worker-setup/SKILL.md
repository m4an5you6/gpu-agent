---
name: gpucloud-worker-setup
description: "Set up GPUCLOUD worker-local environment for Megatron-LM training + vLLM inference. Covers preflight pitfalls, Python venv, PyTorch install, data prep, and the full YAML-driven workflow."
version: 1.2.0
author: hermes
tags: [gpucloud, megatron, training, vllm, worker, mlops, environment-setup]
platforms: [linux]
triggers:
  - "setup gpucloud worker"
  - "megatron-lm environment"
  - "worker goal run"
  - "preflight failed"
  - "train_and_infer"
---

# GPUCLOUD Worker-Local Environment Setup

## When to Use

When a `gpucloud_worker_goal_run` call fails at the **preflight** stage because the environment (venv, packages, data, tokenizer, Megatron-LM) does not exist yet. The YAML `goal.agent_owns` field lists what the agent must set up before GPUCLOUD can run.

## Critical Pitfall: Preflight Runs Before Environment

**GPUCLOUD does NOT set up the environment for you.** The `gpucloud_worker_goal_run` tool immediately runs preflight checks, which require:

- Python venv with PyTorch+CUDA importable from `runtime.python`
- `torch.cuda.is_available()` returning True
- `data_path` directory exists and is readable
- Megatron-LM `entrypoint` file (e.g. `pretrain_gpt.py`) exists
- Rendezvous port bindable

If any of these fail, preflight fails and the workflow stalls at `training_failed`. **You must set up the environment BEFORE calling `gpucloud_worker_goal_run`.**

## Setup Sequence

Execute in this order. Steps 1-5 are prerequisites for preflight to pass.

### Step 1: Create Python Virtual Environment

```bash
# Check python version (YAML may request 3.11 but 3.10 is fine)
python3 --version

# Install venv package if needed (Debian/Ubuntu)
apt-get install -y python3.x-venv

# Create venv at the YAML's env_dir path
python3 -m venv <env_dir>
# e.g.: python3 -m venv /root/gpufree-data/gpucloud-runs/<job-id>/env
```

### Step 2: Install PyTorch with CUDA

```bash
# Upgrade pip first
<env>/bin/pip install --upgrade pip

# Install torch+torchvision from PyTorch index
# NOTE: This is ~2GB, can take 10-15 min on slow networks
<env>/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Pitfall: `tail` pipe causes output buffering.** Do NOT pipe pip output through `tail` — it blocks all output until pip exits. Run pip directly or redirect to file.

**Pitfall: Incomplete wheel in pip cache.** If a previous install was killed, pip may have a partial wheel cached at `/tmp/pip-unpack-*/`. The cached wheel will be rejected as "invalid". Fix: `pip install --no-cache-dir` or remove the tmp dir.

**Pitfall: Slow downloads from download.pytorch.org.** On Chinese networks, use `curl` to download the wheel directly, then install from file:
```bash
curl -L -o /tmp/torch.whl "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
<env>/bin/pip install /tmp/torch.whl
```

**Pitfall: Wheel filename rejected by pip.** If you `curl` the wheel with a short name like `torch-2.5.1+cu121.whl`, pip rejects it with `Invalid wheel filename (wrong number of parts)`. The filename MUST include the full platform tag: `torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl`. Rename before installing:
```bash
cp /tmp/torch-2.5.1+cu121.whl /tmp/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl
```

**Pitfall: CUDA dependency download stalls.** `pip install torch` from the PyTorch index downloads ~1.5GB of bundled NVIDIA packages (nccl, cudnn, nvtx, triton, cublas, etc.) even when the system already has them. On GPU cloud instances with CUDA pre-installed (check with `ldconfig -p | grep cudnn`), use `--no-deps` to skip:
```bash
<env>/bin/pip install --no-deps /tmp/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl
# Then install only the lightweight missing deps:
<env>/bin/pip install networkx jinja2 sympy
# And the NVIDIA Python bindings if torch complains at runtime:
<env>/bin/pip install nvidia-nccl-cu12==2.21.5 nvidia-nvtx-cu12==12.1.105 triton==3.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```
This cuts install time from 15+ min to ~1 min.

### Step 3: Install Other Dependencies

```bash
<env>/bin/pip install transformers datasets einops
```

For vLLM (inference stage):
```bash
<env>/bin/pip install vllm
```

### Step 4: Get Megatron-LM

Option A — Git clone (GitHub):
```bash
git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git <megatron_lm_dir>
```

Option A2 — Git clone (Gitee mirror, for when GitHub is blocked):
```bash
git clone --depth 1 https://gitee.com/mirrors/Megatron-LM.git <megatron_lm_dir>
```

**After cloning, always check out a Python 3.10-compatible tag:**
```bash
cd <megatron_lm_dir> && git fetch --tags && git checkout core_v0.11
```
The `main` branch requires Python 3.12+. See pitfall in Step 5c.

Option B — pip install (different path structure, may not have `pretrain_gpt.py`):
```bash
<env>/bin/pip install megatron-core
```
Note: pip install may place files in a different path structure than git clone. The preflight checks for `<megatron_lm_dir>/pretrain_gpt.py`.

**Pitfall: GitHub blocked.** In some network environments (Chinese cloud), GitHub is unreachable. Use a mirror or pip install instead.

### Step 5: Download and Prepare Data

#### 5a. Download Tokenizer Files (GPT-2)

```bash
mkdir -p <data_dir>/tokenizer
# From HuggingFace (or hf-mirror.com if HF is blocked)
curl -L -o <data_dir>/tokenizer/vocab.json \
  "https://hf-mirror.com/openai-community/gpt2/resolve/main/vocab.json"
curl -L -o <data_dir>/tokenizer/merges.txt \
  "https://hf-mirror.com/openai-community/gpt2/resolve/main/merges.txt"
```

#### 5b. Download WikiText-2 Dataset

The original S3 source (`research.metamind.io.s3.amazonaws.com`) is broken (PermanentRedirect + SSL cert mismatch). Use HuggingFace mirror instead.

**Method A — Parquet download + pandas conversion (fastest, no HF library needed):**
```bash
mkdir -p <data_dir>/raw
cd <data_dir>/raw
# Download parquet files from HF mirror
curl -L -o train.parquet "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet"
curl -L -o validation.parquet "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/validation-00000-of-00001.parquet"
curl -L -o test.parquet "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"

# Convert parquet to raw text
<env>/bin/python -c "
import pandas as pd, os
base = '<data_dir>/raw'
for pq, out in [('train.parquet','wiki.train.raw'), ('validation.parquet','wiki.valid.raw'), ('test.parquet','wiki.test.raw')]:
    df = pd.read_parquet(os.path.join(base, pq))
    with open(os.path.join(base, out), 'w') as f:
        for text in df['text']:
            f.write(text + '\n')
"
```

**Method B — datasets library with HF mirror:**
```bash
HF_ENDPOINT=https://hf-mirror.com <env>/bin/python -c "
import os; os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
for split, name in [('train','wiki.train.raw'),('validation','wiki.valid.raw'),('test','wiki.test.raw')]:
    with open(f'<data_dir>/raw/{name}', 'w') as f:
        for item in ds[split]:
            f.write(item['text'] + '\n')
"
```

**HF Mirror API — list files for any dataset:**
```bash
curl -s "https://hf-mirror.com/api/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1"
```

**Pitfall: preprocess_data.py requires torch.** Megatron-LM's `tools/preprocess_data.py` imports `megatron.core` which imports `torch`. Install torch BEFORE running tokenization, not after.

**Pitfall: Latest Megatron-LM main requires Python 3.12+.** The `main` branch uses `from typing import override` which only exists in Python 3.12+. On Python 3.10/3.11, you get `ImportError: cannot import name 'override' from 'typing'`. **Fix:** check out an older tag after cloning:
```bash
cd <megatron_lm_dir>
git fetch --tags
git checkout core_v0.11   # last tag compatible with Python 3.10
```
Tags `core_v0.11`, `core_v0.12.x` work with Python 3.10. Tags after `core_v0.13` may require Python 3.12+.

**Pitfall: preprocess_data.py expects JSONL, not raw text.** The script calls `json.loads(line)` on each line. If you pass raw text (one sentence per line), you get `JSONDecodeError: Expecting value`. Convert to JSONL first:
```python
import json
for inp, out in [('wiki.train.raw','wiki.train.jsonl'), ...]:
    with open(inp) as fin, open(out, 'w') as fout:
        for line in fin:
            text = line.strip()
            if text:
                fout.write(json.dumps({'text': text}) + '\n')
```
Then pass `--json-keys text` to preprocess_data.py.

**Pitfall: `--dataset-impl mmap` removed in newer tags.** `core_v0.11` preprocess_data.py does not accept `--dataset-impl`. Remove it from the command. The mmap format is the default.

**Pitfall: pybind11 missing for C++ dataset helpers.** Megatron-LM compiles C++ helpers at runtime. If pybind11 headers are not found, you get `fatal error: pybind11/pybind11.h: No such file or directory`. Fix:
```bash
<env>/bin/pip install pybind11[global]   # installs headers system-wide
```
Also ensure `python3-config` exists:
```bash
ln -sf /usr/bin/python3.10-config /usr/local/bin/python3-config
```

#### 5c. Tokenize for Megatron-LM

Use Megatron-LM's built-in preprocessing:
```bash
<env>/bin/python <megatron_lm_dir>/tools/preprocess_data.py \
  --input <data_dir>/raw/train.jsonl \
  --output <data_prefix>_text_document \
  --dataset-impl mmap \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file <data_dir>/tokenizer/vocab.json \
  --merge-file <data_dir>/tokenizer/merges.txt
```

Or use the `preprocess_data.py` with JSONL text field extraction.

**Output naming convention:** `preprocess_data.py --output-prefix <prefix> --json-keys text` creates:
```
<prefix>_text_document.bin
<prefix>_text_document.idx
```
But Megatron's `--data-path <prefix>` expects:
```
<prefix>.bin
<prefix>.idx
```
**You must rename the files** after preprocessing:
```bash
mv <prefix>_text_document.bin <prefix>.bin
mv <prefix>_text_document.idx <prefix>.idx
```

### Step 6: Fix Runtime Pitfalls Before Calling Goal Run

#### torchrun not in PATH

GPUCLOUD launches training via `torchrun`, which lives in the venv's `bin/` directory. If the venv bin is not in PATH, training fails with `torchrun: command not found` (exit 127).

**Fix option A — Add PATH to YAML runtime.env:**
```yaml
runtime:
  env:
    PATH: /path/to/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

**Fix option B — Symlink torchrun globally:**
```bash
ln -sf /path/to/env/bin/torchrun /usr/local/bin/torchrun
```

#### persist_layer_norm assertion failure (no Apex)

Without NVIDIA Apex installed, Megatron-LM uses `torch_norm.py` as fallback, which asserts:
```python
assert not config.persist_layer_norm, "persist_layer_norm not supported by torch LayerNorm"
```

The `--no-persist-layer-norm` flag SHOULD set this to False, but some code paths may still set it True.

**Fix:** Add both to YAML:
```yaml
training:
  safety_flags:
    no_persist_layer_norm: true
  extra_args:
    - --no-persist-layer-norm
```

If the assertion still fires, the `core_v0.11` tag's `arguments.py` maps:
```python
kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
```
Verify the flag is actually in the launch command in the GPUCLOUD logs. If the YAML renderer doesn't pass it, add it explicitly to `extra_args`.

#### data_path prefix vs preflight existence check

**Fundamental mismatch:** The preflight checks `data_path.exists()` (file or directory must exist), but Megatron-LM uses `data_path` as a **prefix** — it constructs `{data_path}.idx` and `{data_path}.bin`. Additionally, Megatron tries to create `{data_path}/cache` as a directory.

**Safest approach — make data_path a DIRECTORY:**
1. Create `data_path` as a DIRECTORY (satisfies preflight `exists()` check)
2. Place `.idx` and `.bin` files as SIBLINGS in the parent directory:
   ```
   data/megatron/wikitext2/       ← directory (data_path points here)
   data/megatron/wikitext2.idx    ← Megatron finds this via prefix + ".idx"
   data/megatron/wikitext2.bin    ← Megatron finds this via prefix + ".bin"
   ```
3. Megatron creates cache inside the directory: `data/megatron/wikitext2/cache/`

This works because:
- Preflight sees `wikitext2` as an existing directory → passes
- Megatron constructs `wikitext2.idx` and `wikitext2.bin` in the parent → finds files
- Cache creation at `wikitext2/cache` → works because `wikitext2` is a directory

**Do NOT create a placeholder FILE at data_path.** If `data_path` is a file, Megatron fails with `NotADirectoryError: [Errno 20] Not a directory: '...wikitext2/cache'` when trying to create the cache directory.

Note: The preprocess_data.py `--output-prefix` creates files as `{prefix}_text_document.{bin,idx}`. You must RENAME them to `{prefix}.{bin,idx}` to match what Megatron expects when `--data-path` is `{prefix}`.

### Step 7: Call gpucloud_worker_goal_run

Once all preflight requirements are met:
```
gpucloud_worker_goal_run()
```

The tool will:
1. Pass preflight → dry-run → auto-start training
2. On subsequent calls: poll training status
3. After training: convert checkpoint to HF format
4. Start vLLM → health check → completed

**Pitfall: Conversion requires explicit command_template.** GPUCLOUD's auto-discovered conversion misses `--model-type GPT` and `--saver` may not exist in core_v0.11. Always provide an explicit conversion command in the YAML:
```yaml
conversion:
  command_template: "<env>/python <megatron>/tools/checkpoint/convert.py --model-type GPT --loader megatron --saver transformers --load-dir <checkpoint_dir> --save-dir <output_dir>"
  auto_discover: false
```
If `--saver transformers` doesn't exist (core_v0.11), use a custom conversion script. See `references/megatron-runtime-pitfalls.md` section 8 for the full pattern with key mapping table.

## Network Environment Notes

### Chinese Cloud / 算力自由 Instances

| Service | Direct Access | Mirror/Alternative |
|---------|--------------|-------------------|
| PyPI (pypi.org) | Via tuna.tsinghua.edu.cn | pip.conf pre-configured |
| HuggingFace (huggingface.co) | Often blocked | hf-mirror.com (API + files) |
| GitHub | Often blocked | gitee.com/mirrors/ or pip install |
| download.pytorch.org | Works but slow (~1.5MB/s) | curl direct download |
| research.metamind.io S3 | Broken (PermanentRedirect + SSL) | Use HF mirror instead |
| NVIDIA CUDA repos | May need proxy | — |

Check pip config: `cat /etc/pip.conf` — often pre-set to `pypi.tuna.tsinghua.edu.cn`.

### HF_ENDPOINT for Python

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Must be set BEFORE importing datasets/transformers
```

Or inline: `HF_ENDPOINT=https://hf-mirror.com python script.py`

## Preflight Check Details

**Complete conversion script:** See `references/megatron_to_hf_conversion.py` for a ready-to-use script that handles the full Megatron → HF GPT-2 conversion with proper key mapping and torch.distributed initialization.

Source: `/usr/local/lib/hermes-agent/hermes_cli/gpucloud_worker.py` `run_worker_preflight()`

| Check | What It Validates |
|-------|-------------------|
| gpu_count | nvidia-smi detects >= required GPUs |
| gpu_vram | Each GPU meets min_vram_gb (if set) |
| workdir_writable | workdir exists and is writable |
| data_path_readable | data_path directory exists |
| checkpoint_dir_writable | checkpoint_dir exists and is writable |
| log_dir_writable | log_dir exists and is writable |
| megatron_entrypoint | entrypoint .py file exists |
| torch_import | `<python> -c "import torch"` succeeds |
| torch_cuda | `torch.cuda.is_available()` is True |
| torch_distributed | `torch.distributed.is_available()` is True |
| torch_nccl | NCCL backend available |
| rendezvous_port_bindable | master_addr:master_port is free (node_rank=0) |

All checks with severity="error" must pass. Warnings are informational.

## Re-running After Failure

The workflow state is persisted at `~/.hermes/gpucloud/worker_goal_runs/<job_id>.json`. Terminal stages (`training_failed`, `conversion_failed`, `completed`) cause `gpucloud_worker_goal_run` to return immediately without re-running.

**Reset by writing a fresh state file** (preferred over deleting — some environments block `rm`):
```python
# Write via write_file tool, overwriting the existing JSON:
{
  "workflow_id": "worker-goal-<job_id>",
  "job_id": "<job_id>",
  "task_file": "<path>",
  "backend": "worker_local",
  "intent": "train_and_infer",
  "stage": "preflight",
  "status": "active",
  "next_action": "run local preflight",
  "created_at": <timestamp>,
  "updated_at": <timestamp>,
  "train": {},
  "conversion": {},
  "inference": {},
  "logs": {},
  "last_error": null
}
```

Then call `gpucloud_worker_goal_run()` again after fixing the environment.

**Important:** The goal state file is rewritten by GPUCLOUD on every call. External edits are detected and warned about but do not prevent execution.

## YAML Structure Reference

See `references/preflight-and-network.md` for the full preflight source code walkthrough, workflow state machine diagram, and network connectivity test results for Chinese cloud instances.

See `references/megatron-runtime-pitfalls.md` for runtime errors: persist_layer_norm assertion, pybind11 missing, torchrun PATH, Python version compatibility, preprocess_data format issues, **checkpoint conversion** (no saver_transformers in core_v0.11, custom HF conversion script pattern with key mapping table and Conv1D transposition).

See `references/megatron_to_hf_gpt2_conversion.py` for a complete, ready-to-run conversion script with configurable paths and model parameters.

Key fields the agent needs from the task YAML:

```yaml
goal:
  mode: train_and_infer       # or "train"
  auto_execute: true           # auto-start training after preflight
  agent_owns:                  # what the agent must set up
    - prepare_isolated_environment
    - fetch_or_install_megatron_lm
    - prepare_wikitext2_raw
    - tokenize_for_megatron
    - ...

environment:
  env_dir: /path/to/venv
  install:
    torch: auto_cuda           # hint for torch install
    megatron_lm: auto

runtime:
  python: /path/to/env/bin/python
  megatron_lm_dir: /path/to/Megatron-LM

dataset:
  source: huggingface
  path: Salesforce/wikitext
  name: wikitext-2-raw-v1
  tokenizer:
    vocab_file: /path/to/vocab.json
    merge_file: /path/to/merges.txt

training:
  entrypoint: pretrain_gpt.py
  data_path: /path/to/data_prefix
  checkpoint_dir: /path/to/checkpoints
```
