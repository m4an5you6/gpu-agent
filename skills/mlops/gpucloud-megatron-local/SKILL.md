---
name: gpucloud-megatron-local
description: Run Megatron-LM training + HF conversion + vLLM inference via GPUCLOUD YAML on local single-GPU worker.
tags: [megatron, gpucloud, training, vllm, gpt2]
triggers:
  - "megatron training on local GPU"
  - "GPUCLOUD worker goal megatron"
  - "train and infer with megatron"
---

# GPUCLOUD Megatron-LM Local Worker Workflow

## Prerequisites
- GPU with CUDA 12.1+ driver (Tesla T4 or better)
- Python 3.10+ with venv
- Network access to PyPI mirror (Tsinghua: pypi.tuna.tsinghua.edu.cn)
- For HF datasets: use hf-mirror.com (huggingface.co is blocked on 算力自由)

## Key Pitfalls

### 1. Megatron-LM Version Compatibility
- **Latest main** requires Python 3.12+ (`typing.override` import)
- Use tag **core_v0.11** for Python 3.10 compatibility
- Clone from gitee.com mirror: `git clone --depth 1 https://gitee.com/mirrors/Megatron-LM.git`
- Then `git fetch --tags && git checkout core_v0.11`

### 2. PyTorch Installation
- System CUDA libs (cudnn, nccl, cublas, cufft, curand, cusolver, cusparse, nvrtc, nvtx) are pre-installed
- Install torch with `--no-deps` to avoid downloading 1.5GB of CUDA libs
- `pip install --no-deps /path/to/torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl`
- Then install small deps: `pip install networkx jinja2 sympy`
- Wheel filename must follow pattern: `name-version+cuNN-cpXY-cpXY-platform.whl`

### 3. Data Preprocessing
- `preprocess_data.py` expects **JSONL** input, not raw text
- Convert raw text to JSONL: each line as `{"text": "..."}` 
- Use `--json-keys text` flag
- Files created as `{prefix}_text_document.{bin,idx}` but Megatron uses `{prefix}.{bin,idx}`
- **Rename files** after preprocessing, or adjust `--data-path` to include `_text_document`
- Megatron uses `--data-path` as PREFIX, appending `.idx` and `.bin`
- Create a DIRECTORY at data_path (not a file) so Megatron can create `data_path/cache` subdir

### 4. Missing Dependencies
- `pip install einops pybind11 pybind11[global] psutil`
- `apt install python3.10-dev` + symlink `python3-config`
- Symlink `torchrun` to `/usr/local/bin/torchrun` or add venv bin to PATH in YAML env

### 5. Megatron Training Flags (no Apex)
- `--no-persist-layer-norm` — required when Apex is not installed
- `--data-cache-path <dir>` — required for dataset caching
- `--transformer-impl local` — use PyTorch fallback implementations
- May need to patch `arguments.py` line: `kw_args['persist_layer_norm'] = False`

### 6. Checkpoint Conversion (core_v0.11)
- No `saver_transformers.py` in core_v0.11
- `convert.py` needs `--model-type GPT` flag
- GPUCLOUD auto-discovery misses this — provide explicit `command_template`
- Or write custom conversion script using `load_plain_tensors()` with `torch.distributed.init_process_group(backend='gloo')`
- Key mapping: `decoder.layers.{X}.self_attention.linear_qkv` -> `h.{X}.attn.c_attn`
- Layer norms stored as `self_attention.linear_qkv.layer_norm_weight` -> `h.{X}.ln_1.weight`

### 7. vLLM Compatibility
- vLLM 0.22.0 (latest) installs torch 2.11.0 requiring CUDA 13+ driver
- For CUDA 12.6 (driver 535.x): use `vllm==0.7.3` with torch 2.5.1
- Must pin ALL deps with `--no-deps` to prevent torch upgrade:
  - `vllm==0.7.3`, `xformers==0.0.28.post1`, `triton==3.1.0`, `transformers==4.48.3`
- See `gpucloud-worker-setup` skill references for full dependency chain

### 8. HF GPT-2 Weight Transposition (Conv1D)
- HF GPT-2 uses `Conv1D` (not `nn.Linear`) — weights stored as `[in, out]` not `[out, in]`
- All Megatron linear weights MUST be transposed before saving:
  - `attn.c_attn.weight`: `.T` (QKV concat)
  - `attn.c_proj.weight`: `.T`
  - `mlp.c_fc.weight`: `.T`
  - `mlp.c_proj.weight`: `.T`
- Biases and layer norms: NOT transposed (1D)
- Without this, vLLM raises `AssertionError: param_data.shape == loaded_weight.shape`

### 8. GPUCLOUD Preflight Checks
- `data_path_readable`: checks `data_path.exists()` — must be file or directory
- `megatron_entrypoint`: checks `pretrain_gpt.py` exists at `megatron_lm_dir`
- `torch_import`: runs python from `runtime.python` path
- Goal state file at `~/.gpucloud/gpucloud/worker_goal_runs/{job_id}.json`
- Terminal stages (`training_failed`, `conversion_failed`, `completed`) block re-runs
- Reset by overwriting state file with `"stage": "preflight"`

## YAML Environment Variables
```yaml
runtime:
  env:
    PATH: /path/to/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    CUDA_DEVICE_MAX_CONNECTIONS: "1"
    NCCL_DEBUG: WARN
    TOKENIZERS_PARALLELISM: "false"
```
