# Megatron-LM Runtime Pitfalls (core_v0.11 + Python 3.10 + no Apex)

## 1. persist_layer_norm assertion in torch_norm.py

**Error:** `AssertionError: persist_layer_norm not supported by torch LayerNorm`

**Source:** `megatron/core/transformer/torch_norm.py` line with:
```python
assert not config.persist_layer_norm, "persist_layer_norm not supported by torch LayerNorm"
```

**Root cause:** Without Apex's FusedLayerNorm, Megatron falls back to `torch_norm.py` (WrappedTorchNorm). This module requires `persist_layer_norm=False`. The `--no-persist-layer-norm` flag should set this via `arguments.py`:
```python
kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
```

**Fix:** Add BOTH to YAML:
```yaml
training:
  safety_flags:
    no_persist_layer_norm: true
  extra_args:
    - --no-persist-layer-norm
```

**Debugging:** If assertion still fires, verify the flag appears in the actual launch command in GPUCLOUD logs. The YAML renderer may not translate safety_flags to extra_args automatically.

**Nuclear option — patch arguments.py directly:** If the flag is in the launch command but the assertion still fires (code path bypasses the flag), patch `arguments.py` line that maps the config:
```python
# In megatron/training/arguments.py, find:
kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
# Replace with:
kw_args['persist_layer_norm'] = False  # Force off when Apex unavailable
```
This is a guaranteed fix because it overrides the config regardless of which code path creates it.

## 2. pybind11 headers missing

**Error:** `fatal error: pybind11/pybind11.h: No such file or directory`

**Source:** Megatron-LM's `megatron/core/datasets/Makefile` compiles C++ helpers at runtime. Requires pybind11 headers and python3-config.

**Fix:**
```bash
<env>/bin/pip install pybind11[global]   # installs headers to system include path
ln -sf /usr/bin/python3.10-config /usr/local/bin/python3-config  # if missing
```

## 3. torchrun not found (exit 127)

**Error:** `bash: line 1: torchrun: command not found`

**Source:** GPUCLOUD wraps the launch command in bash. torchrun is in the venv bin, not system PATH.

**Fix:** Add PATH to YAML runtime.env:
```yaml
runtime:
  env:
    PATH: /path/to/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

## 4. Python 3.12+ requirement on main branch

**Error:** `ImportError: cannot import name 'override' from 'typing'`

**Source:** `megatron/training/models/hybrid.py` uses `from typing import override` (Python 3.12+).

**Fix:** After cloning, `git fetch --tags && git checkout core_v0.11`

## 5. preprocess_data.py JSONDecodeError

**Error:** `json.decoder.JSONDecodeError: Expecting value: line 2 column 1`

**Source:** Script calls `json.loads(line)` on each input line. Expects JSONL format, not raw text.

**Fix:** Convert raw text to JSONL:
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

## 6. --dataset-impl removed

**Error:** `preprocess_data.py: error: unrecognized arguments: --dataset-impl mmap`

**Source:** core_v0.11 preprocess_data.py doesn't have this flag. mmap is the default.

**Fix:** Remove `--dataset-impl mmap` from the command.

## 7. psutil missing for checkpoint saving

**Error:** `ModuleNotFoundError: No module named 'psutil'` → `CheckpointingException: Cannot import a default strategy for: ('save_sharded', 'torch_dist', 1)`

**Source:** Megatron's `torch_dist` checkpoint format needs psutil for memory monitoring during save.

**Fix:** `<env>/bin/pip install psutil`

## 7b. data-cache-path missing

**Error:** `Exception: Failed to write dataset materials to the data cache directory. Please supply a directory to which you have write access via the path_to_cache attribute`

**Source:** Megatron's `BlendedMegatronDatasetConfig` needs a writable cache directory for dataset index files. Without `--data-cache-path`, it tries to create a `cache/` subdirectory under the data_path prefix, which may fail.

**Fix:** Add to YAML extra_args:
```yaml
  extra_args:
    - --data-cache-path
    - /path/to/cache/data
```
Create the directory before training: `mkdir -p /path/to/cache/data`

## 8. Checkpoint conversion: no saver_transformers in core_v0.11

**Error:** `No module named 'saver_transformers'` / `transformers module is not a plugin`

**Source:** GPUCLOUD's auto-discovered conversion uses `--saver transformers`, but core_v0.11 only has `saver_core.py` and `saver_legacy.py`. The HF transformer saver was added in later versions.

**Additionally:** The auto-discovered command misses `--model-type GPT`:
```
convert.py: error: the following arguments are required: --model-type
```

**Fix — provide explicit conversion command_template in YAML:**
```yaml
conversion:
  command_template: "<env>/python <megatron>/tools/checkpoint/convert.py --model-type GPT --loader megatron --saver core --load-dir <checkpoint_dir> --save-dir <output_dir>"
  auto_discover: false
```

**If `--saver core` also fails** (torch.distributed not initialized), use a custom conversion script. The `torch_dist` checkpoint format requires `torch.distributed.init_process_group()` before loading. Pattern:

```python
import sys, os, torch
sys.path.insert(0, '<megatron_lm_dir>')
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29698'  # different from training port
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
torch.distributed.init_process_group(backend='gloo')

from megatron.core.dist_checkpointing import load_plain_tensors
state_dict = load_plain_tensors('<checkpoint_dir>/iter_NNNNNNN')

# Map Megatron keys to HF GPT-2 keys
# Megatron stores layers with first dim = n_layer (stacked)
# HF expects individual h.N.* keys
for k, v in state_dict.items():
    if k.startswith('decoder.layers.'):
        suffix = k.replace('decoder.layers.', '')
        for layer_idx in range(n_layer):
            layer_v = v[layer_idx]  # split along first dim
            # Map suffix to HF key...

torch.distributed.destroy_process_group()
```

**Key mapping (Megatron → HF GPT-2):**
| Megatron key suffix | HF key |
|---|---|
| `embedding.word_embeddings.weight` | `wte.weight` |
| `embedding.position_embeddings.weight` | `wpe.weight` |
| `decoder.final_layernorm.{weight,bias}` | `ln_f.{weight,bias}` |
| `self_attention.linear_qkv.weight` | `h.N.attn.c_attn.weight` |
| `self_attention.linear_qkv.bias` | `h.N.attn.c_attn.bias` |
| `self_attention.linear_proj.{weight,bias}` | `h.N.attn.c_proj.{weight,bias}` |
| `self_attention.linear_qkv.layer_norm_{weight,bias}` | `h.N.ln_1.{weight,bias}` |
| `mlp.linear_fc1.layer_norm_{weight,bias}` | `h.N.ln_2.{weight,bias}` |
| `mlp.linear_fc1.{weight,bias}` | `h.N.mlp.c_fc.{weight,bias}` |
| `mlp.linear_fc2.{weight,bias}` | `h.N.mlp.c_proj.{weight,bias}` |

**Note:** Layer tensors have shape `[n_layer, ...]` — split along dim 0 to get per-layer weights. Convert to float32 for HF compatibility: `v.float()`.

**CRITICAL — Weight Transposition for Conv1D:** HF GPT-2 uses `Conv1D` (not `nn.Linear`) for attention and MLP layers. Conv1D stores weights as `[in_features, out_features]`, while Megatron/nn.Linear uses `[out_features, in_features]`. You MUST transpose all linear layer weights:

```python
# For each layer's linear weights, transpose:
hf_state_dict[f'h.{i}.attn.c_attn.weight'] = megatron_qkv_weight.T   # [384,128] -> [128,384]
hf_state_dict[f'h.{i}.attn.c_proj.weight'] = megatron_proj_weight.T   # [128,128] -> [128,128]
hf_state_dict[f'h.{i}.mlp.c_fc.weight']    = megatron_fc1_weight.T    # [512,128] -> [128,512]
hf_state_dict[f'h.{i}.mlp.c_proj.weight']  = megatron_fc2_weight.T    # [128,512] -> [512,128]
```

Biases are NOT transposed (1D). Layer norms are NOT transposed (1D scale params).

Without transposition, vLLM raises: `AssertionError: param_data.shape == loaded_weight.shape` at `vllm/model_executor/layers/linear.py`.

**Required HF config.json fields for GPT-2:**
```json
{
  "architectures": ["GPT2LMHeadModel"],
  "model_type": "gpt2",
  "vocab_size": 50257,
  "n_positions": <seq_length>,
  "n_embd": <hidden_size>,
  "n_layer": <num_layers>,
  "n_head": <num_attention_heads>,
  "n_inner": <ffn_hidden_size>,
  "activation_function": "gelu_new",
  "bos_token_id": 50256,
  "eos_token_id": 50256
}
```

**Required files for vLLM to load the model:**
```
model_hf/
  config.json              # HF model config
  pytorch_model.bin        # model weights (torch.save)
  vocab.json               # GPT-2 tokenizer vocabulary
  merges.txt               # GPT-2 BPE merges
  tokenizer_config.json    # tokenizer config (model_max_length, tokenizer_class)
  generation_config.json   # generation parameters (optional but recommended)
```

vLLM checks for `config.json` + weight files (`pytorch_model.bin` or `model.safetensors`). Missing `tokenizer_config.json` may cause warnings but won't block loading.

## 9. vLLM installation time

`pip install vllm` downloads ~4GB of CUDA dependencies (cutlass, tilelang, nvidia-cutlass-dsl-libs, etc.). On slow networks this can take 30+ minutes. Install in background and poll.

If vLLM is not needed immediately, defer its installation until after training and conversion complete.

## 10. vLLM torch version conflict with CUDA driver

**Error:** `RuntimeError: The NVIDIA driver on your system is too old (found version 12060)`

**Source:** vLLM 0.22.0 (latest from PyPI as of June 2026) pulls in torch 2.11.0, which requires CUDA 13.0+ driver. Systems with CUDA 12.6 driver (nvidia 535.x) cannot use torch 2.11.0.

**Root cause:** `pip install vllm` without version pin installs the latest, which has `torch==2.11.0` as a dependency. This replaces whatever torch version you had installed. torch 2.11.0's CUDA 13 bindings fail with older drivers.

**Fix:** Pin vLLM to a version compatible with your torch/CUDA:
```bash
# For CUDA 12.6 driver + torch 2.5.x
<env>/bin/pip install --no-deps vllm==0.7.3
```

**IMPORTANT: vLLM 0.7.3 dependency chain (all must be pinned together):**

| Package | Version | Why |
|---------|---------|-----|
| torch | 2.5.1+cu121 | Matches CUDA 12.6 driver |
| vllm | 0.7.3 | Last version supporting torch 2.5.x |
| xformers | 0.0.28.post1 | Matches torch 2.5.1; latest xformers pulls torch 2.10+ |
| triton | 3.1.0 | vLLM 0.7.3 uses `triton.runtime.cache.default_cache_dir` removed in triton 3.6 |
| transformers | 4.48.3 | vLLM 0.7.3 imports `ProcessorMixin` removed in transformers 5.x |
| torchvision | 0.20.1+cu121 | Must match torch 2.5.1 |

Install sequence:
```bash
# 1. Install torch first (no-deps to avoid CUDA bloat)
<env>/bin/pip install --no-deps torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
<env>/bin/pip install networkx jinja2 sympy

# 2. Install vLLM (no-deps to avoid torch upgrade)
<env>/bin/pip install --no-deps vllm==0.7.3

# 3. Pin compatible versions
<env>/bin/pip install --no-deps xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu121
<env>/bin/pip install --no-deps triton==3.1.0
<env>/bin/pip install --no-deps transformers==4.48.3
<env>/bin/pip install --no-deps torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
<env>/bin/pip install --no-deps xgrammar==0.1.11
```

**If you DON'T use `--no-deps`**, pip will upgrade torch to 2.11.0 (vLLM 0.22.0's requirement), which fails with CUDA 12.6 driver.

**Prevention:** Check vLLM's torch requirement before installing:
```bash
pip index versions vllm  # list available versions
pip install --dry-run vllm==0.7.3  # check what it would install
```

**Alternative:** If you must use latest vLLM, the CUDA driver must be upgraded to support CUDA 13.0+. On cloud instances, this may require switching to a newer instance image.
