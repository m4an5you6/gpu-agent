# GPUCLOUD Runtime Pitfalls

## Worker Preflight

Preflight runs before training and does not create the environment. It checks the declared Python runtime, torch/CUDA availability, data paths, Megatron entrypoints, and rendezvous ports. Terminal stages such as `training_failed`, `conversion_failed`, and `completed` prevent automatic reruns until the worker goal state is reset.

## Megatron-LM Version

Recent Megatron-LM `main` may require Python 3.12 because it imports `typing.override`. On Python 3.10/3.11 workers, prefer a compatible tag such as `core_v0.11` unless the task explicitly provides another tested version.

If cloning from GitHub is unreliable, use a mirror or a pre-provisioned source directory. Do not silently swap versions without recording the effective version in logs or validation output.

## PyTorch and CUDA

GPU cloud images often already include CUDA, cuDNN, NCCL, cublas, cufft, curand, nvrtc, and nvtx. When system libraries are present and compatible, installing torch wheels with `--no-deps` can avoid downloading large bundled NVIDIA dependencies.

Wheel filenames must retain full tags such as `torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl`; shortened names can be rejected by pip.

Always verify with the actual worker Python:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

## Megatron-LM Data Preparation

Megatron `preprocess_data.py` expects JSONL, not raw text. Convert each record to a JSON object and pass the matching text key, for example `--json-keys text`.

Some versions produce files as `<prefix>_text_document.bin` and `<prefix>_text_document.idx`; training `--data-path` expects the prefix used by the actual `.bin/.idx` files. Align the prefix or rename files before launch.

If C++ dataset helpers fail to compile, install `pybind11[global]` and ensure `python3-config` is available. If checkpoint saving fails with missing memory-monitor dependencies, install `psutil`.

## No-Apex Megatron Flags

When Apex fused layer norm is unavailable, ensure training disables persistent layer norm and uses local implementations where needed:

```yaml
training:
  extra_args:
    - --no-persist-layer-norm
    - --transformer-impl
    - local
    - --data-cache-path
    - /root/gpufree-data/gpucloud-cache/megatron
```

Create cache directories before training.

## Checkpoint Conversion

Megatron checkpoint conversion is version-specific. Older Megatron tags may lack a Transformers/HF saver and may require `--model-type GPT`. GPUCLOUD should run an explicit `conversion.command_template` only when the task provides it or when deterministic local discovery can prove a compatible converter exists.

For GPT-2 HF conversion, remember that HF GPT-2 uses `Conv1D` weight layout. Attention and MLP linear weights from Megatron generally need transposition before saving to HF format. Missing or wrongly shaped HF files can make vLLM fail during load.

Minimum HF/vLLM-loadable GPT-2 directory markers:

```text
config.json
pytorch_model.bin or model.safetensors
vocab.json
merges.txt
tokenizer_config.json
```

## vLLM Compatibility

Unpinned `pip install vllm` can upgrade torch to a build requiring a newer CUDA driver. Keep torch, vLLM, xformers, triton, torchvision, and transformers compatible with the node driver. For CUDA 12.x worker images, prefer a known-good pinned set instead of accepting latest packages blindly.

Install vLLM after training if disk or network pressure matters; training preflight should not fail just because inference packages are not installed yet unless the task asks for immediate inference.
