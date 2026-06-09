# vLLM Runtime and Model Readiness

## Runtime Compatibility

vLLM can upgrade torch if dependencies are not pinned. Check the node driver, installed CUDA runtime, torch version, and vLLM version together. If the worker already has a known-good torch build for training, avoid unpinned vLLM installs that replace it.

Useful checks:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
nvidia-smi
```

For CUDA 12.x images, a pinned vLLM/torch/xformers/triton/transformers set is usually safer than latest packages.

## HF/vLLM Directory Markers

Common minimum files:

```text
config.json
pytorch_model.bin or model.safetensors
tokenizer.json or tokenizer.model or vocab.json/merges.txt
tokenizer_config.json
```

Validate `config.json` before launch. It should contain `model_type` and architecture fields compatible with vLLM. A directory with only Megatron checkpoint shards is not vLLM loadable.

## Conversion Safety

Megatron checkpoint conversion depends on Megatron version, model architecture, tensor/pipeline parallel settings, tokenizer files, and target HF format. If GPUCLOUD cannot prove a converter is compatible, fail with a diagnostic and ask for `conversion.command_template` or a preconverted `inference.model_path`.

For GPT-2 style conversion, verify HF Conv1D weight layout. Incorrect transposition can produce shape assertions during vLLM model load.

## Health Check

After local vLLM starts, poll a local health endpoint before reporting success. If health fails, return:

- pid status
- exit code if present
- log tail
- resolved model path
- host and port
- package versions when available
