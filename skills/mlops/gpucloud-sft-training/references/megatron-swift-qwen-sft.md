# Megatron-SWIFT and Qwen SFT

The deployment backend preset `qwen36-megatron-swift-sft-lora` configures Qwen SFT through Megatron-SWIFT. GPUCLOUD should represent this as structured training config.

## Runner Meaning

`megatron.backend=swift` means:

```yaml
training:
  framework: megatron-lm
  runner: swift_megatron
```

It does not mean SWIFT replaces GPUCLOUD orchestration or becomes a separate training framework.

## Qwen LoRA SFT Defaults

The deployment preset uses:

- model: `Qwen/Qwen3.6-35B-A3B`
- dataset: `swift/sharegpt:common-zh`
- `distributed: true`
- `hrl3d.enabled: true`
- `megatron.enabled: true`
- `megatron.backend: swift`
- `megatron.auto_setup: true`
- `swift.train_type: sft`
- `swift.max_length: 1024`
- `swift.train_samples: 200`

Default LoRA args:

- `tuner_type: lora`
- `target_modules: all-linear`
- `lora_rank: 8`
- `lora_alpha: 32`
- `lora_dropout: 0.05`
- `use_vllm: false`
- `overlong_filter: true`
- `packing: false`
- `padding_free: false`

## Extra Args Handling

Parse backend `extra_args` with `shlex`. Known options should become structured `training.swift` fields. Unknown options should remain as validation warnings or a preserved raw list; do not silently convert unknown strings into executable shell.

Example structured form:

```yaml
training:
  runner: swift_megatron
  training_type: sft
  swift:
    train_type: sft
    model: Qwen/Qwen3.6-35B-A3B
    dataset: swift/sharegpt:common-zh
    max_length: 1024
    train_samples: 200
    tuner_type: lora
    target_modules: all-linear
    lora_rank: 8
    lora_alpha: 32
    lora_dropout: 0.05
    use_vllm: false
    overlong_filter: true
    packing: false
    padding_free: false
```

## Multi-Node Cache Warning

For multi-node Megatron-SWIFT, ensure model and dataset cache paths are consistent across workers. If the platform lacks a shared cache, preflight should produce a clear diagnostic instead of launching ranks that may see different model or dataset contents.
