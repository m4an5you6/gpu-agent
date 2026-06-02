#!/usr/bin/env python3
"""
Convert Megatron-LM core_v0.11 torch_dist checkpoint to HuggingFace GPT-2 format.

Usage:
    python convert_megatron_to_hf.py

Requirements:
    - torch with CUDA support
    - Megatron-LM in sys.path (set MEGATRON_LM_DIR below)
    - Checkpoint in torch_dist format (default Megatron save format)

Key details:
    - torch_dist checkpoints require torch.distributed.init_process_group()
    - HF GPT-2 uses Conv1D: weights are [in, out] not [out, in] -> must transpose
    - Megatron stores per-layer tensors stacked: shape [n_layer, ...] -> split dim 0
    - Layer norms stored as qkv.layer_norm_weight -> ln_1, mlp.linear_fc1.layer_norm_weight -> ln_2
"""

import sys
import os
import json
import torch

# === CONFIGURE THESE ===
MEGATRON_LM_DIR = "/root/gpufree-data/gpucloud-runs/gpt2-wikitext2-local/Megatron-LM"
CHECKPOINT_DIR = "/root/gpufree-data/gpucloud-runs/gpt2-wikitext2-local/checkpoints/iter_0000020"
OUTPUT_DIR = "/root/gpufree-data/gpucloud-runs/gpt2-wikitext2-local/model_hf"
VOCAB_FILE = "/root/gpufree-data/gpucloud-runs/gpt2-wikitext2-local/data/tokenizer/vocab.json"
MERGE_FILE = "/root/gpufree-data/gpucloud-runs/gpt2-wikitext2-local/data/tokenizer/merges.txt"

# Model config (must match training YAML)
MODEL_CONFIG = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "vocab_size": 50257,
    "n_positions": 128,
    "n_embd": 128,
    "n_layer": 2,
    "n_head": 2,
    "n_inner": 512,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "layer_norm_epsilon": 1e-5,
    "initializer_range": 0.02,
    "bos_token_id": 50256,
    "eos_token_id": 50256,
}
# === END CONFIG ===

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.path.insert(0, MEGATRON_LM_DIR)

    # Initialize distributed (required for torch_dist checkpoint loading)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29697'  # Different from training port
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    torch.distributed.init_process_group(backend='gloo')

    from megatron.core.dist_checkpointing import load_plain_tensors
    print(f"Loading checkpoint from {CHECKPOINT_DIR}...")
    state_dict = load_plain_tensors(CHECKPOINT_DIR)

    n_layer = MODEL_CONFIG["n_layer"]
    hf_state_dict = {}

    for k, v in state_dict.items():
        if not hasattr(v, 'shape'):
            continue
        v = v.float()  # Convert fp16 to fp32 for HF

        # === Embedding ===
        if k == 'embedding.word_embeddings.weight':
            hf_state_dict['wte.weight'] = v
        elif k == 'embedding.position_embeddings.weight':
            hf_state_dict['wpe.weight'] = v
        # === Final Layer Norm ===
        elif k == 'decoder.final_layernorm.weight':
            hf_state_dict['ln_f.weight'] = v
        elif k == 'decoder.final_layernorm.bias':
            hf_state_dict['ln_f.bias'] = v
        # === Per-Layer Tensors (stacked: shape [n_layer, ...]) ===
        elif k.startswith('decoder.layers.'):
            suffix = k.replace('decoder.layers.', '')
            for i in range(n_layer):
                layer_v = v[i] if v.dim() > 1 and v.shape[0] == n_layer else v

                # Attention QKV (CONV1D: must transpose!)
                if suffix == 'self_attention.linear_qkv.weight':
                    hf_state_dict[f'h.{i}.attn.c_attn.weight'] = layer_v.T
                elif suffix == 'self_attention.linear_qkv.bias':
                    hf_state_dict[f'h.{i}.attn.c_attn.bias'] = layer_v
                # Attention projection (CONV1D: must transpose!)
                elif suffix == 'self_attention.linear_proj.weight':
                    hf_state_dict[f'h.{i}.attn.c_proj.weight'] = layer_v.T
                elif suffix == 'self_attention.linear_proj.bias':
                    hf_state_dict[f'h.{i}.attn.c_proj.bias'] = layer_v
                # Layer norm 1 (stored as qkv layer_norm)
                elif suffix == 'self_attention.linear_qkv.layer_norm_weight':
                    hf_state_dict[f'h.{i}.ln_1.weight'] = layer_v
                elif suffix == 'self_attention.linear_qkv.layer_norm_bias':
                    hf_state_dict[f'h.{i}.ln_1.bias'] = layer_v
                # Layer norm 2 (stored as fc1 layer_norm)
                elif suffix == 'mlp.linear_fc1.layer_norm_weight':
                    hf_state_dict[f'h.{i}.ln_2.weight'] = layer_v
                elif suffix == 'mlp.linear_fc1.layer_norm_bias':
                    hf_state_dict[f'h.{i}.ln_2.bias'] = layer_v
                # MLP fc1 (CONV1D: must transpose!)
                elif suffix == 'mlp.linear_fc1.weight':
                    hf_state_dict[f'h.{i}.mlp.c_fc.weight'] = layer_v.T
                elif suffix == 'mlp.linear_fc1.bias':
                    hf_state_dict[f'h.{i}.mlp.c_fc.bias'] = layer_v
                # MLP fc2 (CONV1D: must transpose!)
                elif suffix == 'mlp.linear_fc2.weight':
                    hf_state_dict[f'h.{i}.mlp.c_proj.weight'] = layer_v.T
                elif suffix == 'mlp.linear_fc2.bias':
                    hf_state_dict[f'h.{i}.mlp.c_proj.bias'] = layer_v
                else:
                    print(f"  WARNING: unmapped layer key: {k}")

    print(f"\nHF state dict ({len(hf_state_dict)} tensors):")
    for k in sorted(hf_state_dict.keys()):
        print(f"  {k}: {hf_state_dict[k].shape}")

    # Save config
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(MODEL_CONFIG, f, indent=2)

    # Save model weights
    torch.save(hf_state_dict, os.path.join(OUTPUT_DIR, 'pytorch_model.bin'))

    # Copy tokenizer files
    import shutil
    shutil.copy2(VOCAB_FILE, os.path.join(OUTPUT_DIR, 'vocab.json'))
    shutil.copy2(MERGE_FILE, os.path.join(OUTPUT_DIR, 'merges.txt'))

    # Save tokenizer config
    tok_config = {"model_max_length": MODEL_CONFIG["n_positions"], "tokenizer_class": "GPT2Tokenizer"}
    with open(os.path.join(OUTPUT_DIR, 'tokenizer_config.json'), 'w') as f:
        json.dump(tok_config, f, indent=2)

    # Save generation config
    gen_config = {"bos_token_id": 50256, "eos_token_id": 50256, "do_sample": True, "max_length": MODEL_CONFIG["n_positions"]}
    with open(os.path.join(OUTPUT_DIR, 'generation_config.json'), 'w') as f:
        json.dump(gen_config, f, indent=2)

    print(f"\nDone! Model saved to {OUTPUT_DIR}")
    print(f"Files: {os.listdir(OUTPUT_DIR)}")
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
