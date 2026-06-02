#!/usr/bin/env python3
"""
Convert Megatron-LM torch_dist checkpoint (core_v0.11) to HuggingFace GPT-2 format.

This script handles the core_v0.11 checkpoint format where:
- Per-layer tensors have shape [n_layers, ...] (stacked along dim 0)
- torch.distributed must be initialized for load_plain_tensors()
- No saver_transformers exists, so we do manual key mapping

Usage:
    python megatron_to_hf_conversion.py \
        --checkpoint-dir checkpoints/iter_0000020 \
        --output-dir model_hf \
        --vocab-file data/tokenizer/vocab.json \
        --merge-file data/tokenizer/merges.txt \
        --megatron-lm-dir /path/to/Megatron-LM \
        --n-layers 2 --hidden-size 128 --n-heads 2 --n-inner 512 \
        --seq-length 128 --vocab-size 50257
"""
import argparse
import json
import os
import shutil
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True, help="Path to iter_NNNNNNN directory")
    p.add_argument("--output-dir", required=True, help="Output HF model directory")
    p.add_argument("--vocab-file", required=True)
    p.add_argument("--merge-file", required=True)
    p.add_argument("--megatron-lm-dir", default=None, help="Add to sys.path")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--n-inner", type=int, default=512)
    p.add_argument("--seq-length", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--master-port", type=int, default=29698)
    return p.parse_args()


def init_distributed(addr, port):
    os.environ["MASTER_ADDR"] = str(addr)
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="gloo")


def load_checkpoint(checkpoint_dir):
    from megatron.core.dist_checkpointing import load_plain_tensors
    print(f"Loading checkpoint from {checkpoint_dir}...")
    state_dict = load_plain_tensors(checkpoint_dir)
    print(f"Loaded {len(state_dict)} tensors")
    return state_dict


def build_hf_state_dict(state_dict, n_layers):
    """Map Megatron tensor names to HuggingFace GPT-2 names.

    In core_v0.11, per-layer tensors are stored with shape [n_layers, ...].
    We split along dim 0 to get individual layer weights.
    """
    hf = {}
    for k, v in state_dict.items():
        if not hasattr(v, "shape"):
            continue
        v = v.float()

        if k == "embedding.word_embeddings.weight":
            hf["wte.weight"] = v
        elif k == "embedding.position_embeddings.weight":
            hf["wpe.weight"] = v
        elif k == "decoder.final_layernorm.weight":
            hf["ln_f.weight"] = v
        elif k == "decoder.final_layernorm.bias":
            hf["ln_f.bias"] = v
        elif k.startswith("decoder.layers."):
            suffix = k.replace("decoder.layers.", "")
            for i in range(n_layers):
                lv = v[i] if v.dim() > 1 and v.shape[0] == n_layers else v

                mapping = {
                    "self_attention.linear_qkv.weight": f"h.{i}.attn.c_attn.weight",
                    "self_attention.linear_qkv.bias": f"h.{i}.attn.c_attn.bias",
                    "self_attention.linear_proj.weight": f"h.{i}.attn.c_proj.weight",
                    "self_attention.linear_proj.bias": f"h.{i}.attn.c_proj.bias",
                    "self_attention.linear_qkv.layer_norm_weight": f"h.{i}.ln_1.weight",
                    "self_attention.linear_qkv.layer_norm_bias": f"h.{i}.ln_1.bias",
                    "mlp.linear_fc1.layer_norm_weight": f"h.{i}.ln_2.weight",
                    "mlp.linear_fc1.layer_norm_bias": f"h.{i}.ln_2.bias",
                    "mlp.linear_fc1.weight": f"h.{i}.mlp.c_fc.weight",
                    "mlp.linear_fc1.bias": f"h.{i}.mlp.c_fc.bias",
                    "mlp.linear_fc2.weight": f"h.{i}.mlp.c_proj.weight",
                    "mlp.linear_fc2.bias": f"h.{i}.mlp.c_proj.bias",
                }
                if suffix in mapping:
                    hf[mapping[suffix]] = lv
                else:
                    print(f"  WARNING: unmapped layer key: {k}")
        else:
            print(f"  WARNING: unmapped key: {k}")
    return hf


def save_hf_model(hf_state_dict, config, output_dir, vocab_file, merge_file):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    torch.save(hf_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    shutil.copy2(vocab_file, os.path.join(output_dir, "vocab.json"))
    shutil.copy2(merge_file, os.path.join(output_dir, "merges.txt"))

    gen_config = {
        "bos_token_id": config.get("bos_token_id", 50256),
        "eos_token_id": config.get("eos_token_id", 50256),
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    tok_config = {
        "model_max_length": config.get("n_positions", 128),
        "tokenizer_class": "GPT2Tokenizer",
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2)

    print(f"Saved {len(hf_state_dict)} tensors to {output_dir}")
    print("Files:", os.listdir(output_dir))


def main():
    args = parse_args()

    if args.megatron_lm_dir:
        sys.path.insert(0, args.megatron_lm_dir)

    init_distributed(args.master_addr, args.master_port)
    state_dict = load_checkpoint(args.checkpoint_dir)
    hf_state_dict = build_hf_state_dict(state_dict, args.n_layers)

    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": args.vocab_size,
        "n_positions": args.seq_length,
        "n_embd": args.hidden_size,
        "n_layer": args.n_layers,
        "n_head": args.n_heads,
        "n_inner": args.n_inner,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
    }

    save_hf_model(hf_state_dict, config, args.output_dir, args.vocab_file, args.merge_file)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
