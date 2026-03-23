"""Convert CKIP HuggingFace BERT weights → MLX format."""

import json, sys, os
from pathlib import Path
from safetensors import safe_open
import mlx.core as mx
import mlx.nn as nn
from bert_mlx import BertForTokenClassification

# Map HF key names → our MLX key names
def map_key(k):
    k = k.replace("bert.embeddings.", "embeddings.")
    k = k.replace("bert.encoder.layer.", "layers.")
    k = k.replace(".attention.self.", ".attention.")
    k = k.replace(".attention.output.dense.", ".attention_output_dense.")
    k = k.replace(".attention.output.LayerNorm.", ".attention_output_LayerNorm.")
    k = k.replace(".intermediate.dense.", ".intermediate_dense.")
    k = k.replace(".output.dense.", ".output_dense.")
    k = k.replace(".output.LayerNorm.", ".output_LayerNorm.")
    return k

def convert(model_name, out_dir):
    from huggingface_hub import snapshot_download
    hf_dir = snapshot_download(model_name)
    print(f"Downloaded {model_name} → {hf_dir}")

    # Load config
    with open(os.path.join(hf_dir, "config.json")) as f:
        config = json.load(f)
    config["num_labels"] = config.get("num_labels", config.get("id2label", {}).__len__() or 2)

    # Load safetensors or pytorch weights
    st_path = os.path.join(hf_dir, "model.safetensors")
    if os.path.exists(st_path):
        hf_weights = {}
        with safe_open(st_path, framework="numpy") as f:
            for k in f.keys():
                hf_weights[k] = mx.array(f.get_tensor(k))
    else:
        import torch
        pt_path = os.path.join(hf_dir, "pytorch_model.bin")
        pt = torch.load(pt_path, map_location="cpu", weights_only=True)
        hf_weights = {k: mx.array(v.numpy()) for k, v in pt.items()}

    # Map keys
    mlx_weights = {}
    skipped = []
    for k, v in hf_weights.items():
        if "position_ids" in k or "pooler" in k:
            skipped.append(k)
            continue
        new_k = map_key(k)
        mlx_weights[new_k] = v

    if skipped:
        print(f"Skipped: {skipped}")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    mx.save_safetensors(os.path.join(out_dir, "weights.safetensors"), mlx_weights)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy vocab.txt
    vocab_src = os.path.join(hf_dir, "vocab.txt")
    if os.path.exists(vocab_src):
        import shutil
        shutil.copy(vocab_src, os.path.join(out_dir, "vocab.txt"))

    print(f"Saved MLX model → {out_dir}")
    print(f"  weights: {len(mlx_weights)} tensors")
    print(f"  num_labels: {config['num_labels']}")

    # Verify load
    model = BertForTokenClassification(config)
    model.load_weights(os.path.join(out_dir, "weights.safetensors"))
    print("  ✓ model loads OK")

if __name__ == "__main__":
    tasks = [
        ("ckiplab/bert-base-chinese-ws",  "models/ws"),
        ("ckiplab/bert-base-chinese-pos", "models/pos"),
        ("ckiplab/bert-base-chinese-ner", "models/ner"),
    ]
    for hf_name, out in tasks:
        print(f"\n{'='*50}")
        convert(hf_name, out)
