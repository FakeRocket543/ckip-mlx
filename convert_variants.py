"""Convert fp32 models → fp16 and 8-bit quantized variants."""

import json, os, shutil
import mlx.core as mx
import mlx.nn as nn
from bert_mlx import BertForTokenClassification

TASKS = ["ws", "pos", "ner"]

def load_model(task_dir):
    with open(os.path.join(task_dir, "config.json")) as f:
        config = json.load(f)
    config["num_labels"] = len(config.get("id2label", {})) or config.get("num_labels", 2)
    model = BertForTokenClassification(config)
    model.load_weights(os.path.join(task_dir, "weights.safetensors"))
    mx.eval(model.parameters())
    return model, config

def save_variant(model, config, src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_weights(os.path.join(out_dir, "weights.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    vocab_src = os.path.join(src_dir, "vocab.txt")
    if os.path.exists(vocab_src):
        shutil.copy(vocab_src, os.path.join(out_dir, "vocab.txt"))

def get_size_mb(path):
    total = 0
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)
    return total / 1024 / 1024

def main():
    for task in TASKS:
        src = f"models/{task}"
        model, config = load_model(src)
        print(f"\n{'='*50}")
        print(f"[{task.upper()}] fp32: {get_size_mb(src):.1f} MB")

        # fp16
        fp16_dir = f"models/{task}-fp16"
        fp16_model = BertForTokenClassification(config)
        fp16_model.load_weights(os.path.join(src, "weights.safetensors"))
        fp16_model.apply(lambda v: v.astype(mx.float16) if isinstance(v, mx.array) else v)
        mx.eval(fp16_model.parameters())
        save_variant(fp16_model, config, src, fp16_dir)
        print(f"  fp16: {get_size_mb(fp16_dir):.1f} MB")

        # 8-bit quantized
        q8_dir = f"models/{task}-q8"
        q8_model = BertForTokenClassification(config)
        q8_model.load_weights(os.path.join(src, "weights.safetensors"))
        mx.eval(q8_model.parameters())
        nn.quantize(q8_model, bits=8)
        mx.eval(q8_model.parameters())
        save_variant(q8_model, config, src, q8_dir)
        print(f"  q8:   {get_size_mb(q8_dir):.1f} MB")

    print(f"\n{'='*50}")
    print("Done! 各版本目錄：")
    for task in TASKS:
        for suffix in ["", "-fp16", "-q8"]:
            d = f"models/{task}{suffix}"
            if os.path.exists(d):
                print(f"  {d:20s} {get_size_mb(d):>8.1f} MB")

if __name__ == "__main__":
    main()
