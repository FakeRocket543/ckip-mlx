"""
Full benchmark: fp32 / fp16 / q8 — accuracy, speed, RAM
Using Wikipedia "臺灣" article (CC BY-SA 4.0, ~39k chars)
"""

import json, os, time, statistics, tracemalloc, re
import mlx.core as mx
import mlx.nn as nn
from bert_mlx import BertForTokenClassification

MAX_SEQ = 510  # 512 - [CLS] - [SEP]

class WordPieceTokenizer:
    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path) as f:
            for i, line in enumerate(f):
                self.vocab[line.strip()] = i
        self.unk_id = self.vocab.get("[UNK]", 100)
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)
        self.pad_id = self.vocab.get("[PAD]", 0)

    def encode_chunks(self, text):
        """Split text into 510-char chunks, return list of (input_ids, mask, spans, chunk_text)."""
        chunks = []
        for start in range(0, len(text), MAX_SEQ):
            chunk = text[start:start + MAX_SEQ]
            ids = [self.cls_id] + [self.vocab.get(ch, self.unk_id) for ch in chunk] + [self.sep_id]
            spans = [None] + list(range(len(chunk))) + [None]
            chunks.append((mx.array([ids]), mx.array([[1]*len(ids)]), spans, chunk))
        return chunks

def load_model(task_dir, quantize_bits=None):
    with open(os.path.join(task_dir, "config.json")) as f:
        config = json.load(f)
    config["num_labels"] = len(config.get("id2label", {})) or config.get("num_labels", 2)
    model = BertForTokenClassification(config)
    if quantize_bits:
        nn.quantize(model, bits=quantize_bits)
    model.load_weights(os.path.join(task_dir, "weights.safetensors"))
    mx.eval(model.parameters())
    return model, config

def predict_all_chunks(model, chunks):
    """Run inference on all chunks, return flat list of per-token preds."""
    all_preds = []
    for input_ids, mask, spans, chunk_text in chunks:
        out = model(input_ids, attention_mask=mask)
        mx.eval(out)
        preds = mx.argmax(out, axis=-1).tolist()[0]
        all_preds.extend(preds)
    return all_preds

def bench_speed(model, chunks, n_runs=10):
    """Benchmark: run n times, return median ms."""
    # warmup
    predict_all_chunks(model, chunks)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_all_chunks(model, chunks)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)

def get_model_ram_mb(task_dir, quantize_bits=None):
    """Measure model RAM by loading and checking weight sizes."""
    with open(os.path.join(task_dir, "config.json")) as f:
        config = json.load(f)
    config["num_labels"] = len(config.get("id2label", {})) or config.get("num_labels", 2)
    model = BertForTokenClassification(config)
    if quantize_bits:
        nn.quantize(model, bits=quantize_bits)
    model.load_weights(os.path.join(task_dir, "weights.safetensors"))
    mx.eval(model.parameters())
    # Sum all parameter bytes
    total_bytes = 0
    from mlx.utils import tree_flatten
    for k, v in tree_flatten(model.parameters()):
        total_bytes += v.nbytes
    return total_bytes / 1024 / 1024

def get_file_size_mb(task_dir):
    w = os.path.join(task_dir, "weights.safetensors")
    return os.path.getsize(w) / 1024 / 1024 if os.path.exists(w) else 0

def main():
    # Load text
    with open("wiki_taiwan.txt") as f:
        raw = f.read()

    # Clean: remove section headers, extra whitespace, keep only CJK + punctuation
    text = re.sub(r'\n==+[^=]+=+\n', '', raw)
    text = re.sub(r'\s+', '', text)
    # Remove ASCII-heavy segments (references, URLs, etc.)
    # Keep chars that CKIP BERT can handle (CJK + Chinese punctuation + digits)
    cleaned = []
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' or ch in '，。、；：！？「」『』（）—…──《》〈〉' or '\uff00' <= ch <= '\uffef':
            cleaned.append(ch)
    text = ''.join(cleaned)

    print(f"來源：維基百科「臺灣」條目 (CC BY-SA 4.0)")
    print(f"清理後文字長度：{len(text)} 字")

    tokenizer = WordPieceTokenizer("models/ws/vocab.txt")
    chunks = tokenizer.encode_chunks(text)
    print(f"分段數：{len(chunks)} chunks (每段最多 {MAX_SEQ} 字)")

    # System info
    import platform, subprocess
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem_bytes = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
    mem_gb = mem_bytes / 1024**3

    print(f"\n環境：{platform.system()} {platform.machine()}")
    print(f"晶片：{chip}")
    print(f"記憶體：{mem_gb:.0f} GB")
    print(f"Python：{platform.python_version()}")
    mlx_ver = subprocess.run([".venv/bin/python3", "-c", "import mlx; print(mlx.__version__)"], capture_output=True, text=True).stdout.strip()
    print(f"MLX：{mlx_ver}")

    N_RUNS = 10
    variants = [
        ("fp32", "models/{task}", None),
        ("fp16", "models/{task}-fp16", None),
        ("q8",   "models/{task}", 8),  # quantize on the fly from fp32 source
    ]

    # Use pre-converted q8 weights if available
    for task in ["ws", "pos", "ner"]:
        print(f"\n{'='*70}")
        print(f"[{task.upper()}]")
        print(f"{'─'*70}")
        print(f"{'Variant':>8} │ {'Disk':>10} │ {'RAM':>10} │ {'Speed':>12} │ {'Accuracy':>10}")
        print(f"{'─'*70}")

        # Get fp32 baseline preds
        fp32_dir = f"models/{task}"
        fp32_model, config = load_model(fp32_dir)
        fp32_preds = predict_all_chunks(fp32_model, chunks)
        fp32_speed = bench_speed(fp32_model, chunks, N_RUNS)
        fp32_ram = get_model_ram_mb(fp32_dir)
        fp32_disk = get_file_size_mb(fp32_dir)
        print(f"{'fp32':>8} │ {fp32_disk:>8.1f}MB │ {fp32_ram:>8.1f}MB │ {fp32_speed:>8.1f} ms  │ {'baseline':>10}")
        del fp32_model

        # fp16
        fp16_dir = f"models/{task}-fp16"
        fp16_model, _ = load_model(fp16_dir)
        fp16_preds = predict_all_chunks(fp16_model, chunks)
        fp16_speed = bench_speed(fp16_model, chunks, N_RUNS)
        fp16_ram = get_model_ram_mb(fp16_dir)
        fp16_disk = get_file_size_mb(fp16_dir)
        fp16_match = sum(1 for a, b in zip(fp32_preds, fp16_preds) if a == b)
        fp16_acc = fp16_match / len(fp32_preds) * 100
        print(f"{'fp16':>8} │ {fp16_disk:>8.1f}MB │ {fp16_ram:>8.1f}MB │ {fp16_speed:>8.1f} ms  │ {fp16_acc:>9.2f}%")
        del fp16_model

        # q8
        q8_dir = f"models/{task}-q8"
        q8_model, _ = load_model(q8_dir, quantize_bits=8)
        q8_preds = predict_all_chunks(q8_model, chunks)
        q8_speed = bench_speed(q8_model, chunks, N_RUNS)
        q8_ram = get_model_ram_mb(q8_dir, quantize_bits=8)
        q8_disk = get_file_size_mb(q8_dir)
        q8_match = sum(1 for a, b in zip(fp32_preds, q8_preds) if a == b)
        q8_acc = q8_match / len(fp32_preds) * 100
        print(f"{'q8':>8} │ {q8_disk:>8.1f}MB │ {q8_ram:>8.1f}MB │ {q8_speed:>8.1f} ms  │ {q8_acc:>9.2f}%")
        del q8_model

        print(f"{'─'*70}")
        print(f"  tokens: {len(fp32_preds)}, {N_RUNS} runs median, speed = total inference time for all {len(chunks)} chunks")

if __name__ == "__main__":
    main()
