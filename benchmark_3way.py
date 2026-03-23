"""
3-framework benchmark: CKIP-Transformers / HF Transformers / MLX
fp32 + fp16 for all; q8 for MLX only (PyTorch 2.10 deprecated old quant API)
Text: Wikipedia 臺灣 (CC BY-SA 4.0)
"""

import json, os, re, time, statistics, platform, subprocess
import torch
import mlx.core as mx
import mlx.nn as mnn
from bert_mlx import BertForTokenClassification as MLXBert

MAX_SEQ = 510
N_RUNS = 5

with open("wiki_taiwan.txt") as f:
    raw = f.read()
text = re.sub(r'\s+', '', raw)
text = ''.join(ch for ch in text if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' or ch in '，。、；：！？「」『』（）—…──《》〈〉' or '\uff00' <= ch <= '\uffef')

# ── Shared tokenizer & decoders ──

class WordPieceTokenizer:
    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path) as f:
            for i, line in enumerate(f):
                self.vocab[line.strip()] = i
        self.unk_id = self.vocab.get("[UNK]", 100)
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)

    def encode_chunks(self, text):
        chunks = []
        for start in range(0, len(text), MAX_SEQ):
            chunk = text[start:start + MAX_SEQ]
            ids = [self.cls_id] + [self.vocab.get(ch, self.unk_id) for ch in chunk] + [self.sep_id]
            spans = [None] + list(range(len(chunk))) + [None]
            chunks.append((ids, spans, chunk))
        return chunks

def decode_ws(preds, spans, ct):
    words, cur = [], ""
    for i, s in enumerate(spans):
        if s is None: continue
        if preds[i] == 0 and cur: words.append(cur); cur = ct[s]
        else: cur += ct[s]
    if cur: words.append(cur)
    return words

def decode_pos(preds, ws_words, id2label, spans):
    result, ci = [], 0
    for word in ws_words:
        tag = "?"
        for i, s in enumerate(spans):
            if s == ci: tag = id2label.get(str(preds[i]), "?"); break
        result.append({"word": word, "pos": tag}); ci += len(word)
    return result

def decode_ner(preds, spans, ct, id2label, g_off):
    ents, cur_type, cur_start, cur_end = [], None, 0, 0
    for i, s in enumerate(spans):
        if s is None:
            if cur_type: ents.append({"text": ct[cur_start:cur_end+1], "type": cur_type, "start": cur_start+g_off}); cur_type = None
            continue
        label = id2label.get(str(preds[i]), "O")
        if label.startswith("S-"):
            if cur_type: ents.append({"text": ct[cur_start:cur_end+1], "type": cur_type, "start": cur_start+g_off})
            ents.append({"text": ct[s], "type": label[2:], "start": s+g_off}); cur_type = None
        elif label.startswith("B-"):
            if cur_type: ents.append({"text": ct[cur_start:cur_end+1], "type": cur_type, "start": cur_start+g_off})
            cur_type, cur_start, cur_end = label[2:], s, s
        elif label.startswith("I-") and cur_type: cur_end = s
        elif label.startswith("E-") and cur_type:
            cur_end = s; ents.append({"text": ct[cur_start:cur_end+1], "type": cur_type, "start": cur_start+g_off}); cur_type = None
        else:
            if cur_type: ents.append({"text": ct[cur_start:cur_end+1], "type": cur_type, "start": cur_start+g_off}); cur_type = None
    if cur_type: ents.append({"text": ct[cur_start:cur_end+1], "type": cur_type, "start": cur_start+g_off})
    return ents

def run_pipeline(infer_fn, chunks, configs):
    """Generic pipeline: WS → POS → NER using any infer_fn(task, ids) → preds"""
    ws_id2l = configs["ws"].get("id2label", {})
    pos_id2l = configs["pos"].get("id2label", {})
    ner_id2l = configs["ner"].get("id2label", {})

    all_ws = []
    for ids, spans, ct in chunks:
        preds = infer_fn("ws", ids)
        all_ws.extend(decode_ws(preds, spans, ct))

    all_pos, wi = [], 0
    for ids, spans, ct in chunks:
        preds = infer_fn("pos", ids)
        cw, rem = [], len(ct)
        while wi < len(all_ws) and rem > 0:
            w = all_ws[wi]
            if len(w) <= rem: cw.append(w); rem -= len(w); wi += 1
            else: break
        all_pos.extend(decode_pos(preds, cw, pos_id2l, spans))

    all_ner, g_off = [], 0
    for ids, spans, ct in chunks:
        preds = infer_fn("ner", ids)
        all_ner.extend(decode_ner(preds, spans, ct, ner_id2l, g_off))
        g_off += len(ct)

    return all_ws, all_pos, all_ner

def bench_speed(infer_fn, chunks, n_runs):
    """Time all 3 tasks over all chunks."""
    # warmup
    for ids, spans, ct in chunks[:3]:
        for t in ["ws","pos","ner"]: infer_fn(t, ids)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for ids, spans, ct in chunks:
            for t in ["ws","pos","ner"]: infer_fn(t, ids)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)

# ══════════════════════════════════════════════════════════════
# MLX backend
# ══════════════════════════════════════════════════════════════

def make_mlx_infer(variant):
    suffix = "" if variant == "fp32" else f"-{variant}"
    qbits = 8 if variant == "q8" else None
    models = {}
    for task in ["ws","pos","ner"]:
        d = f"models/{task}{suffix}"
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        cfg["num_labels"] = len(cfg.get("id2label", {})) or cfg.get("num_labels", 2)
        m = MLXBert(cfg)
        if qbits: mnn.quantize(m, bits=qbits)
        m.load_weights(os.path.join(d, "weights.safetensors"))
        mx.eval(m.parameters())
        models[task] = m

    def infer(task, ids):
        out = models[task](mx.array([ids]), attention_mask=mx.array([[1]*len(ids)]))
        mx.eval(out)
        return mx.argmax(out, axis=-1).tolist()[0]
    return infer

# ══════════════════════════════════════════════════════════════
# HF Transformers backend (MPS)
# ══════════════════════════════════════════════════════════════

def make_transformers_infer(variant):
    from transformers import BertForTokenClassification
    device = torch.device("mps")
    hf_names = {"ws": "ckiplab/bert-base-chinese-ws", "pos": "ckiplab/bert-base-chinese-pos", "ner": "ckiplab/bert-base-chinese-ner"}
    models = {}
    for task in ["ws","pos","ner"]:
        m = BertForTokenClassification.from_pretrained(hf_names[task])
        if variant == "fp16": m = m.half()
        m = m.to(device).eval()
        models[task] = m

    def infer(task, ids):
        t_ids = torch.tensor([ids], device=device)
        t_mask = torch.ones_like(t_ids, device=device)
        with torch.no_grad():
            out = models[task](t_ids, attention_mask=t_mask).logits
        torch.mps.synchronize()
        return out.argmax(dim=-1).cpu().tolist()[0]
    return infer

# ══════════════════════════════════════════════════════════════
# CKIP-Transformers backend (official API, MPS)
# ══════════════════════════════════════════════════════════════

def run_ckip(variant):
    """CKIP uses its own tokenizer/pipeline, so we handle it separately."""
    from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
    device = torch.device("mps")

    sents = [s + "。" for s in text.split("。") if s]

    ws = CkipWordSegmenter(model="bert-base", device=-1)
    pos = CkipPosTagger(model="bert-base", device=-1)
    ner = CkipNerChunker(model="bert-base", device=-1)

    for obj in [ws, pos, ner]:
        if variant == "fp16": obj.model = obj.model.half().to(device)
        else: obj.model.to(device)
        obj.device = device

    # warmup
    ws(sents[:5]); torch.mps.synchronize()

    # results
    ws_out = ws(sents)
    pos_out = pos(ws_out, use_delim=True)
    ner_out = ner(sents)
    torch.mps.synchronize()

    all_ws = [w for sw in ws_out for w in sw]
    all_pos = [{"word": w, "pos": p} for sw, sp in zip(ws_out, pos_out) for w, p in zip(sw, sp)]
    all_ner = []
    g_off = 0
    for si, se in enumerate(ner_out):
        for e in se:
            all_ner.append({"text": e.word, "type": e.ner, "start": e.idx[0] + g_off})
        g_off += len(sents[si])

    # speed
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        wo = ws(sents)
        pos(wo, use_delim=True)
        ner(sents)
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return all_ws, all_pos, all_ner, statistics.median(times)

# ══════════════════════════════════════════════════════════════

def diff_vs(label, bws, bpos, bner, vws, vpos, vner):
    ws_ok = (vws == bws)
    minp = min(len(bpos), len(vpos))
    pos_d = sum(1 for i in range(minp) if bpos[i]["pos"] != vpos[i]["pos"]) + abs(len(bpos)-len(vpos))
    bs = {(e["text"],e["type"],e["start"]) for e in bner}
    vs = {(e["text"],e["type"],e["start"]) for e in vner}
    nm, ne = len(bs-vs), len(vs-bs)
    ws_s = "✓" if ws_ok else f"✗ ({len(bws)} vs {len(vws)})"
    pos_s = "✓" if pos_d == 0 else f"{pos_d} 處不同"
    ner_s = "✓" if nm==0 and ne==0 else f"漏{nm}/多{ne}"
    return ws_s, pos_s, ner_s

def main():
    chip = subprocess.run(["sysctl","-n","machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem_gb = int(subprocess.run(["sysctl","-n","hw.memsize"], capture_output=True, text=True).stdout.strip()) / 1024**3
    mlx_ver = subprocess.run([".venv/bin/python3","-c","import mlx.core; print(mlx.core.__version__)"], capture_output=True, text=True).stdout.strip()

    print(f"{'='*72}")
    print(f"CKIP BERT-base 三框架完整比較")
    print(f"{'='*72}")
    print(f"文字：維基百科「臺灣」(CC BY-SA 4.0), {len(text)} 字")
    print(f"環境：{chip} / {mem_gb:.0f}GB")
    print(f"      Python {platform.python_version()} / PyTorch {torch.__version__} / MLX {mlx_ver}")

    tokenizer = WordPieceTokenizer("models/ws/vocab.txt")
    chunks = tokenizer.encode_chunks(text)
    print(f"      {len(chunks)} chunks × max {MAX_SEQ} 字, {N_RUNS} runs median\n")

    configs = {}
    for task in ["ws","pos","ner"]:
        with open(f"models/{task}/config.json") as f:
            configs[task] = json.load(f)

    results = {}  # key -> (ws, pos, ner, speed_ms)

    # ── MLX ──
    for v in ["fp32", "fp16", "q8"]:
        key = f"mlx-{v}"
        print(f"▶ {key}...", end=" ", flush=True)
        infer = make_mlx_infer(v)
        ws, pos, ner = run_pipeline(infer, chunks, configs)
        speed = bench_speed(infer, chunks, N_RUNS)
        results[key] = (ws, pos, ner, speed)
        print(f"{speed:.0f}ms / ws:{len(ws)} pos:{len(pos)} ner:{len(ner)}")
        with open(f"result_{key.replace('-','_')}.json", "w", encoding="utf-8") as f:
            json.dump({"framework":"mlx","variant":v,"ws":ws,"pos":pos,"ner":ner}, f, ensure_ascii=False, indent=2)

    # ── Transformers (HF) ──
    for v in ["fp32", "fp16"]:
        key = f"hf-{v}"
        print(f"▶ {key}...", end=" ", flush=True)
        infer = make_transformers_infer(v)
        ws, pos, ner = run_pipeline(infer, chunks, configs)
        speed = bench_speed(infer, chunks, N_RUNS)
        results[key] = (ws, pos, ner, speed)
        print(f"{speed:.0f}ms / ws:{len(ws)} pos:{len(pos)} ner:{len(ner)}")
        with open(f"result_{key.replace('-','_')}.json", "w", encoding="utf-8") as f:
            json.dump({"framework":"transformers","variant":v,"ws":ws,"pos":pos,"ner":ner}, f, ensure_ascii=False, indent=2)

    # ── CKIP ──
    for v in ["fp32", "fp16"]:
        key = f"ckip-{v}"
        print(f"▶ {key}...", end=" ", flush=True)
        ws, pos, ner, speed = run_ckip(v)
        results[key] = (ws, pos, ner, speed)
        print(f"{speed:.0f}ms / ws:{len(ws)} pos:{len(pos)} ner:{len(ner)}")
        with open(f"result_{key.replace('-','_')}.json", "w", encoding="utf-8") as f:
            json.dump({"framework":"ckip","variant":v,"ws":ws,"pos":pos,"ner":ner}, f, ensure_ascii=False, indent=2)

    # ══════════════════════════════════════════════════════════
    # Speed table
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"速度 (WS+POS+NER 全跑, {len(text)} 字)")
    print(f"{'='*72}")
    print(f"{'Framework':<22} │ {'fp32':>10} │ {'fp16':>10} │ {'q8':>10}")
    print(f"{'─'*22}─┼{'─'*12}┼{'─'*12}┼{'─'*12}")
    for fw, label in [("mlx","MLX"), ("hf","HF Transformers/MPS"), ("ckip","CKIP官方/MPS")]:
        f32 = results.get(f"{fw}-fp32", (None,None,None,0))[3]
        f16 = results.get(f"{fw}-fp16", (None,None,None,0))[3]
        q8  = results.get(f"{fw}-q8",  (None,None,None,0))[3]
        f32s = f"{f32:.0f}ms" if f32 else "—"
        f16s = f"{f16:.0f}ms" if f16 else "—"
        q8s  = f"{q8:.0f}ms" if q8 else "—"
        print(f"{label:<22} │ {f32s:>10} │ {f16s:>10} │ {q8s:>10}")

    # ══════════════════════════════════════════════════════════
    # Accuracy table (vs mlx-fp32 as baseline)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"精度 (vs mlx-fp32 baseline)")
    print(f"{'='*72}")
    bws, bpos, bner, _ = results["mlx-fp32"]
    print(f"{'Config':<22} │ {'WS':>12} │ {'POS':>14} │ {'NER':>14}")
    print(f"{'─'*22}─┼{'─'*14}┼{'─'*16}┼{'─'*16}")
    for key in ["mlx-fp16","mlx-q8","hf-fp32","hf-fp16","ckip-fp32","ckip-fp16"]:
        if key not in results: continue
        vws, vpos, vner, _ = results[key]
        ws_s, pos_s, ner_s = diff_vs(key, bws, bpos, bner, vws, vpos, vner)
        print(f"{key:<22} │ {ws_s:>12} │ {pos_s:>14} │ {ner_s:>14}")

    print(f"\n所有 JSON 已儲存 ✓")

if __name__ == "__main__":
    main()
