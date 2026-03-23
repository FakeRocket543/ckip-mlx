"""Run WS/POS/NER with fp32/fp16/q8, save JSON + diff summary."""

import json, os, re
import mlx.core as mx
import mlx.nn as nn
from bert_mlx import BertForTokenClassification

MAX_SEQ = 510

with open("wiki_taiwan.txt") as f:
    raw = f.read()
text = re.sub(r'\s+', '', raw)
text = ''.join(ch for ch in text if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' or ch in '，。、；：！？「」『』（）—…──《》〈〉' or '\uff00' <= ch <= '\uffef')

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

def run_ws(model, chunks):
    words_all = []
    for input_ids, mask, spans, chunk_text in chunks:
        out = model(input_ids, attention_mask=mask); mx.eval(out)
        preds = mx.argmax(out, axis=-1).tolist()[0]
        words, cur = [], ""
        for i, s in enumerate(spans):
            if s is None: continue
            if preds[i] == 0 and cur:
                words.append(cur); cur = chunk_text[s]
            else:
                cur += chunk_text[s]
        if cur: words.append(cur)
        words_all.extend(words)
    return words_all

def run_pos(model, config, chunks, ws_words):
    id2label = config.get("id2label", {})
    result, wi = [], 0
    for input_ids, mask, spans, chunk_text in chunks:
        out = model(input_ids, attention_mask=mask); mx.eval(out)
        preds = mx.argmax(out, axis=-1).tolist()[0]
        chunk_words, remaining = [], len(chunk_text)
        while wi < len(ws_words) and remaining > 0:
            w = ws_words[wi]
            if len(w) <= remaining:
                chunk_words.append(w); remaining -= len(w); wi += 1
            else: break
        ci = 0
        for word in chunk_words:
            tag = "?"
            for i, s in enumerate(spans):
                if s == ci: tag = id2label.get(str(preds[i]), "?"); break
            result.append({"word": word, "pos": tag})
            ci += len(word)
    return result

def run_ner(model, config, chunks):
    id2label = config.get("id2label", {})
    all_ner, g_off = [], 0
    for input_ids, mask, spans, chunk_text in chunks:
        out = model(input_ids, attention_mask=mask); mx.eval(out)
        preds = mx.argmax(out, axis=-1).tolist()[0]
        ents, cur_type, cur_start, cur_end = [], None, 0, 0
        for i, s in enumerate(spans):
            if s is None:
                if cur_type: ents.append({"text": chunk_text[cur_start:cur_end+1], "type": cur_type, "start": cur_start + g_off}); cur_type = None
                continue
            label = id2label.get(str(preds[i]), "O")
            if label.startswith("S-"):
                if cur_type: ents.append({"text": chunk_text[cur_start:cur_end+1], "type": cur_type, "start": cur_start + g_off})
                ents.append({"text": chunk_text[s], "type": label[2:], "start": s + g_off}); cur_type = None
            elif label.startswith("B-"):
                if cur_type: ents.append({"text": chunk_text[cur_start:cur_end+1], "type": cur_type, "start": cur_start + g_off})
                cur_type, cur_start, cur_end = label[2:], s, s
            elif label.startswith("I-") and cur_type: cur_end = s
            elif label.startswith("E-") and cur_type:
                cur_end = s; ents.append({"text": chunk_text[cur_start:cur_end+1], "type": cur_type, "start": cur_start + g_off}); cur_type = None
            else:
                if cur_type: ents.append({"text": chunk_text[cur_start:cur_end+1], "type": cur_type, "start": cur_start + g_off}); cur_type = None
        if cur_type: ents.append({"text": chunk_text[cur_start:cur_end+1], "type": cur_type, "start": cur_start + g_off})
        all_ner.extend(ents); g_off += len(chunk_text)
    return all_ner

tokenizer = WordPieceTokenizer("models/ws/vocab.txt")
chunks = tokenizer.encode_chunks(text)
print(f"文字: {len(text)} 字, {len(chunks)} chunks\n")

variants = [
    ("fp32", "models/{task}", None),
    ("fp16", "models/{task}-fp16", None),
    ("q8",   "models/{task}-q8", 8),
]

results = {}
for vname, dir_tpl, qbits in variants:
    print(f"── {vname} ──")
    ws_model, _ = load_model(dir_tpl.format(task="ws"), qbits)
    ws_words = run_ws(ws_model, chunks)
    del ws_model

    pos_model, pos_cfg = load_model(dir_tpl.format(task="pos"), qbits)
    pos_result = run_pos(pos_model, pos_cfg, chunks, ws_words)
    del pos_model

    ner_model, ner_cfg = load_model(dir_tpl.format(task="ner"), qbits)
    ner_result = run_ner(ner_model, ner_cfg, chunks)
    del ner_model

    results[vname] = {"ws": ws_words, "pos": pos_result, "ner": ner_result}
    print(f"  WS: {len(ws_words)} 詞 / POS: {len(pos_result)} / NER: {len(ner_result)} 實體")

    out = {"source": "維基百科「臺灣」(CC BY-SA 4.0)", "variant": vname,
           "text_length": len(text), "ws": ws_words, "pos": pos_result, "ner": ner_result}
    path = f"wiki_taiwan_{vname}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  → {path}")

# ── Diff ──
print(f"\n{'='*60}")
print("差異比較 (vs fp32)")
print(f"{'='*60}")

fp32 = results["fp32"]
for vname in ["fp16", "q8"]:
    v = results[vname]
    print(f"\n── {vname} vs fp32 ──")

    # WS diff
    ws_diffs = []
    i32, iv = 0, 0
    # Compare by joining back to text and re-splitting
    fp32_str = "｜".join(fp32["ws"])
    v_str = "｜".join(v["ws"])
    if fp32_str == v_str:
        print(f"  WS: 完全一致 ✓")
    else:
        # Find differing words by position
        pos32, posv = 0, 0
        diffs = []
        for w32 in fp32["ws"]:
            end32 = pos32 + len(w32)
            # find corresponding words in variant
            vwords = []
            while posv < end32 and iv < len(v["ws"]):
                vw = v["ws"][iv]
                vwords.append(vw)
                posv += len(vw)
                iv += 1
            if vwords != [w32]:
                diffs.append((pos32, w32, vwords))
            pos32 = end32
        print(f"  WS: {len(diffs)} 處不同")
        for pos, orig, new in diffs[:10]:
            print(f"    @{pos}: 「{orig}」→「{'｜'.join(new)}」")
        if len(diffs) > 10: print(f"    ...+{len(diffs)-10}")

    # POS diff
    pos_diffs = [(i, a["word"], a["pos"], b["pos"]) for i, (a, b) in enumerate(zip(fp32["pos"], v["pos"])) if a["pos"] != b["pos"]]
    if not pos_diffs:
        print(f"  POS: 完全一致 ✓")
    else:
        print(f"  POS: {len(pos_diffs)} 處不同")
        for i, w, a, b in pos_diffs[:10]:
            print(f"    「{w}」: {a} → {b}")
        if len(pos_diffs) > 10: print(f"    ...+{len(pos_diffs)-10}")

    # NER diff
    fp32_ner_set = {(e["text"], e["type"], e["start"]) for e in fp32["ner"]}
    v_ner_set = {(e["text"], e["type"], e["start"]) for e in v["ner"]}
    missing = fp32_ner_set - v_ner_set
    extra = v_ner_set - fp32_ner_set
    if not missing and not extra:
        print(f"  NER: 完全一致 ✓")
    else:
        print(f"  NER: 漏掉 {len(missing)} / 多出 {len(extra)}")
        for t, ty, s in sorted(missing)[:5]:
            print(f"    漏: [{ty}] {t} @{s}")
        for t, ty, s in sorted(extra)[:5]:
            print(f"    多: [{ty}] {t} @{s}")
        if len(missing) > 5 or len(extra) > 5: print(f"    ...")
