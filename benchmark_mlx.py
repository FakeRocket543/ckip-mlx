"""Full comparison: MLX vs PyTorch MPS — output correctness + speed."""

import time, json, os, statistics
import mlx.core as mx
from bert_mlx import BertForTokenClassification

SENTENCES = [
    "中央研究院資訊科學研究所開發了自然語言處理工具。",
    "台積電今天股價上漲三十元，市值突破二十兆。",
    "總統府前舉行國慶大典，數萬民眾到場觀禮。",
    "他在台北市信義區的誠品書店買了三本小說。",
    "國立台灣大學醫學院附設醫院的急診室人滿為患。",
    "高雄港是台灣最大的國際商港，每年處理數百萬個貨櫃。",
    "花蓮太魯閣國家公園的峽谷景觀聞名世界。",
    "中華電信宣布五G網路覆蓋率已達百分之九十。",
    "台灣半導體產業在全球供應鏈中扮演關鍵角色。",
    "故宮博物院收藏了超過六十萬件珍貴文物。",
]

# ── Tokenizer ────────────────────────────────────────

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

    def encode_batch(self, texts):
        all_ids, all_spans = [], []
        max_len = 0
        for text in texts:
            tids = [self.vocab.get(ch, self.unk_id) for ch in text]
            spans = list(range(len(text)))
            ids = [self.cls_id] + tids + [self.sep_id]
            spans = [None] + spans + [None]
            all_ids.append(ids)
            all_spans.append(spans)
            max_len = max(max_len, len(ids))
        input_ids, attention_mask = [], []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return mx.array(input_ids), mx.array(attention_mask), all_spans

# ── Decode ───────────────────────────────────────────

def decode_ws(preds, spans, text):
    words, cur = [], ""
    for i, span_idx in enumerate(spans):
        if span_idx is None:
            continue
        ch = text[span_idx]
        if preds[i] == 0 and cur:  # B = new word
            words.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        words.append(cur)
    return words

def decode_pos(preds, ws_words, id2label, spans):
    tags = []
    # Map: for each word, take the prediction at its first character
    char_idx = 0
    for word in ws_words:
        # Find the span index for this character position
        token_pos = None
        for i, s in enumerate(spans):
            if s == char_idx:
                token_pos = i
                break
        if token_pos is not None:
            tags.append(id2label.get(str(preds[token_pos]), "?"))
        else:
            tags.append("?")
        char_idx += len(word)
    return tags

def decode_ner(preds, spans, text, id2label):
    """BIOES NER decode."""
    entities = []
    cur_type, cur_start, cur_end = None, 0, 0
    for i, span_idx in enumerate(spans):
        if span_idx is None:
            if cur_type:
                entities.append((text[cur_start:cur_end+1], cur_type, cur_start))
                cur_type = None
            continue
        label = id2label.get(str(preds[i]), "O")
        if label.startswith("S-"):
            if cur_type:
                entities.append((text[cur_start:cur_end+1], cur_type, cur_start))
            entities.append((text[span_idx], label[2:], span_idx))
            cur_type = None
        elif label.startswith("B-"):
            if cur_type:
                entities.append((text[cur_start:cur_end+1], cur_type, cur_start))
            cur_type = label[2:]
            cur_start = span_idx
            cur_end = span_idx
        elif label.startswith("I-") and cur_type:
            cur_end = span_idx
        elif label.startswith("E-") and cur_type:
            cur_end = span_idx
            entities.append((text[cur_start:cur_end+1], cur_type, cur_start))
            cur_type = None
        else:
            if cur_type:
                entities.append((text[cur_start:cur_end+1], cur_type, cur_start))
                cur_type = None
    if cur_type:
        entities.append((text[cur_start:cur_end+1], cur_type, cur_start))
    return entities

# ── Load model ───────────────────────────────────────

def load_model(task_dir):
    with open(os.path.join(task_dir, "config.json")) as f:
        config = json.load(f)
    config["num_labels"] = len(config.get("id2label", {})) or config.get("num_labels", 2)
    model = BertForTokenClassification(config)
    model.load_weights(os.path.join(task_dir, "weights.safetensors"))
    mx.eval(model.parameters())
    return model, config

# ── PyTorch MPS baseline ─────────────────────────────

def pytorch_baseline():
    import torch
    from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
    mps = torch.device("mps")
    results = {}
    for name, Cls in [("ws", CkipWordSegmenter), ("pos", CkipPosTagger), ("ner", CkipNerChunker)]:
        m = Cls(model="bert-base", device=-1)
        m.model.to(mps)
        m.device = mps
        # warmup
        if name == "pos":
            m(results["ws_out"], use_delim=True)
        elif name == "ws":
            m(SENTENCES)
        else:
            m(SENTENCES)
        times = []
        out = None
        for _ in range(20):
            t0 = time.perf_counter()
            if name == "pos":
                out = m(results["ws_out"], use_delim=True)
            else:
                out = m(SENTENCES)
            torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        results[f"{name}_times"] = times
        results[f"{name}_out"] = out
    return results

# ── MLX benchmark ────────────────────────────────────

def mlx_benchmark(tokenizer):
    input_ids, attention_mask, spans = tokenizer.encode_batch(SENTENCES)
    results = {}
    for task in ["ws", "pos", "ner"]:
        model, config = load_model(f"models/{task}")
        id2label = config.get("id2label", {})
        # warmup
        out = model(input_ids, attention_mask=attention_mask)
        mx.eval(out)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            out = model(input_ids, attention_mask=attention_mask)
            mx.eval(out)
            times.append((time.perf_counter() - t0) * 1000)
        preds = mx.argmax(out, axis=-1).tolist()
        results[f"{task}_times"] = times
        results[f"{task}_preds"] = preds
        results[f"{task}_id2label"] = id2label
    results["spans"] = spans
    return results

def main():
    tokenizer = WordPieceTokenizer("models/ws/vocab.txt")

    print("=" * 65)
    print("  CKIP BERT-base: MLX vs PyTorch MPS (20 runs, 10 sentences)")
    print("=" * 65)

    # PyTorch MPS
    print("\n▶ Running PyTorch MPS...")
    pt = pytorch_baseline()

    # MLX
    print("▶ Running MLX...")
    ml = mlx_benchmark(tokenizer)

    # ── Speed comparison ─────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'Task':>4} │ {'PyTorch MPS':>18} │ {'MLX':>18} │ {'Speedup':>8}")
    print(f"{'─'*65}")
    for task in ["ws", "pos", "ner"]:
        pt_med = statistics.median(pt[f"{task}_times"])
        ml_med = statistics.median(ml[f"{task}_times"])
        speedup = pt_med / ml_med
        print(f"{task.upper():>4} │ {pt_med:>12.2f} ms    │ {ml_med:>12.2f} ms    │ {speedup:>6.2f}x")
    print(f"{'─'*65}")

    # ── Output comparison ────────────────────────────
    print(f"\n{'='*65}")
    print("Output comparison (all 10 sentences):")
    print(f"{'='*65}")

    spans = ml["spans"]
    ws_preds = ml["ws_preds"]

    for i, sent in enumerate(SENTENCES):
        print(f"\n[{i}] {sent}")

        # WS
        mlx_ws = decode_ws(ws_preds[i], spans[i], sent)
        pt_ws = pt["ws_out"][i]
        match = "✓" if mlx_ws == pt_ws else "✗"
        print(f"  WS {match}: {mlx_ws}")
        if mlx_ws != pt_ws:
            print(f"  PT   : {pt_ws}")

        # POS
        mlx_pos = decode_pos(ml["pos_preds"][i], mlx_ws, ml["pos_id2label"], spans[i])
        pt_pos = pt["pos_out"][i]
        match = "✓" if mlx_pos == pt_pos else "✗"
        print(f"  POS {match}: {mlx_pos}")
        if mlx_pos != pt_pos:
            print(f"  PT    : {pt_pos}")

        # NER
        mlx_ner = decode_ner(ml["ner_preds"][i], spans[i], sent, ml["ner_id2label"])
        pt_ner = [(e.word, e.ner, e.idx[0]) for e in pt["ner_out"][i]]
        match = "✓" if mlx_ner == pt_ner else "✗"
        print(f"  NER {match}: {mlx_ner}")
        if mlx_ner != pt_ner:
            print(f"  PT     : {pt_ner}")

if __name__ == "__main__":
    main()
