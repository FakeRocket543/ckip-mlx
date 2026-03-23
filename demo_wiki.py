"""Run WS/POS/NER on Wikipedia article, save results as JSON."""

import json, os, re
import mlx.core as mx
from bert_mlx import BertForTokenClassification

MAX_SEQ = 510

with open("wiki_taiwan.txt") as f:
    raw = f.read()
text = re.sub(r'\s+', '', raw)
cleaned = [ch for ch in text if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' or ch in '，。、；：！？「」『』（）—…──《》〈〉' or '\uff00' <= ch <= '\uffef']
text = ''.join(cleaned)

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

def load_model(task_dir):
    with open(os.path.join(task_dir, "config.json")) as f:
        config = json.load(f)
    config["num_labels"] = len(config.get("id2label", {})) or config.get("num_labels", 2)
    model = BertForTokenClassification(config)
    model.load_weights(os.path.join(task_dir, "weights.safetensors"))
    mx.eval(model.parameters())
    return model, config

def decode_ws(preds, spans, text):
    words, cur = [], ""
    for i, s in enumerate(spans):
        if s is None: continue
        ch = text[s]
        if preds[i] == 0 and cur:
            words.append(cur)
            cur = ch
        else:
            cur += ch
    if cur: words.append(cur)
    return words

def decode_pos(preds, ws_words, id2label, spans):
    tags, ci = [], 0
    for word in ws_words:
        for i, s in enumerate(spans):
            if s == ci:
                tags.append(id2label.get(str(preds[i]), "?"))
                break
        else:
            tags.append("?")
        ci += len(word)
    return tags

def decode_ner(preds, spans, text, id2label):
    entities, cur_type, cur_start, cur_end = [], None, 0, 0
    for i, s in enumerate(spans):
        if s is None:
            if cur_type:
                entities.append({"text": text[cur_start:cur_end+1], "type": cur_type, "start": cur_start})
                cur_type = None
            continue
        label = id2label.get(str(preds[i]), "O")
        if label.startswith("S-"):
            if cur_type:
                entities.append({"text": text[cur_start:cur_end+1], "type": cur_type, "start": cur_start})
            entities.append({"text": text[s], "type": label[2:], "start": s})
            cur_type = None
        elif label.startswith("B-"):
            if cur_type:
                entities.append({"text": text[cur_start:cur_end+1], "type": cur_type, "start": cur_start})
            cur_type, cur_start, cur_end = label[2:], s, s
        elif label.startswith("I-") and cur_type:
            cur_end = s
        elif label.startswith("E-") and cur_type:
            cur_end = s
            entities.append({"text": text[cur_start:cur_end+1], "type": cur_type, "start": cur_start})
            cur_type = None
        else:
            if cur_type:
                entities.append({"text": text[cur_start:cur_end+1], "type": cur_type, "start": cur_start})
                cur_type = None
    if cur_type:
        entities.append({"text": text[cur_start:cur_end+1], "type": cur_type, "start": cur_start})
    return entities

tokenizer = WordPieceTokenizer("models/ws/vocab.txt")
chunks = tokenizer.encode_chunks(text)
print(f"文字: {len(text)} 字, {len(chunks)} chunks")

# WS
ws_model, ws_config = load_model("models/ws")
all_ws_words = []
offset = 0
for input_ids, mask, spans, chunk_text in chunks:
    out = ws_model(input_ids, attention_mask=mask)
    mx.eval(out)
    preds = mx.argmax(out, axis=-1).tolist()[0]
    words = decode_ws(preds, spans, chunk_text)
    all_ws_words.extend(words)
print(f"WS: {len(all_ws_words)} 詞")

# POS
pos_model, pos_config = load_model("models/pos")
pos_id2label = pos_config.get("id2label", {})
all_pos = []
wi = 0
for input_ids, mask, spans, chunk_text in chunks:
    out = pos_model(input_ids, attention_mask=mask)
    mx.eval(out)
    preds = mx.argmax(out, axis=-1).tolist()[0]
    # figure out which ws_words belong to this chunk
    chunk_words = []
    remaining = len(chunk_text)
    while wi < len(all_ws_words) and remaining > 0:
        w = all_ws_words[wi]
        if len(w) <= remaining:
            chunk_words.append(w)
            remaining -= len(w)
            wi += 1
        else:
            break
    tags = decode_pos(preds, chunk_words, pos_id2label, spans)
    all_pos.extend(list(zip(chunk_words, tags)))
print(f"POS: {len(all_pos)} 詞性標注")

# NER
ner_model, ner_config = load_model("models/ner")
ner_id2label = ner_config.get("id2label", {})
all_ner = []
global_offset = 0
for input_ids, mask, spans, chunk_text in chunks:
    out = ner_model(input_ids, attention_mask=mask)
    mx.eval(out)
    preds = mx.argmax(out, axis=-1).tolist()[0]
    ents = decode_ner(preds, spans, chunk_text, ner_id2label)
    for e in ents:
        e["start"] += global_offset
    all_ner.extend(ents)
    global_offset += len(chunk_text)
print(f"NER: {len(all_ner)} 實體")

# Save
result = {
    "source": "維基百科「臺灣」條目 (CC BY-SA 4.0)",
    "text_length": len(text),
    "ws": all_ws_words,
    "pos": [{"word": w, "pos": t} for w, t in all_pos],
    "ner": all_ner,
}

out_path = "wiki_taiwan_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n已儲存 → {out_path}")

# Preview
print(f"\n── WS 前 50 詞 ──")
print("｜".join(all_ws_words[:50]))
print(f"\n── POS 前 30 ──")
for w, t in all_pos[:30]:
    print(f"  {w}/{t}", end="")
print(f"\n\n── NER 前 20 ──")
for e in all_ner[:20]:
    print(f"  [{e['type']}] {e['text']} @{e['start']}")
