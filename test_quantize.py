"""Test quantization impact on WS/POS/NER accuracy vs fp32 baseline."""

import json, os, copy
import mlx.core as mx
import mlx.nn as nn
from bert_mlx import BertForTokenClassification

# 一整篇長文
TEXT = """
台灣積體電路製造股份有限公司董事長劉德音今天在台北國際會議中心出席二零二三年全球半導體論壇，
發表專題演講。他表示台積電將在高雄楠梓科技園區投資超過一兆新台幣，興建七奈米及二十八奈米晶圓廠，
預計二零二六年開始量產。行政院長陳建仁隨後也到場致詞，強調政府將全力支持半導體產業發展。
經濟部長王美花指出，台灣半導體產業去年產值達到四兆三千億元，佔全球市場百分之二十六。
中央研究院院長廖俊智在會中發表研究報告，指出人工智慧晶片的需求將在未來五年成長三倍。
聯發科技執行長蔡力行也分享了該公司在五G通訊晶片領域的最新進展，預計明年推出新一代旗艦處理器。
台灣大學電機工程學系教授李琳山表示，台灣在半導體人才培育方面仍需加強，建議教育部增設相關學程。
鴻海精密工業創辦人郭台銘則透過視訊連線，從美國威斯康辛州的工廠發表談話，
他認為全球供應鏈正在重組，台灣必須把握這個歷史性的機遇。
會議最後由國家發展委員會主任委員龔明鑫做總結，他宣布將成立半導體產業發展基金，
初期規模為五百億元，用於支持新創企業及前瞻技術研發。
與會人士包括來自美國英特爾、韓國三星電子、日本東京威力科創等國際大廠的高階主管，
共同探討後摩爾定律時代的技術挑戰與商業機會。
""".strip().replace("\n", "")

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

    def encode(self, text):
        ids = [self.cls_id] + [self.vocab.get(ch, self.unk_id) for ch in text] + [self.sep_id]
        spans = [None] + list(range(len(text))) + [None]
        return mx.array([ids]), mx.array([[1]*len(ids)]), spans

def load_model(task_dir):
    with open(os.path.join(task_dir, "config.json")) as f:
        config = json.load(f)
    config["num_labels"] = len(config.get("id2label", {})) or config.get("num_labels", 2)
    model = BertForTokenClassification(config)
    model.load_weights(os.path.join(task_dir, "weights.safetensors"))
    mx.eval(model.parameters())
    return model, config

def get_preds(model, input_ids, attention_mask):
    out = model(input_ids, attention_mask=attention_mask)
    mx.eval(out)
    return mx.argmax(out, axis=-1).tolist()[0]

def quantize_model(model, bits):
    """Return a quantized copy."""
    q = copy.deepcopy(model)
    nn.quantize(q, bits=bits)
    mx.eval(q.parameters())
    return q

def compare(task, fp32_preds, q_preds, id2label, label):
    total = len(fp32_preds)
    diff = sum(1 for a, b in zip(fp32_preds, q_preds) if a != b)
    match_rate = (total - diff) / total * 100
    print(f"  {label}: {total - diff}/{total} 一致 ({match_rate:.2f}%)", end="")
    if diff > 0:
        print(f"  ← {diff} 個不同：", end="")
        shown = 0
        for i, (a, b) in enumerate(zip(fp32_preds, q_preds)):
            if a != b and shown < 5:
                la = id2label.get(str(a), str(a))
                lb = id2label.get(str(b), str(b))
                print(f" [{i}]{la}→{lb}", end="")
                shown += 1
        if diff > 5:
            print(f" ...+{diff-5}", end="")
    print()
    return diff

def main():
    tokenizer = WordPieceTokenizer("models/ws/vocab.txt")
    input_ids, attention_mask, spans = tokenizer.encode(TEXT)
    print(f"文章長度: {len(TEXT)} 字, token 數: {input_ids.shape[1]}\n")

    for task in ["ws", "pos", "ner"]:
        model, config = load_model(f"models/{task}")
        id2label = config.get("id2label", {})
        fp32_preds = get_preds(model, input_ids, attention_mask)

        print(f"{'='*60}")
        print(f"[{task.upper()}] labels={config['num_labels']}")

        for bits in [8, 4, 2]:
            q_model = quantize_model(model, bits)
            q_preds = get_preds(q_model, input_ids, attention_mask)
            compare(task, fp32_preds, q_preds, id2label, f"{bits}-bit")

        print()

if __name__ == "__main__":
    main()
