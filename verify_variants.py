"""Verify fp16 and q8 variants against fp32 baseline."""

import json, os
import mlx.core as mx
import mlx.nn as nn
from bert_mlx import BertForTokenClassification

TEXT = """台灣積體電路製造股份有限公司董事長劉德音今天在台北國際會議中心出席二零二三年全球半導體論壇，\
發表專題演講。他表示台積電將在高雄楠梓科技園區投資超過一兆新台幣，興建七奈米及二十八奈米晶圓廠，\
預計二零二六年開始量產。行政院長陳建仁隨後也到場致詞，強調政府將全力支持半導體產業發展。\
經濟部長王美花指出，台灣半導體產業去年產值達到四兆三千億元，佔全球市場百分之二十六。\
中央研究院院長廖俊智在會中發表研究報告，指出人工智慧晶片的需求將在未來五年成長三倍。\
聯發科技執行長蔡力行也分享了該公司在五G通訊晶片領域的最新進展，預計明年推出新一代旗艦處理器。\
台灣大學電機工程學系教授李琳山表示，台灣在半導體人才培育方面仍需加強，建議教育部增設相關學程。\
鴻海精密工業創辦人郭台銘則透過視訊連線，從美國威斯康辛州的工廠發表談話，\
他認為全球供應鏈正在重組，台灣必須把握這個歷史性的機遇。\
會議最後由國家發展委員會主任委員龔明鑫做總結，他宣布將成立半導體產業發展基金，\
初期規模為五百億元，用於支持新創企業及前瞻技術研發。\
與會人士包括來自美國英特爾、韓國三星電子、日本東京威力科創等國際大廠的高階主管，\
共同探討後摩爾定律時代的技術挑戰與商業機會。"""

class WordPieceTokenizer:
    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path) as f:
            for i, line in enumerate(f):
                self.vocab[line.strip()] = i
        self.unk_id = self.vocab.get("[UNK]", 100)
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)

    def encode(self, text):
        ids = [self.cls_id] + [self.vocab.get(ch, self.unk_id) for ch in text] + [self.sep_id]
        return mx.array([ids]), mx.array([[1]*len(ids)])

def load_and_predict(task_dir, input_ids, mask):
    with open(os.path.join(task_dir, "config.json")) as f:
        config = json.load(f)
    config["num_labels"] = len(config.get("id2label", {})) or config.get("num_labels", 2)
    model = BertForTokenClassification(config)
    weights_path = os.path.join(task_dir, "weights.safetensors")

    # Check if quantized (q8 dir will have quantized weights)
    if "-q8" in task_dir:
        nn.quantize(model, bits=8)
    model.load_weights(weights_path)
    mx.eval(model.parameters())

    out = model(input_ids, attention_mask=mask)
    mx.eval(out)
    return mx.argmax(out, axis=-1).tolist()[0]

def main():
    tok = WordPieceTokenizer("models/ws/vocab.txt")
    input_ids, mask = tok.encode(TEXT)
    print(f"文章: {len(TEXT)} 字\n")

    for task in ["ws", "pos", "ner"]:
        with open(f"models/{task}/config.json") as f:
            id2label = json.load(f).get("id2label", {})

        fp32 = load_and_predict(f"models/{task}", input_ids, mask)
        fp16 = load_and_predict(f"models/{task}-fp16", input_ids, mask)
        q8   = load_and_predict(f"models/{task}-q8", input_ids, mask)

        total = len(fp32)
        fp16_diff = sum(1 for a, b in zip(fp32, fp16) if a != b)
        q8_diff   = sum(1 for a, b in zip(fp32, q8) if a != b)

        print(f"[{task.upper()}]")
        print(f"  fp16: {total-fp16_diff}/{total} 一致 ({(total-fp16_diff)/total*100:.2f}%)", end="")
        if fp16_diff:
            diffs = [(i, id2label.get(str(a),"?"), id2label.get(str(b),"?")) for i,(a,b) in enumerate(zip(fp32,fp16)) if a!=b]
            print(f"  差異: {diffs[:5]}")
        else:
            print(" ✓")

        print(f"  q8:   {total-q8_diff}/{total} 一致 ({(total-q8_diff)/total*100:.2f}%)", end="")
        if q8_diff:
            diffs = [(i, id2label.get(str(a),"?"), id2label.get(str(b),"?")) for i,(a,b) in enumerate(zip(fp32,q8)) if a!=b]
            print(f"  差異: {diffs[:5]}")
        else:
            print(" ✓")
        print()

if __name__ == "__main__":
    main()
