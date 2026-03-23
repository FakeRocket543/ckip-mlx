# CKIP BERT MLX — 繁體中文 NLP (斷詞/詞性/實體辨識)

CKIP BERT-base 的 Apple MLX 原生實作，從 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 的 HuggingFace 權重轉換而來。

在 Apple Silicon 上比 CKIP 官方 PyTorch 版本快 **5 倍**，比 HuggingFace Transformers (MPS) 快 **20%**。

## 功能

- **WS** — 中文斷詞 (Word Segmentation)
- **POS** — 詞性標注 (Part-of-Speech Tagging, 60 類)
- **NER** — 命名實體辨識 (Named Entity Recognition, 73 類 BIOES)

## 模型版本

| 版本 | 單模型大小 | 三模型全載 RAM | 適合機器 | 精度 (vs fp32) |
|------|--------:|------------:|---------|--------------|
| fp32 | 388 MB | 1.2 GB | 16GB+ Mac | baseline |
| fp16 | 194 MB | 582 MB | 8GB+ Mac | WS 100% / POS 99.98% / NER 99.99% |
| q8 | 110 MB | 330 MB | 8GB Mac (吃緊) | WS 100% / POS 99.69% / NER 99.90% |

## 下載模型

模型權重託管於 HuggingFace：[FakeRockert543/ckip-mlx](https://huggingface.co/FakeRockert543/ckip-mlx)

```bash
# 需要先安裝 huggingface_hub
pip install huggingface_hub

# 下載全部模型
huggingface-cli download FakeRockert543/ckip-mlx --local-dir models
```

## 安裝

```bash
# 建立虛擬環境
python3 -m venv .venv && source .venv/bin/activate

# 安裝依賴
pip install mlx safetensors
```

## 快速開始

```python
import json, os
import mlx.core as mx
from bert_mlx import BertForTokenClassification

# 載入模型 (可替換為 fp16 或 q8 版本)
task_dir = "models/ws"  # 或 models/ws-fp16, models/ws-q8
with open(os.path.join(task_dir, "config.json")) as f:
    config = json.load(f)
config["num_labels"] = len(config.get("id2label", {}))

model = BertForTokenClassification(config)
model.load_weights(os.path.join(task_dir, "weights.safetensors"))
mx.eval(model.parameters())

# Tokenize (單字切分)
text = "台積電今天股價上漲三十元"
vocab = {}
with open(os.path.join(task_dir, "vocab.txt")) as f:
    for i, line in enumerate(f):
        vocab[line.strip()] = i

ids = [vocab["[CLS]"]] + [vocab.get(ch, vocab["[UNK]"]) for ch in text] + [vocab["[SEP]"]]
input_ids = mx.array([ids])
attention_mask = mx.array([[1] * len(ids)])

# 推論
logits = model(input_ids, attention_mask=attention_mask)
mx.eval(logits)
preds = mx.argmax(logits, axis=-1).tolist()[0]

# 解碼斷詞 (B=0: 詞首, I=1: 詞中)
words, cur = [], ""
for i, ch in enumerate(text):
    p = preds[i + 1]  # +1 for [CLS]
    if p == 0 and cur:
        words.append(cur)
        cur = ch
    else:
        cur += ch
if cur:
    words.append(cur)
print(words)
# ['台積電', '今天', '股價', '上漲', '三十', '元']
```

## 從 HuggingFace 權重轉換

如果你想自己從頭轉換：

```bash
pip install torch ckip-transformers huggingface_hub
python convert.py
```

## 產生量化版本

```bash
python convert_variants.py
```

## Benchmark

### 測試環境

- Apple M4 Max / 128GB RAM
- macOS / Python 3.14.3
- PyTorch 2.10.0 / MLX 0.31.0

### 測試資料

維基百科「臺灣」條目 (CC BY-SA 4.0)，36,245 字，72 chunks × 510 字

### 速度比較 (WS+POS+NER 全跑, 10 runs median)

| Framework | fp32 | fp16 |
|-----------|-----:|-----:|
| **MLX** | **2,869 ms** | 3,092 ms |
| HF Transformers (MPS) | 3,532 ms | 3,096 ms |
| CKIP 官方 (MPS) | 14,926 ms | 11,850 ms |

MLX fp32 vs CKIP 官方 fp32: **5.2x 加速**
MLX fp32 vs HF Transformers fp32: **1.23x 加速**

### 量化精度 (36,245 字長文測試, vs fp32 baseline)

| 版本 | WS | POS (60類) | NER (73類) |
|------|:--:|:----------:|:----------:|
| fp16 | 100% ✓ | 99.98% (1 處不同) | 99.99% (多 1 實體) |
| q8 | 100% ✓ | 99.69% (28 處不同) | 99.90% (漏14/多16) |

### 跨框架精度驗證

HF Transformers fp32 與 MLX fp32 的 WS/POS/NER 結果**完全一致**，確認實作正確。

## 檔案結構

```
├── bert_mlx.py          # MLX BERT encoder (<100 行)
├── convert.py           # HuggingFace → MLX 權重轉換
├── convert_variants.py  # 產生 fp16 / q8 版本
├── benchmark_clean.py   # 三框架完整 benchmark
├── demo_wiki.py         # 維基百科長文示範
├── models/
│   ├── ws/              # 斷詞 fp32
│   ├── ws-fp16/         # 斷詞 fp16
│   ├── ws-q8/           # 斷詞 q8
│   ├── pos/             # 詞性 fp32
│   ├── pos-fp16/
│   ├── pos-q8/
│   ├── ner/             # 實體 fp32
│   ├── ner-fp16/
│   └── ner-q8/
└── wiki_taiwan_*.json   # 測試結果 JSON
```

## 授權

本專案依循原始 CKIP Transformers 的授權條款，採用 [GPL-3.0 License](LICENSE)。

原始模型權重來自 [ckiplab](https://github.com/ckiplab/ckip-transformers)（中央研究院資訊科學研究所）。

## 致謝

- [CKIP Lab](https://ckip.iis.sinica.edu.tw/) — 原始 BERT 模型訓練與發布
- [Apple MLX](https://github.com/ml-explore/mlx) — Apple Silicon 機器學習框架
