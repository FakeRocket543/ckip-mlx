---
language:
  - zh
license: gpl-3.0
library_name: mlx
tags:
  - mlx
  - bert
  - token-classification
  - word-segmentation
  - pos-tagging
  - named-entity-recognition
  - traditional-chinese
  - ckip
datasets:
  - ckiplab/ckip-transformers
base_model:
  - ckiplab/bert-base-chinese-ws
  - ckiplab/bert-base-chinese-pos
  - ckiplab/bert-base-chinese-ner
---

# CKIP BERT-base MLX — 繁體中文 WS/POS/NER

Apple MLX 原生版本的 CKIP BERT-base 繁體中文 NLP 模型，專為 Apple Silicon 最佳化。

- 📦 原始碼：[GitHub — FakeRocket543/ckip-mlx](https://github.com/FakeRocket543/ckip-mlx)
- 🤗 模型權重：[HuggingFace — FakeRockert543/ckip-mlx](https://huggingface.co/FakeRockert543/ckip-mlx)（本頁）
- 🍎 CoreML 版本：[HuggingFace — FakeRockert543/ckip-coreml](https://huggingface.co/FakeRockert543/ckip-coreml)

從 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 的 HuggingFace 權重轉換而來。在 Apple Silicon 上比 CKIP 官方 PyTorch 版本快 **5 倍**。

## 模型說明

| 任務 | 說明 | 標籤數 | 原始模型 |
|------|------|------:|---------|
| WS | 中文斷詞 | 2 (B/I) | ckiplab/bert-base-chinese-ws |
| POS | 詞性標注 | 60 | ckiplab/bert-base-chinese-pos |
| NER | 命名實體辨識 | 73 (BIOES) | ckiplab/bert-base-chinese-ner |

## 可用版本

| 版本 | 單模型大小 | 精度 (vs fp32) | 建議用途 |
|------|--------:|--------------|---------|
| fp32 | 388 MB | baseline | 追求完全精度 |
| fp16 | 194 MB | WS 100% / POS 99.98% / NER 99.99% | **推薦預設** |
| q8 | 110 MB | WS 100% / POS 99.69% / NER 99.90% | 8GB RAM Mac |

## 從零開始使用（完整步驟）

### 1. 環境準備

需要 macOS + Apple Silicon (M1/M2/M3/M4)。

```bash
# 建立專案目錄
git clone https://github.com/FakeRocket543/ckip-mlx.git
cd ckip-mlx

# 建立虛擬環境
python3 -m venv .venv && source .venv/bin/activate

# 安裝依賴
pip install mlx safetensors huggingface_hub
```

### 2. 下載模型權重

```bash
# 從 HuggingFace 下載全部模型到 models/ 目錄
huggingface-cli download FakeRockert543/ckip-mlx --local-dir models
```

下載後目錄結構：

```
ckip-mlx/
├── bert_mlx.py          # MLX BERT 推論引擎
├── models/
│   ├── ws/              # 斷詞 fp32
│   ├── ws-fp16/         # 斷詞 fp16（推薦）
│   ├── ws-q8/           # 斷詞 q8
│   ├── pos/             # 詞性 fp32
│   ├── pos-fp16/
│   ├── pos-q8/
│   ├── ner/             # 實體 fp32
│   ├── ner-fp16/
│   └── ner-q8/
```

### 3. 執行斷詞

```python
import json, os
import mlx.core as mx
from bert_mlx import BertForTokenClassification

# 載入模型（推薦 fp16）
task_dir = "models/ws-fp16"
with open(os.path.join(task_dir, "config.json")) as f:
    config = json.load(f)
config["num_labels"] = len(config.get("id2label", {}))

model = BertForTokenClassification(config)
model.load_weights(os.path.join(task_dir, "weights.safetensors"))
mx.eval(model.parameters())

# 載入詞表
vocab = {}
with open(os.path.join(task_dir, "vocab.txt")) as f:
    for i, line in enumerate(f):
        vocab[line.strip()] = i

# Tokenize（BERT 單字切分）
text = "台積電今天股價上漲三十元"
ids = [vocab["[CLS]"]] + [vocab.get(ch, vocab["[UNK]"]) for ch in text] + [vocab["[SEP]"]]
input_ids = mx.array([ids])
attention_mask = mx.array([[1] * len(ids)])

# 推論
logits = model(input_ids, attention_mask=attention_mask)
mx.eval(logits)
preds = mx.argmax(logits, axis=-1).tolist()[0]

# 解碼斷詞結果（B=0: 詞首, I=1: 詞中）
words, cur = [], ""
for i, ch in enumerate(text):
    p = preds[i + 1]  # +1 跳過 [CLS]
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

### 4. 執行詞性標注

```python
# 載入 POS 模型
pos_dir = "models/pos-fp16"
with open(os.path.join(pos_dir, "config.json")) as f:
    pos_config = json.load(f)
pos_config["num_labels"] = len(pos_config.get("id2label", {}))

pos_model = BertForTokenClassification(pos_config)
pos_model.load_weights(os.path.join(pos_dir, "weights.safetensors"))
mx.eval(pos_model.parameters())

# 推論（input_ids 同上）
logits = pos_model(input_ids, attention_mask=attention_mask)
mx.eval(logits)
preds = mx.argmax(logits, axis=-1).tolist()[0]

# 解碼詞性
id2label = pos_config["id2label"]
for i, ch in enumerate(text):
    label = id2label[str(preds[i + 1])]
    print(f"{ch} → {label}")
# 台 → Nb  積 → Nb  電 → Nb  今 → Nd  天 → Nd ...
```

### 5. 執行命名實體辨識

```python
# 載入 NER 模型
ner_dir = "models/ner-fp16"
with open(os.path.join(ner_dir, "config.json")) as f:
    ner_config = json.load(f)
ner_config["num_labels"] = len(ner_config.get("id2label", {}))

ner_model = BertForTokenClassification(ner_config)
ner_model.load_weights(os.path.join(ner_dir, "weights.safetensors"))
mx.eval(ner_model.parameters())

# 推論
logits = ner_model(input_ids, attention_mask=attention_mask)
mx.eval(logits)
preds = mx.argmax(logits, axis=-1).tolist()[0]

# 解碼實體（BIOES 格式）
id2label = ner_config["id2label"]
for i, ch in enumerate(text):
    label = id2label[str(preds[i + 1])]
    if label != "O":
        print(f"{ch} → {label}")
# 台 → B-ORG  積 → I-ORG  電 → E-ORG
```

## 速度

測試環境：Apple M4 Max / 128GB / MLX 0.31.0
測試資料：維基百科「臺灣」條目，36,245 字，10 runs median

| Framework | fp32 | fp16 |
|-----------|-----:|-----:|
| **MLX** | **2,869 ms** | 3,092 ms |
| HF Transformers (MPS) | 3,532 ms | 3,096 ms |
| CKIP 官方 (MPS) | 14,926 ms | 11,850 ms |

## 跨框架驗證

HF Transformers fp32 (MPS) 與 MLX fp32 的 WS/POS/NER 輸出**完全一致**，確認轉換正確。

## 授權

[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)，依循原始 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 授權。

## 致謝

- [CKIP Lab, 中央研究院資訊科學研究所](https://ckip.iis.sinica.edu.tw/) — 原始模型
- [Apple MLX](https://github.com/ml-explore/mlx) — ML 框架
