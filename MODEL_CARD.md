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

Apple MLX 原生版本的 CKIP BERT-base 繁體中文 NLP 模型。

## 模型說明

本模型從 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 的 HuggingFace 權重轉換為 MLX 格式，專為 Apple Silicon 優化。

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

## 速度

測試環境：Apple M4 Max / 128GB / MLX 0.31.0
測試資料：維基百科「臺灣」條目，36,245 字

| Framework | fp32 | fp16 |
|-----------|-----:|-----:|
| **MLX** | **2,869 ms** | 3,092 ms |
| HF Transformers (MPS) | 3,532 ms | 3,096 ms |
| CKIP 官方 (MPS) | 14,926 ms | 11,850 ms |

## 使用方式

```python
import json, os
import mlx.core as mx
from bert_mlx import BertForTokenClassification

task_dir = "models/ws-fp16"  # 推薦 fp16
with open(os.path.join(task_dir, "config.json")) as f:
    config = json.load(f)
config["num_labels"] = len(config.get("id2label", {}))

model = BertForTokenClassification(config)
model.load_weights(os.path.join(task_dir, "weights.safetensors"))
mx.eval(model.parameters())

# tokenize + inference
text = "台積電今天股價上漲"
vocab = {}
with open(os.path.join(task_dir, "vocab.txt")) as f:
    for i, line in enumerate(f):
        vocab[line.strip()] = i

ids = [vocab["[CLS]"]] + [vocab.get(ch, vocab["[UNK]"]) for ch in text] + [vocab["[SEP]"]]
logits = model(mx.array([ids]), attention_mask=mx.array([[1]*len(ids)]))
mx.eval(logits)
preds = mx.argmax(logits, axis=-1).tolist()[0]
```

## 量化精度詳細測試

以維基百科「臺灣」條目 36,245 字測試，與 fp32 逐 token 比對：

### fp16
- WS: 36,389 tokens 完全一致 ✓
- POS: 僅 1 token 不同 (「耗能」VJ→Na)
- NER: 多辨識出 1 個實體 (「官話」LANGUAGE)

### q8
- WS: 36,389 tokens 完全一致 ✓
- POS: 28 tokens 不同，多為相近詞性混淆 (Nc↔Nb, Nv↔VC)
- NER: 漏 14 / 多 16 個實體 (2,773 個中)

## 跨框架驗證

HF Transformers fp32 (MPS) 與 MLX fp32 的 WS/POS/NER 輸出**完全一致**，確認轉換正確。

## 授權

[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)，依循原始 [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) 授權。

## 致謝

- [CKIP Lab, 中央研究院資訊科學研究所](https://ckip.iis.sinica.edu.tw/) — 原始模型
- [Apple MLX](https://github.com/ml-explore/mlx) — ML 框架
