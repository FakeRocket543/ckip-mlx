# Changelog

## [0.1.0] - 2025-03-23

### 新增
- MLX 原生 BERT encoder 實作 (`bert_mlx.py`)
- HuggingFace → MLX 權重轉換工具 (`convert.py`)
- 支援三個 CKIP 任務：WS (斷詞) / POS (詞性) / NER (實體辨識)
- 三種精度版本：fp32 / fp16 / q8
- fp16/q8 量化版本產生工具 (`convert_variants.py`)
- 三框架完整 benchmark (MLX / HF Transformers / CKIP 官方)
- 維基百科「臺灣」條目 36,245 字長文測試與結果 JSON
- WordPiece tokenizer 純 Python 實作
- WS/POS/NER decode 邏輯

### Benchmark 結果 (Apple M4 Max)
- MLX fp32 比 CKIP 官方快 5.2 倍
- MLX fp32 比 HF Transformers (MPS) 快 1.23 倍
- fp16 精度幾乎無損 (WS 100%, POS 99.98%, NER 99.99%)
- q8 WS 完全無損, POS 99.69%, NER 99.90%
