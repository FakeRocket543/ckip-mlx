# CKIP BERT-base 全框架測試報告

## 測試環境

- 晶片：Apple M4 Max
- 記憶體：128 GB
- 系統：macOS 26.3.1
- Python：3.14.3 (MLX/PyTorch)、3.13 (CoreML)
- PyTorch：2.10.0
- MLX：0.31.0
- CoreML Tools：9.0

## 測試資料

- 來源：維基百科「臺灣」條目 (CC BY-SA 4.0)
- 文字長度：36,245 字（清理後純中文+標點）
- 分段：72 chunks × 510 字 (BERT max_position = 512)
- 測速：10 runs，取 median

---

## 1. 速度比較

### WS + POS + NER 全跑（36,245 字）

| Framework | fp32 | fp16 | q8 |
|-----------|-----:|-----:|---:|
| **CoreML** | 2,879 ms | **2,352 ms** ⚡ | 3,240 ms |
| **MLX** | 2,869 ms | 3,092 ms | 3,206 ms |
| **HF Transformers (MPS)** | 3,532 ms | 3,096 ms | — |
| **CKIP 官方 (MPS)** | 14,926 ms | 11,850 ms | — |

- 最快：CoreML fp16 (2,352 ms)
- MLX fp32 次之 (2,869 ms)
- CKIP 官方最慢，MLX fp32 比它快 5.2 倍
- PyTorch q8：PyTorch 2.10 已移除舊量化 API，無法測試

---

## 2. 精度比較

所有精度以 MLX fp32 為 baseline，逐 token 比對（共 36,389 tokens）。

### MLX

| 版本 | WS | POS (60類) | NER (73類) |
|------|:--:|:----------:|:----------:|
| fp32 | baseline | baseline | baseline |
| fp16 | 100% ✓ | 99.997% (1 處) | 99.997% (多1實體) |
| q8 | 100% ✓ | 99.69% (28 處) | 99.90% (漏14/多16) |

### CoreML

| 版本 | WS | POS (60類) | NER (73類) |
|------|:--:|:----------:|:----------:|
| fp32 | 100% ✓ | 100% ✓ | 100% ✓ |
| fp16 | 100.00% (1 處) | 99.97% (11 處) | 99.99% (3 處) |
| q8 | 99.96% (13 處) | 98.83% (425 處) | 99.76% (89 處) |

### HF Transformers (MPS)

| 版本 | WS | POS | NER |
|------|:--:|:---:|:---:|
| fp32 | 100% ✓ | 100% ✓ | 100% ✓ |
| fp16 | 100% ✓ | 99.99% (3 處) | 99.99% (漏2/多3) |

### CKIP 官方 (MPS)

CKIP 官方 API 有自己的 tokenizer 和分句邏輯，pipeline 不同，無法逐 token 比對。
WS 前 20 詞抽樣比較：與 MLX/HF 完全一致。

---

## 3. 模型大小

### 單模型

| 版本 | MLX | CoreML |
|------|----:|-------:|
| fp32 | 388 MB | 388 MB |
| fp16 | 194 MB | 194 MB |
| q8 | 110 MB | 98 MB |

### 三模型 (WS+POS+NER) 全載

| 版本 | MLX | CoreML |
|------|----:|-------:|
| fp32 | 1,164 MB | 1,164 MB |
| fp16 | 582 MB | 582 MB |
| q8 | 330 MB | 294 MB |

---

## 4. 記憶體需求建議

| 裝置 RAM | 建議版本 | 三模型 RAM |
|----------|---------|--------:|
| 8 GB | MLX q8 或 CoreML q8 | 294~330 MB |
| 8 GB+ | MLX fp16 或 CoreML fp16 | 582 MB |
| 16 GB+ | fp32 (任何框架) | 1,164 MB |

---

## 5. 各框架特性比較

| | MLX | CoreML | HF Transformers | CKIP 官方 |
|---|:---:|:------:|:---------------:|:---------:|
| Apple Silicon 優化 | ✓ GPU | ✓ ANE+GPU | ✓ MPS | ✓ MPS |
| iOS 可用 | ✗ | ✓ | ✗ | ✗ |
| Python API | ✓ | ✓ | ✓ | ✓ |
| Swift API | ✗ | ✓ | ✗ | ✗ |
| 動態長度 1-512 | ✓ | ✓ | ✓ | ✓ |
| 量化 (q8) | ✓ | ✓ | ✗ (API 已移除) | ✗ |
| 安裝依賴 | mlx | coremltools | torch+transformers | torch+ckip-transformers |
| 最快精度 | fp32 | fp16 | fp16 | fp16 |

---

## 6. 量化精度總結

### WS（斷詞，2 labels）— 最耐量化

所有框架、所有精度，WS 幾乎完全無損。二元分類的 margin 夠大。

### POS（詞性，60 labels）— fp16 安全，q8 有風險

- fp16：所有框架 ≥99.97%，可安心使用
- q8：MLX 99.69% 可接受，CoreML 98.83% 偏低（425 處不同）

### NER（命名實體，73 labels BIOES）— fp16 安全，q8 尚可

- fp16：所有框架 ≥99.99%
- q8：MLX 99.90%，CoreML 99.76%

---

## 7. 結論

1. **桌面/伺服器推薦**：MLX fp32 或 fp16，最簡單、精度最好
2. **iOS/macOS App**：CoreML fp16，速度最快 (2,352ms)，可用 ANE
3. **低記憶體裝置**：CoreML q8 (294MB) 或 MLX q8 (330MB)
4. **跨框架驗證**：MLX fp32 = HF Transformers fp32 = CoreML fp32，三者完全一致
5. **vs CKIP 官方**：速度快 5 倍，結果一致，無需 ckip-transformers 依賴
