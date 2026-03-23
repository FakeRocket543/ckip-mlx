#!/usr/bin/env python3
"""CKIP Transformers baseline benchmark — ws / pos / ner"""

import time, json, statistics
import torch
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

MODEL = "bert-base"
DEVICE = -1  # load on CPU first, we'll move to MPS manually

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

def bench(name, fn, inputs, n=5):
    """Run fn n times, report stats."""
    times = []
    result = None
    for i in range(n):
        t0 = time.perf_counter()
        result = fn(inputs)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  run {i+1}: {elapsed:.1f} ms")
    avg = statistics.mean(times)
    med = statistics.median(times)
    print(f"  → avg {avg:.1f} ms | median {med:.1f} ms | per-sentence {avg/len(inputs):.1f} ms")
    return result

def main():
    print(f"Model: {MODEL}  Device: MPS")
    print(f"Sentences: {len(SENTENCES)}\n")

    mps = torch.device("mps")

    # --- Word Segmentation ---
    print("Loading WS model...")
    t0 = time.perf_counter()
    ws = CkipWordSegmenter(model=MODEL, device=DEVICE)
    ws.model.to(mps)
    ws.device = mps
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f} ms\n")

    print("WS benchmark:")
    ws_result = bench("WS", lambda s: ws(s), SENTENCES)

    # --- POS Tagging ---
    print("\nLoading POS model...")
    t0 = time.perf_counter()
    pos = CkipPosTagger(model=MODEL, device=DEVICE)
    pos.model.to(mps)
    pos.device = mps
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f} ms\n")

    print("POS benchmark:")
    pos_result = bench("POS", lambda s: pos(s, use_delim=True), ws_result)

    # --- NER ---
    print("\nLoading NER model...")
    t0 = time.perf_counter()
    ner = CkipNerChunker(model=MODEL, device=DEVICE)
    ner.model.to(mps)
    ner.device = mps
    print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f} ms\n")

    print("NER benchmark:")
    ner_result = bench("NER", lambda s: ner(s), SENTENCES)

    # --- Sample output ---
    print("\n" + "="*60)
    print("Sample output (sentence 0):")
    print(f"  原文: {SENTENCES[0]}")
    print(f"  斷詞: {ws_result[0]}")
    print(f"  詞性: {pos_result[0]}")
    print(f"  NER:  {ner_result[0]}")

if __name__ == "__main__":
    main()
