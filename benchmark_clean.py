"""
Clean 3-framework benchmark (re-run)
MLX / HF Transformers (MPS) / CKIP官方 (MPS)
fp32 + fp16 each; MLX also q8
Wikipedia 臺灣 (CC BY-SA 4.0), 36k chars
"""

import json, os, re, time, statistics, platform, subprocess, gc
import torch
import mlx.core as mx
import mlx.nn as mnn
from bert_mlx import BertForTokenClassification as MLXBert

MAX_SEQ = 510
N_RUNS = 10  # more runs for stability

# ── Text ──
with open("wiki_taiwan.txt") as f:
    raw = f.read()
text = re.sub(r'\s+', '', raw)
text = ''.join(ch for ch in text if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' or ch in '，。、；：！？「」『』（）—…──《》〈〉' or '\uff00' <= ch <= '\uffef')

# ── Tokenizer ──
class WPTokenizer:
    def __init__(self, path):
        self.vocab = {}
        with open(path) as f:
            for i, line in enumerate(f):
                self.vocab[line.strip()] = i
        self.unk = self.vocab.get("[UNK]", 100)
        self.cls = self.vocab.get("[CLS]", 101)
        self.sep = self.vocab.get("[SEP]", 102)

    def chunks(self, text):
        out = []
        for s in range(0, len(text), MAX_SEQ):
            c = text[s:s+MAX_SEQ]
            ids = [self.cls] + [self.vocab.get(ch, self.unk) for ch in c] + [self.sep]
            spans = [None] + list(range(len(c))) + [None]
            out.append((ids, spans, c))
        return out

# ── Decoders ──
def dec_ws(preds, spans, ct):
    words, cur = [], ""
    for i, s in enumerate(spans):
        if s is None: continue
        if preds[i] == 0 and cur: words.append(cur); cur = ct[s]
        else: cur += ct[s]
    if cur: words.append(cur)
    return words

def dec_ner(preds, spans, ct, id2l, g):
    ents, ctype, cs, ce = [], None, 0, 0
    for i, s in enumerate(spans):
        if s is None:
            if ctype: ents.append({"text":ct[cs:ce+1],"type":ctype,"start":cs+g}); ctype=None
            continue
        l = id2l.get(str(preds[i]),"O")
        if l.startswith("S-"):
            if ctype: ents.append({"text":ct[cs:ce+1],"type":ctype,"start":cs+g})
            ents.append({"text":ct[s],"type":l[2:],"start":s+g}); ctype=None
        elif l.startswith("B-"):
            if ctype: ents.append({"text":ct[cs:ce+1],"type":ctype,"start":cs+g})
            ctype,cs,ce = l[2:],s,s
        elif l.startswith("I-") and ctype: ce=s
        elif l.startswith("E-") and ctype:
            ce=s; ents.append({"text":ct[cs:ce+1],"type":ctype,"start":cs+g}); ctype=None
        else:
            if ctype: ents.append({"text":ct[cs:ce+1],"type":ctype,"start":cs+g}); ctype=None
    if ctype: ents.append({"text":ct[cs:ce+1],"type":ctype,"start":cs+g})
    return ents

# ══════════════════════════════════════════════════════════════
# MLX
# ══════════════════════════════════════════════════════════════
def mlx_bench(variant, chunks, configs):
    suffix = "" if variant=="fp32" else f"-{variant}"
    qbits = 8 if variant=="q8" else None
    models = {}
    for t in ["ws","pos","ner"]:
        d = f"models/{t}{suffix}"
        with open(os.path.join(d,"config.json")) as f: cfg=json.load(f)
        cfg["num_labels"]=len(cfg.get("id2label",{})) or cfg.get("num_labels",2)
        m=MLXBert(cfg)
        if qbits: mnn.quantize(m,bits=qbits)
        m.load_weights(os.path.join(d,"weights.safetensors"))
        mx.eval(m.parameters())
        models[t]=m

    def infer(task, ids):
        o=models[task](mx.array([ids]),attention_mask=mx.array([[1]*len(ids)]))
        mx.eval(o)
        return mx.argmax(o,axis=-1).tolist()[0]

    # warmup x2
    for _ in range(2):
        for ids,sp,ct in chunks[:5]:
            for t in models: infer(t,ids)

    # speed
    times=[]
    for _ in range(N_RUNS):
        t0=time.perf_counter()
        for ids,sp,ct in chunks:
            for t in models: infer(t,ids)
        times.append((time.perf_counter()-t0)*1000)

    # results (single run)
    pos_id2l=configs["pos"].get("id2label",{})
    ner_id2l=configs["ner"].get("id2label",{})
    all_ws,all_pos,all_ner=[],[],[]
    wi,g=0,0
    for ids,spans,ct in chunks:
        wp=infer("ws",ids); all_ws.extend(dec_ws(wp,spans,ct))
    wi=0
    for ids,spans,ct in chunks:
        pp=infer("pos",ids)
        cw,rem=[],len(ct)
        while wi<len(all_ws) and rem>0:
            w=all_ws[wi]
            if len(w)<=rem: cw.append(w);rem-=len(w);wi+=1
            else: break
        ci=0
        for word in cw:
            tag="?"
            for i,s in enumerate(spans):
                if s==ci: tag=pos_id2l.get(str(pp[i]),"?"); break
            all_pos.append({"word":word,"pos":tag}); ci+=len(word)
    g=0
    for ids,spans,ct in chunks:
        np=infer("ner",ids)
        all_ner.extend(dec_ner(np,spans,ct,ner_id2l,g)); g+=len(ct)

    del models; gc.collect()
    return statistics.median(times), all_ws, all_pos, all_ner

# ══════════════════════════════════════════════════════════════
# HF Transformers (MPS)
# ══════════════════════════════════════════════════════════════
def hf_bench(variant, chunks, configs):
    from transformers import BertForTokenClassification
    device=torch.device("mps")
    hf={"ws":"ckiplab/bert-base-chinese-ws","pos":"ckiplab/bert-base-chinese-pos","ner":"ckiplab/bert-base-chinese-ner"}
    models={}
    for t in ["ws","pos","ner"]:
        m=BertForTokenClassification.from_pretrained(hf[t])
        if variant=="fp16": m=m.half()
        models[t]=m.to(device).eval()

    def infer(task,ids):
        t_ids=torch.tensor([ids],device=device)
        with torch.no_grad():
            o=models[task](t_ids,attention_mask=torch.ones_like(t_ids)).logits
        torch.mps.synchronize()
        return o.argmax(-1).cpu().tolist()[0]

    # warmup x2
    for _ in range(2):
        for ids,sp,ct in chunks[:5]:
            for t in models: infer(t,ids)

    times=[]
    for _ in range(N_RUNS):
        t0=time.perf_counter()
        for ids,sp,ct in chunks:
            for t in models: infer(t,ids)
        times.append((time.perf_counter()-t0)*1000)

    pos_id2l=configs["pos"].get("id2label",{})
    ner_id2l=configs["ner"].get("id2label",{})
    all_ws,all_pos,all_ner=[],[],[]
    for ids,spans,ct in chunks:
        all_ws.extend(dec_ws(infer("ws",ids),spans,ct))
    wi=0
    for ids,spans,ct in chunks:
        pp=infer("pos",ids)
        cw,rem=[],len(ct)
        while wi<len(all_ws) and rem>0:
            w=all_ws[wi]
            if len(w)<=rem: cw.append(w);rem-=len(w);wi+=1
            else: break
        ci=0
        for word in cw:
            tag="?"
            for i,s in enumerate(spans):
                if s==ci: tag=pos_id2l.get(str(pp[i]),"?"); break
            all_pos.append({"word":word,"pos":tag}); ci+=len(word)
    g=0
    for ids,spans,ct in chunks:
        all_ner.extend(dec_ner(infer("ner",ids),spans,ct,ner_id2l,g)); g+=len(ct)

    del models; gc.collect(); torch.mps.empty_cache()
    return statistics.median(times), all_ws, all_pos, all_ner

# ══════════════════════════════════════════════════════════════
# CKIP 官方
# ══════════════════════════════════════════════════════════════
def ckip_bench(variant):
    from ckip_transformers.nlp import CkipWordSegmenter,CkipPosTagger,CkipNerChunker
    device=torch.device("mps")
    sents=[s+"。" for s in text.split("。") if s]

    ws=CkipWordSegmenter(model="bert-base",device=-1)
    pos=CkipPosTagger(model="bert-base",device=-1)
    ner=CkipNerChunker(model="bert-base",device=-1)
    for obj in [ws,pos,ner]:
        if variant=="fp16": obj.model=obj.model.half().to(device)
        else: obj.model.to(device)
        obj.device=device

    # warmup x2
    for _ in range(2): ws(sents[:10]); torch.mps.synchronize()

    times=[]
    for _ in range(N_RUNS):
        t0=time.perf_counter()
        wo=ws(sents); pos(wo,use_delim=True); ner(sents)
        torch.mps.synchronize()
        times.append((time.perf_counter()-t0)*1000)

    wo=ws(sents); po=pos(wo,use_delim=True); no=ner(sents)
    torch.mps.synchronize()

    all_ws=[w for s in wo for w in s]
    all_pos=[{"word":w,"pos":p} for sw,sp in zip(wo,po) for w,p in zip(sw,sp)]
    all_ner=[]
    g=0
    for si,se in enumerate(no):
        for e in se: all_ner.append({"text":e.word,"type":e.ner,"start":e.idx[0]+g})
        g+=len(sents[si])

    del ws,pos,ner; gc.collect(); torch.mps.empty_cache()
    return statistics.median(times), all_ws, all_pos, all_ner

# ══════════════════════════════════════════════════════════════
def diff_vs(bws,bpos,bner,vws,vpos,vner):
    ws_ok = "✓" if vws==bws else f"差 (詞數 {len(bws)} vs {len(vws)})"
    minp=min(len(bpos),len(vpos))
    pd=sum(1 for i in range(minp) if bpos[i]["pos"]!=vpos[i]["pos"])+abs(len(bpos)-len(vpos))
    pos_s = "✓" if pd==0 else f"{pd} 處"
    bs={(e["text"],e["type"],e["start"]) for e in bner}
    vs={(e["text"],e["type"],e["start"]) for e in vner}
    nm,ne=len(bs-vs),len(vs-bs)
    ner_s = "✓" if nm==0 and ne==0 else f"漏{nm}/多{ne}"
    return ws_s if 'ws_s' in dir() else ws_ok, pos_s, ner_s

def main():
    chip=subprocess.run(["sysctl","-n","machdep.cpu.brand_string"],capture_output=True,text=True).stdout.strip()
    mem_gb=int(subprocess.run(["sysctl","-n","hw.memsize"],capture_output=True,text=True).stdout.strip())/1024**3
    mlx_ver=subprocess.run([".venv/bin/python3","-c","import mlx.core;print(mlx.core.__version__)"],capture_output=True,text=True).stdout.strip()

    print(f"{'='*74}")
    print(f"  CKIP BERT-base 三框架完整比較 (clean re-run)")
    print(f"{'='*74}")
    print(f"  文字：維基百科「臺灣」(CC BY-SA 4.0), {len(text)} 字")
    print(f"  環境：{chip} / {mem_gb:.0f}GB RAM")
    print(f"  版本：Python {platform.python_version()} / PyTorch {torch.__version__} / MLX {mlx_ver}")

    tok=WPTokenizer("models/ws/vocab.txt")
    chunks=tok.chunks(text)
    print(f"  分段：{len(chunks)} chunks × max {MAX_SEQ} 字 (BERT max_position=512)")
    print(f"  測速：{N_RUNS} runs, 取 median")

    configs={}
    for t in ["ws","pos","ner"]:
        with open(f"models/{t}/config.json") as f: configs[t]=json.load(f)

    R={}  # key -> (speed, ws, pos, ner)

    tests = [
        ("mlx-fp32",    lambda: mlx_bench("fp32", chunks, configs)),
        ("mlx-fp16",    lambda: mlx_bench("fp16", chunks, configs)),
        ("mlx-q8",      lambda: mlx_bench("q8",   chunks, configs)),
        ("hf-fp32",     lambda: hf_bench("fp32",  chunks, configs)),
        ("hf-fp16",     lambda: hf_bench("fp16",  chunks, configs)),
        ("ckip-fp32",   lambda: ckip_bench("fp32")),
        ("ckip-fp16",   lambda: ckip_bench("fp16")),
    ]

    for key, fn in tests:
        print(f"\n  ▶ {key}...", end=" ", flush=True)
        speed, ws, pos, ner = fn()
        R[key] = (speed, ws, pos, ner)
        print(f"{speed:.0f}ms  (ws:{len(ws)} pos:{len(pos)} ner:{len(ner)})")
        with open(f"result_{key.replace('-','_')}.json","w",encoding="utf-8") as f:
            json.dump({"key":key,"ws":ws,"pos":pos,"ner":ner},f,ensure_ascii=False,indent=2)

    # ── Speed table ──
    print(f"\n{'='*74}")
    print(f"  速度比較 (WS+POS+NER 全跑, {len(text)} 字, {N_RUNS} runs median)")
    print(f"{'='*74}")
    print(f"  {'Framework':<24}│ {'fp32':>10} │ {'fp16':>10} │ {'q8':>10}")
    print(f"  {'─'*24}┼{'─'*12}┼{'─'*12}┼{'─'*12}")
    for fw,label in [("mlx","MLX"),("hf","HF Transformers/MPS"),("ckip","CKIP 官方/MPS")]:
        vals=[]
        for v in ["fp32","fp16","q8"]:
            k=f"{fw}-{v}"
            if k in R: vals.append(f"{R[k][0]:.0f}ms")
            else: vals.append("—")
        print(f"  {label:<24}│ {vals[0]:>10} │ {vals[1]:>10} │ {vals[2]:>10}")

    # ── Accuracy table (vs mlx-fp32) ──
    print(f"\n{'='*74}")
    print(f"  精度比較 (vs mlx-fp32)")
    print(f"{'='*74}")
    _,bws,bpos,bner = R["mlx-fp32"]
    print(f"  {'Config':<24}│ {'WS':>10} │ {'POS':>12} │ {'NER':>14}")
    print(f"  {'─'*24}┼{'─'*12}┼{'─'*14}┼{'─'*16}")
    for key in ["mlx-fp16","mlx-q8","hf-fp32","hf-fp16","ckip-fp32","ckip-fp16"]:
        if key not in R: continue
        _,vws,vpos,vner = R[key]
        ws_s,pos_s,ner_s = diff_vs(bws,bpos,bner,vws,vpos,vner)
        print(f"  {key:<24}│ {ws_s:>10} │ {pos_s:>12} │ {ner_s:>14}")

    # ── Cross-framework WS spot check ──
    print(f"\n{'='*74}")
    print(f"  WS 前 20 詞抽樣比較")
    print(f"{'='*74}")
    for key in ["mlx-fp32","hf-fp32","ckip-fp32"]:
        _,ws,_,_ = R[key]
        print(f"  {key:<14}: {'｜'.join(ws[:20])}")

    print(f"\n  所有 JSON 已儲存 ✓")

if __name__=="__main__":
    main()
