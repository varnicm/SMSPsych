#!/usr/bin/env python
"""Smishing-only: PT & PF micro AND macro P/R/F1, + SPR sweep 50-100%."""
import sys, json
from collections import Counter, defaultdict
from pathlib import Path
GOLD_FILES=["gold_final.jsonl","gold_final_batch2.jsonl"]; TAU=30
SCORE_GOLD="gold_final.jsonl"; THRS=[0.5,0.6,0.7,0.8,0.9,1.0]
def load(p):
    d={}
    for l in open(p,encoding="utf-8"):
        l=l.strip()
        if not l: continue
        r=json.loads(l)
        d[r["id"]]={"pts":set(x.strip().lower() for x in r.get("pts",[])),
                    "pfs":set(x.strip().lower() for x in r.get("pfs",[])),
                    "source":r.get("source","unknown")}
    return d
def evalset():
    pt=Counter(); pf=Counter()
    for fn in GOLD_FILES:
        if not Path(fn).exists(): continue
        for l in open(fn,encoding="utf-8"):
            l=l.strip()
            if not l: continue
            r=json.loads(l)
            for x in set(v.strip().lower() for v in r.get("pts",[])): pt[x]+=1
            for x in set(v.strip().lower() for v in r.get("pfs",[])): pf[x]+=1
    return {l for l,c in pt.items() if c>=TAU},{l for l,c in pf.items() if c>=TAU}
def counts(gold,pred,kind,ids,allowed):
    per=defaultdict(lambda:[0,0,0])
    for i in ids:
        g=gold[i][kind]&allowed; p=pred[i][kind]&allowed
        for lab in g&p: per[lab][0]+=1
        for lab in p-g: per[lab][1]+=1
        for lab in g-p: per[lab][2]+=1
    return per
def prf(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp else 0.0; R=tp/(tp+fn) if tp+fn else 0.0
    return P,R,(2*P*R/(P+R) if P+R else 0.0)
def micro(per):
    TP=sum(v[0] for v in per.values()); FP=sum(v[1] for v in per.values()); FN=sum(v[2] for v in per.values())
    return prf(TP,FP,FN)
def macro(per):
    vs=[prf(*v) for v in per.values() if (v[0]+v[2])>0]
    if not vs: return (0.0,0.0,0.0)
    n=len(vs)
    return (sum(x[0] for x in vs)/n, sum(x[1] for x in vs)/n, sum(x[2] for x in vs)/n)
def spr(gold,pred,ids,thr):
    s=0
    for i in ids:
        g=gold[i]["pts"]|gold[i]["pfs"]; p=pred[i]["pts"]|pred[i]["pfs"]
        if not p or len(p&g)/len(p)>=thr: s+=1
    return s/(len(ids) or 1)
def nm(f):
    f=f.lower()
    if "bert_batch2_all" in f: return "BERT(all-label)"
    if "bert_batch2" in f: return "BERT(25-label)"
    if "lora_gemma" in f: return "LoRA Gemma-4B"
    if "lora_qwen9b" in f: return "LoRA Qwen-9B"
    if "lora_qwen" in f: return "LoRA Qwen-4B"
    if "lora_llama" in f: return "LoRA Llama-8B"
    if "lora_phi" in f: return "LoRA Phi-3"
    if "gemma" in f: return "Gemma-4B(prompt)"
    if "qwen3.5-9b" in f: return "Qwen-9B(prompt)"
    if "qwen3.5-4b" in f: return "Qwen-4B(prompt)"
    if "llama" in f: return "Llama-8B(prompt)"
    if "phi" in f: return "Phi-3(prompt)"
    return Path(f).name[:16]
def main():
    PT,PF=evalset(); gold=load(SCORE_GOLD)
    cols=["PTmiP","PTmiR","PTmiF","PTmaP","PTmaR","PTmaF","PFmiP","PFmiR","PFmiF","PFmaP","PFmaR","PFmaF"]+[f"S{int(t*100)}" for t in THRS]
    print(f"{'model':<17}"+"".join(f"{c:>7}" for c in cols))
    for f in sys.argv[1:]:
        pred=load(f)
        ids=[i for i in gold if i in pred and gold[i]["source"]=="smishing"]
        pt=counts(gold,pred,"pts",ids,PT); pf=counts(gold,pred,"pfs",ids,PF)
        ami=micro(pt); ama=macro(pt); bmi=micro(pf); bma=macro(pf)
        vals=[ami[0],ami[1],ami[2],ama[0],ama[1],ama[2],bmi[0],bmi[1],bmi[2],bma[0],bma[1],bma[2]]+[spr(gold,pred,ids,t) for t in THRS]
        print(f"{nm(f):<17}"+"".join(f"{v:>7.2f}" for v in vals))
if __name__=="__main__":
    main()
