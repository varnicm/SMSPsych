#!/usr/bin/env python
"""
Per-label / micro / macro / SPR / example-based scoring over a FIXED evaluated
label set (support >= TAU). SPR + empty-rate use each model's FULL predicted set.
Example-based (per-message) P/R/F1 credit partial correctness per message.
Usage: python score_prf_v2.py bert_expanded_pred_thr08.jsonl
"""
import sys, json, glob
from pathlib import Path
from collections import defaultdict, Counter
GOLD_FILES = ["gold_final.jsonl", "gold_final_batch2.jsonl"]
TAU = 30
SCORE_GOLD = "gold_final.jsonl"
def load(path):
    d={}
    for line in open(path, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        d[r["id"]]={"pts":set(x.lower() for x in r.get("pts",[])),
                    "pfs":set(x.lower() for x in r.get("pfs",[])),
                    "source":r.get("source","unknown")}
    return d
def evaluated_labels():
    pt=Counter(); pf=Counter()
    for fn in GOLD_FILES:
        if not Path(fn).exists(): continue
        for line in open(fn, encoding="utf-8"):
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            for x in set(v.lower() for v in r.get("pts",[])): pt[x]+=1
            for x in set(v.lower() for v in r.get("pfs",[])): pf[x]+=1
    PT={l for l,c in pt.items() if c>=TAU}
    PF={l for l,c in pf.items() if c>=TAU}
    return PT, PF
def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    f=2*p*r/(p+r) if p+r else 0.0
    return p,r,f
def score(gold, pred, kind, ids, allowed):
    per=defaultdict(lambda:[0,0,0]); support=defaultdict(int)
    for i in ids:
        g=gold[i][kind] & allowed
        p=pred[i][kind] & allowed
        for lab in g: support[lab]+=1
        for lab in g & p: per[lab][0]+=1
        for lab in p - g: per[lab][1]+=1
        for lab in g - p: per[lab][2]+=1
    return per, support
def aggregate(per, allowed):
    TP=sum(v[0] for v in per.values())
    FP=sum(v[1] for v in per.values())
    FN=sum(v[2] for v in per.values())
    micro=prf(TP,FP,FN)
    fs=[prf(*per[l])[2] for l in per if (per[l][0]+per[l][2])>0]
    macro=sum(fs)/len(fs) if fs else 0.0
    return micro, macro
# def spr(gold, pred, ids):
#     safe=0; empty=0
#     for i in ids:
#         g=gold[i]["pts"] | gold[i]["pfs"]
#         p=pred[i]["pts"] | pred[i]["pfs"]
#         if not p: empty+=1
#         if p.issubset(g): safe+=1
#     n=len(ids) or 1
#     return safe/n, empty/n

def spr(gold, pred, ids, thr=1.0):
    """Safe-prediction rate at a per-message precision threshold, over the FULL
    predicted set. thr=1.0 = strict (all predicted labels supported); thr=0.8
    counts a message safe if >=80% of its predicted labels are supported."""
    safe=0; empty=0
    for i in ids:
        g=gold[i]["pts"] | gold[i]["pfs"]
        p=pred[i]["pts"] | pred[i]["pfs"]
        if not p:
            empty+=1; safe+=1; continue
        if len(p & g)/len(p) >= thr: safe+=1
    n=len(ids) or 1
    return safe/n, empty/n
    
    
def example_based(gold, pred, kind, ids, allowed, thr=0.8):
    """Per-message P/R/F1 for one label kind, over messages with >=1 reference or
    predicted label of that kind. frac = share of messages with >=1 reference
    label whose per-message F1 >= thr."""
    Ps=[]; Rs=[]; Fs=[]; above=0; denom=0
    for i in ids:
        g=gold[i][kind] & allowed
        p=pred[i][kind] & allowed
        if not g and not p: continue
        inter=len(g & p)
        Ps.append(inter/len(p) if p else 0.0)
        Rs.append(inter/len(g) if g else 0.0)
        Fs.append(2*inter/(len(g)+len(p)) if (len(g)+len(p)) else 0.0)
        if g:
            denom+=1
            if Fs[-1]>=thr: above+=1
    n=len(Fs) or 1
    return sum(Ps)/n, sum(Rs)/n, sum(Fs)/n, (above/denom if denom else 0.0), denom
def report_block(title, gold, pred, kind, ids, allowed):
    per, support = score(gold, pred, kind, ids, allowed)
    micro, macro = aggregate(per, allowed)
    print(f"  -- {title} --")
    rows=[(lab, support[lab], *prf(*per[lab])) for lab in allowed if support[lab]>0]
    rows.sort(key=lambda x:-x[1])
    if rows:
        print(f"     {'label':<24}{'sup':>4} {'P':>6} {'R':>6} {'F1':>6}")
        for lab,sup,p,r,f in rows:
            print(f"     {lab:<24}{sup:>4} {p:>6.2f} {r:>6.2f} {f:>6.2f}")
    print(f"     micro-F1={micro[2]:.3f}  macro-F1={macro:.3f}  "
          f"(micro P={micro[0]:.2f} R={micro[1]:.2f})")
    print()
def report_example(gold, pred, ids, PT, PF):
    for kind, allowed, name in [("pts",PT,"PT"),("pfs",PF,"PF")]:
        mp,mr,mf,frac,denom=example_based(gold,pred,kind,ids,allowed)
        print(f"  example-based {name}: P={mp:.3f} R={mr:.3f} F1={mf:.3f}"
              f"   (msgs with >=1 {name}={denom}, frac F1>=0.8: {frac:.3f})")
    print()
def benign_rates(gold, pred, ids, allowed):
    silent=0; ref_attr=0
    for i in ids:
        p=(pred[i]["pts"] | pred[i]["pfs"]) & allowed
        g=(gold[i]["pts"] | gold[i]["pfs"]) & allowed
        if not p: silent+=1
        if p - g: ref_attr+=1
    n=len(ids) or 1
    return silent/n, ref_attr/n
def main():
    PT_LABELS, PF_LABELS = evaluated_labels()
    ALL = PT_LABELS | PF_LABELS
    print(f"Evaluated label set (support >= {TAU}): "
          f"{len(PT_LABELS)} PT, {len(PF_LABELS)} PF, {len(ALL)} total\n")
    files=sys.argv[1:]
    if not files:
        print("Usage: python score_prf_v2.py <pred.jsonl> [...]"); return
    gold=load(SCORE_GOLD)
    for f in files:
        pred=load(f)
        ids=[i for i in gold if i in pred]
        sm=[i for i in ids if gold[i]["source"]=="smishing"]
        bn=[i for i in ids if gold[i]["source"]=="benign"]
        print("="*64); print(Path(f).name, f"({len(ids)} messages)"); print("="*64)
        print("ALL MESSAGES")
        report_block("PT", gold, pred, "pts", ids, PT_LABELS)
        report_block("PF", gold, pred, "pfs", ids, PF_LABELS)
        # s,e=spr(gold,pred,ids)
        # print(f"  SPR={s:.3f}  empty-prediction-rate={e:.3f}")
        s,e=spr(gold,pred,ids)
        s8,_=spr(gold,pred,ids,0.7)
        print(f"  SPR(strict)={s:.3f}  SPR@0.7={s8:.3f}  empty-prediction-rate={e:.3f}")
        report_example(gold,pred,ids,PT_LABELS,PF_LABELS)
        print("SMISHING ONLY")
        report_block("PT", gold, pred, "pts", sm, PT_LABELS)
        report_block("PF", gold, pred, "pfs", sm, PF_LABELS)
        # s,e=spr(gold,pred,sm)
        # print(f"  SPR={s:.3f}  empty-prediction-rate={e:.3f}")
        s,e=spr(gold,pred,sm)
        s8,_=spr(gold,pred,sm,0.7)
        print(f"  SPR(strict)={s:.3f}  SPR@0.7={s8:.3f}  empty-prediction-rate={e:.3f}")
        report_example(gold,pred,sm,PT_LABELS,PF_LABELS)
        print("BENIGN ONLY")
        sil,ra=benign_rates(gold,pred,bn,ALL)
        print(f"  silence-rate={sil:.3f}  reference-attribution-rate={ra:.3f}  "
              f"(n={len(bn)})\n")
if __name__=="__main__":
    main()
