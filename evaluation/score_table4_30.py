import json
from collections import Counter
GOLD=["gold_final.jsonl","gold_final_batch2.jsonl"]; TAU=30
def load(files):
    d={}
    for fn in files:
        for l in open(fn,encoding="utf-8"):
            l=l.strip()
            if not l: continue
            r=json.loads(l)
            d[r["id"]]={"pts":set(x.strip().lower() for x in r.get("pts",[])),
                        "pfs":set(x.strip().lower() for x in r.get("pfs",[]))}
    return d
def prf(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp else 0.0; R=tp/(tp+fn) if tp+fn else 0.0
    return P,R,(2*P*R/(P+R) if P+R else 0.0)
gold=load(GOLD); pred=load(["bert_expanded_all_pred_thr08.jsonl"])
ids=[i for i in gold if i in pred]
sup=Counter(); ptset=set(); pfset=set()
for i in gold:
    for x in gold[i]["pts"]: sup[x]+=1; ptset.add(x)
    for x in gold[i]["pfs"]: sup[x]+=1; pfset.add(x)
PT=sorted([l for l in ptset if sup[l]>=TAU],key=lambda l:-sup[l])
PF=sorted([l for l in pfset if sup[l]>=TAU],key=lambda l:-sup[l])
def pl(lab,kind):
    tp=fp=fn=0
    for i in ids:
        g=lab in gold[i][kind]; p=lab in pred[i][kind]
        if g and p: tp+=1
        elif p: fp+=1
        elif g: fn+=1
    return tp,fp,fn
TP=FP=FN=0; ps=[];rs=[];fs=[]
print("== per-label (>=30) ==")
for labs,kind in [(PT,"pts"),(PF,"pfs")]:
    for lab in labs:
        tp,fp,fn=pl(lab,kind); P,R,F=prf(tp,fp,fn)
        print(f"{lab} & {sup[lab]} & {P:.2f} & {R:.2f} & {F:.2f} \\\\")
        TP+=tp;FP+=fp;FN+=fn
        if tp+fn>0: ps.append(P);rs.append(R);fs.append(F)
n=len(fs)
mi=prf(TP,FP,FN)
print("Micro  %.3f %.3f %.3f"%mi)
print("Macro  %.3f %.3f %.3f"%(sum(ps)/n,sum(rs)/n,sum(fs)/n))
s=0
for i in ids:
    g=gold[i]["pts"]|gold[i]["pfs"]; p=pred[i]["pts"]|pred[i]["pfs"]
    if not p or p<=g: s+=1
print("SPR    %.3f"%(s/len(ids)))
