import sys, json
from collections import Counter
from pathlib import Path
GOLD_FILES=["gold_final.jsonl","gold_final_batch2.jsonl"]; TAU=30
SCORE_GOLD="gold_final.jsonl"
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
def f1(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp else 0.0; R=tp/(tp+fn) if tp+fn else 0.0
    return 2*P*R/(P+R) if P+R else 0.0
SHORT={"bert_batch2_all":"BERT","lora_gemma":"Gem$_L$","lora_qwen9b":"Qw9$_L$",
"lora_qwen":"Qw4$_L$","lora_llama":"Lla$_L$","lora_phi":"Phi$_L$",
"qwen3.5-9b":"Qw9$_P$","qwen3.5-4b":"Qw4$_P$","gemma":"Gem$_P$","llama":"Lla$_P$","phi":"Phi$_P$"}
def nm(f):
    f=f.lower()
    for k,v in SHORT.items():
        if k in f: return v
    return Path(f).name[:6]
def main():
    PTe,PFe=evalset()
    gold=load(SCORE_GOLD)
    ids=[i for i in gold if gold[i]["source"]=="smishing"]
    preds=[(nm(f),load(f)) for f in sys.argv[1:]]
    n=len(preds)
    def sup(lab,kind): return sum(1 for i in ids if lab in gold[i][kind])
    PT=sorted(PTe,key=lambda l:-sup(l,"pts")); PF=sorted(PFe,key=lambda l:-sup(l,"pfs"))
    def F(pred,lab,kind):
        tp=fp=fn=0
        for i in ids:
            if i not in pred: continue
            g=lab in gold[i][kind]; p=lab in pred[i][kind]
            if g and p: tp+=1
            elif p: fp+=1
            elif g: fn+=1
        return f1(tp,fp,fn)
    print(r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}")
    print(r"\caption{Per-label $F_1$ on smishing messages for all models, over the 25 evaluated labels (support $\ge 30$). $L$=LoRA, $P$=prompted. Best per label in \textbf{bold}.}")
    print(r"\label{tab:per_label_f1}")
    print(r"\begin{tabular}{lr"+"c"*n+"}")
    print(r"\toprule")
    print("Label & Sup & "+" & ".join(nm for nm,_ in preds)+r" \\")
    print(r"\midrule")
    for kind,labs,title in [("pts",PT,"Techniques (PT)"),("pfs",PF,"Factors (PF)")]:
        print(r"\multicolumn{%d}{l}{\textit{%s}} \\"%(2+n,title))
        for lab in labs:
            vals=[F(pred,lab,kind) for _,pred in preds]; mx=max(vals)
            cells=[(r"\textbf{%.2f}"%v if v>=mx-1e-9 and mx>0 else "%.2f"%v) for v in vals]
            print(f"{lab} & {sup(lab,kind)} & "+" & ".join(cells)+r" \\")
        print(r"\midrule")
    print(r"\bottomrule\end{tabular}\end{table*}")
if __name__=="__main__": main()
