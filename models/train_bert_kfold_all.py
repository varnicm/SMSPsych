#!/usr/bin/env python
"""
K-fold BERT multi-label PT/PF classification on the EXPANDED dataset (2110 msgs).

Pools gold_final.jsonl (396) + gold_final_batch2.jsonl (1714) = 2110 messages,
same architecture / protocol as train_bert_kfold.py so results are directly
comparable to the 396-only run. Only change: more data + MIN_SUPPORT raised.

Writes held-out predictions for scoring with score_prf.py.

Run:
  conda activate slm && unset PYTHONPATH
  export HF_HOME=./hf_cache
  python train_bert_kfold_expanded.py
"""

import json, random, numpy as np
from pathlib import Path
from collections import Counter

GOLD_FILES = ["gold_final.jsonl", "gold_final_batch2.jsonl"]   # pooled
MIN_SUPPORT = 1                  # all-label head
BASE_MODEL = "bert-base-uncased"
MAX_LEN = 128
SEED = 42
EPOCHS = 20
LR = 2e-5
BATCH = 16
K = 5
VAL_FRAC_OF_TRAIN = 0.15
FIXED_THR = 0.8

random.seed(SEED); np.random.seed(SEED)


def micro_prf(Y, P):
    tp=((P==1)&(Y==1)).sum(); fp=((P==1)&(Y==0)).sum(); fn=((P==0)&(Y==1)).sum()
    pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
    return pr,rc,(2*pr*rc/(pr+rc) if pr+rc else 0.0)

def per_label(Y,P):
    tp=((P==1)&(Y==1)).sum(0); fp=((P==1)&(Y==0)).sum(0); fn=((P==0)&(Y==1)).sum(0)
    sup=Y.sum(0)
    pr=np.divide(tp,tp+fp,out=np.zeros(len(tp),float),where=(tp+fp)>0)
    rc=np.divide(tp,tp+fn,out=np.zeros(len(tp),float),where=(tp+fn)>0)
    f1=np.divide(2*pr*rc,pr+rc,out=np.zeros(len(pr)),where=(pr+rc)>0)
    return pr,rc,f1,sup

def macro_f1(Y,P):
    _,_,f1,sup=per_label(Y,P); m=sup>0
    return f1[m].mean() if m.any() else 0.0

def safe_rate(Y,P):
    s=0
    for i in range(len(Y)):
        ps=set(np.where(P[i]==1)[0]); gs=set(np.where(Y[i]==1)[0])
        if ps and ps.issubset(gs): s+=1
    return s/len(Y)


def main():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(SEED)
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    rows=[]
    for fn in GOLD_FILES:
        if not Path(fn).exists():
            print(f"WARNING: {fn} not found"); continue
        cnt=0
        for l in open(fn,encoding="utf-8"):
            l=l.strip()
            if l: rows.append(json.loads(l)); cnt+=1
        print(f"  loaded {cnt} from {fn}")
    print(f"total: {len(rows)} messages\n")

    def labels_of(r):
        return ([f"PT::{x.lower()}" for x in r.get("pts",[])]+
                [f"PF::{x.lower()}" for x in r.get("pfs",[])])
    sup=Counter()
    for r in rows:
        for l in set(labels_of(r)): sup[l]+=1
    LABELS=sorted([l for l,c in sup.items() if c>=MIN_SUPPORT])
    lab2i={l:i for i,l in enumerate(LABELS)}
    print(f"{len(LABELS)} labels with support >= {MIN_SUPPORT}:")
    for l in LABELS: print(f"   {l}  ({sup[l]})")
    print()

    def vec(r):
        y=np.zeros(len(LABELS),dtype=np.float32)
        for l in labels_of(r):
            if l in lab2i: y[lab2i[l]]=1.0
        return y

    tok=AutoTokenizer.from_pretrained(BASE_MODEL)

    class DS(Dataset):
        def __init__(self,ids): self.ids=list(ids)
        def __len__(self): return len(self.ids)
        def __getitem__(self,k):
            r=rows[self.ids[k]]
            enc=tok(r["message"],truncation=True,max_length=MAX_LEN,
                    padding="max_length",return_tensors="pt")
            return enc["input_ids"][0],enc["attention_mask"][0],torch.tensor(vec(r))

    class Clf(torch.nn.Module):
        def __init__(self,n):
            super().__init__()
            self.enc=AutoModel.from_pretrained(BASE_MODEL)
            self.drop=torch.nn.Dropout(0.3)
            self.head=torch.nn.Linear(self.enc.config.hidden_size,n)
        def forward(self,ids,mask):
            out=self.enc(input_ids=ids,attention_mask=mask).last_hidden_state[:,0]
            return self.head(self.drop(out))

    def collect(model,loader):
        model.eval(); Ps=[];Ys=[]
        with torch.no_grad():
            for ids,mask,y in loader:
                Ps.append(torch.sigmoid(model(ids.to(device),mask.to(device))).cpu().numpy())
                Ys.append(y.numpy())
        return np.vstack(Ps),np.vstack(Ys)

    idx=list(range(len(rows))); random.shuffle(idx)
    folds=[idx[i::K] for i in range(K)]
    pooled_P=np.zeros((len(rows),len(LABELS)))
    pooled_Y=np.zeros((len(rows),len(LABELS)))
    pooled_thr=np.zeros(len(rows))
    fold_metrics=[]

    for k in range(K):
        test_ids=folds[k]
        train_pool=[i for i in idx if i not in set(test_ids)]
        random.shuffle(train_pool)
        n_val=int(len(train_pool)*VAL_FRAC_OF_TRAIN)
        val_ids=train_pool[:n_val]; train_ids=train_pool[n_val:]
        print(f"--- fold {k+1}/{K}: train {len(train_ids)} val {len(val_ids)} test {len(test_ids)} ---")

        tr=DataLoader(DS(train_ids),batch_size=BATCH,shuffle=True)
        va=DataLoader(DS(val_ids),batch_size=BATCH)
        teDL=DataLoader(DS(test_ids),batch_size=BATCH)

        model=Clf(len(LABELS)).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
        lossf=torch.nn.BCEWithLogitsLoss()

        best=-1; best_state=None; bad=0
        for ep in range(EPOCHS):
            model.train()
            for ids,mask,y in tr:
                opt.zero_grad()
                lossf(model(ids.to(device),mask.to(device)),y.to(device)).backward()
                opt.step()
            Pv,Yv=collect(model,va)
            _,_,vf1=micro_prf(Yv,(Pv>=0.5).astype(int))
            if vf1>best: best=vf1; best_state={kk:v.cpu().clone() for kk,v in model.state_dict().items()}; bad=0
            else:
                bad+=1
                if bad>=4: break
        model.load_state_dict(best_state)

        Pv,Yv=collect(model,va)
        best_thr=0.5; best_score=-1
        for thr in [0.4,0.5,0.6,0.7,0.8]:
            p,r,f=micro_prf(Yv,(Pv>=thr).astype(int))
            score=2*p+f
            if score>best_score: best_score=score; best_thr=thr

        Pt,Yt=collect(model,teDL)
        for j,mid in enumerate(test_ids):
            pooled_P[mid]=Pt[j]; pooled_Y[mid]=Yt[j]; pooled_thr[mid]=best_thr
        pr,rc,f=micro_prf(Yt,(Pt>=best_thr).astype(int))
        fold_metrics.append((best_thr,pr,rc,f,safe_rate(Yt,(Pt>=best_thr).astype(int))))
        print(f"    thr={best_thr}  test micro P={pr:.3f} R={rc:.3f} F1={f:.3f}")

    print("\n"+"="*56)
    print("EXPANDED K-FOLD SUMMARY (precision-favoring threshold)")
    print("="*56)
    arr=np.array([m[1:] for m in fold_metrics])
    print(f"mean micro  P={arr[:,0].mean():.3f}  R={arr[:,1].mean():.3f}  "
          f"F1={arr[:,2].mean():.3f}  safe={arr[:,3].mean():.3f}")
    pooled_pred=(pooled_P>=pooled_thr[:,None]).astype(int)
    p,r,f=micro_prf(pooled_Y,pooled_pred)
    print(f"pooled micro  P={p:.3f} R={r:.3f} F1={f:.3f}  "
          f"macro-F1={macro_f1(pooled_Y,pooled_pred):.3f}  safe={safe_rate(pooled_Y,pooled_pred):.3f}")

    print("\npooled per-label (all folds, held-out):")
    pr,rc,f1,su=per_label(pooled_Y,pooled_pred)
    for i in np.argsort(-su):
        if su[i]>0:
            print(f"  {LABELS[i]:<28}{int(su[i]):>5}{pr[i]:>6.2f}{rc[i]:>6.2f}{f1[i]:>6.2f}")
    p5,r5,f5=micro_prf(pooled_Y,(pooled_P>=0.5).astype(int))
    print(f"\n[reference] pooled @0.5: P={p5:.3f} R={r5:.3f} F1={f5:.3f} "
          f"safe={safe_rate(pooled_Y,(pooled_P>=0.5).astype(int)):.3f}")

    def write_preds(pm,path):
        with open(path,"w",encoding="utf-8") as f:
            for mi,r in enumerate(rows):
                on=[LABELS[j] for j in np.where(pm[mi]==1)[0]]
                pts=[l.split("::",1)[1] for l in on if l.startswith("PT::")]
                pfs=[l.split("::",1)[1] for l in on if l.startswith("PF::")]
                f.write(json.dumps({"id":r.get("id",f"m{mi}"),"message":r.get("message",""),
                                    "source":r.get("source","unknown"),
                                    "pts":sorted(pts),"pfs":sorted(pfs)},ensure_ascii=False)+"\n")
    write_preds((pooled_P>=FIXED_THR).astype(int),"bert_expanded_all_pred_thr08.jsonl")
    write_preds((pooled_P>=0.5).astype(int),"bert_expanded_all_pred_thr05.jsonl")
    print("\nWrote bert_expanded_all_pred_thr08.jsonl and bert_expanded_all_pred_thr05.jsonl")


if __name__=="__main__":
    main()
