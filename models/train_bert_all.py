
#!/usr/bin/env python

"""

Single-split BERT, matching the LoRA protocol: train on gold_final_batch2.jsonl

(1714), test on gold_final.jsonl (396). Label set derived from TRAIN ONLY (strict

separation). Checkpoint + threshold fixed at 0.8 (reported); 0.5 is a sensitivity

file. Asserts no message overlaps train/test. Writes predictions for score_prf_v2.py.

Run:

  conda activate slm && unset PYTHONPATH

  export HF_HOME=./hf_cache

  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  python train_bert_single.py

"""

import json, random

from collections import Counter

import numpy as np

TRAIN_FILE="gold_final_batch2.jsonl"; TEST_FILE="gold_final.jsonl"

MIN_SUPPORT=1; BASE_MODEL="bert-base-uncased"

MAX_LEN=128; SEED=42; EPOCHS=20; LR=2e-5; BATCH=16

VAL_FRAC_OF_TRAIN=0.15; FIXED_THR=0.8; PATIENCE=4

random.seed(SEED); np.random.seed(SEED)

def micro_prf(Y,P):

    tp=((P==1)&(Y==1)).sum(); fp=((P==1)&(Y==0)).sum(); fn=((P==0)&(Y==1)).sum()

    pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0

    return pr,rc,(2*pr*rc/(pr+rc) if pr+rc else 0.0)

def load_rows(fn):

    return [json.loads(l) for l in open(fn,encoding="utf-8") if l.strip()]

def labels_of(r):

    return ([f"PT::{x.strip().lower()}" for x in r.get("pts",[])]+

            [f"PF::{x.strip().lower()}" for x in r.get("pfs",[])])

def main():

    import torch

    from torch.utils.data import Dataset, DataLoader

    from transformers import AutoTokenizer, AutoModel

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(SEED)

        torch.backends.cudnn.benchmark=False

        torch.backends.cudnn.deterministic=True

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp=device.type=="cuda"; print(f"device: {device}")

    train_rows=load_rows(TRAIN_FILE); test_rows=load_rows(TEST_FILE)

    # --- separation assertion ---

    tr_txt={r["message"].strip().lower() for r in train_rows}

    te_txt={r["message"].strip().lower() for r in test_rows}

    ov=tr_txt & te_txt

    assert not ov, f"ERROR: {len(ov)} messages appear in BOTH train and test"

    print(f"separation OK: no overlap ({len(tr_txt)} train / {len(te_txt)} test unique)")

    rows=train_rows+test_rows; n_train=len(train_rows)

    # --- label set from TRAIN ONLY ---

    sup=Counter()

    for r in train_rows:

        for l in set(labels_of(r)): sup[l]+=1

    LABELS=sorted([l for l,c in sup.items() if c>=MIN_SUPPORT])

    lab2i={l:i for i,l in enumerate(LABELS)}

    print(f"{len(LABELS)} labels with support >= {MIN_SUPPORT} (from train only):")

    for l in LABELS: print(f"   {l}  ({sup[l]})")

    targets=np.zeros((len(rows),len(LABELS)),dtype=np.float32)

    for ri,r in enumerate(rows):

        for l in labels_of(r):

            if l in lab2i: targets[ri,lab2i[l]]=1.0

    tok=AutoTokenizer.from_pretrained(BASE_MODEL)

    enc=tok([r["message"] for r in rows],truncation=True,max_length=MAX_LEN,

            padding="max_length",return_tensors="pt")

    input_ids=enc["input_ids"]; attn=enc["attention_mask"]; yt=torch.from_numpy(targets)

    class DS(Dataset):

        def __init__(self,ids): self.ids=list(ids)

        def __len__(self): return len(self.ids)

        def __getitem__(self,k):

            i=self.ids[k]; return input_ids[i],attn[i],yt[i]

    class Clf(torch.nn.Module):

        def __init__(self,n):

            super().__init__()

            self.enc=AutoModel.from_pretrained(BASE_MODEL)

            self.drop=torch.nn.Dropout(0.3)

            self.head=torch.nn.Linear(self.enc.config.hidden_size,n)

        def forward(self,ids,mask):

            out=self.enc(input_ids=ids,attention_mask=mask).last_hidden_state[:,0]

            return self.head(self.drop(out))

    def loader(ids,shuffle=False):

        return DataLoader(DS(ids),batch_size=BATCH,shuffle=shuffle,pin_memory=use_amp)

    def collect(model,dl):

        model.eval(); Ps=[];Ys=[]

        with torch.inference_mode():

            for ids,mask,y in dl:

                ids=ids.to(device);mask=mask.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):

                    logits=model(ids,mask)

                Ps.append(torch.sigmoid(logits).float().cpu().numpy()); Ys.append(y.numpy())

        return np.vstack(Ps),np.vstack(Ys)

    train_pool=list(range(n_train)); random.shuffle(train_pool)

    n_val=int(len(train_pool)*VAL_FRAC_OF_TRAIN)

    val_ids=train_pool[:n_val]; train_ids=train_pool[n_val:]

    test_ids=list(range(n_train,len(rows)))

    print(f"train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)}")

    # --- per-label coverage in train/val ---

    tr_c=targets[train_ids].sum(0); va_c=targets[val_ids].sum(0)

    print("per-label support (train / val):")

    for i,l in enumerate(LABELS):

        flag=" <-- LOW" if (tr_c[i]<5 or va_c[i]<1) else ""

        print(f"   {l:<28} {int(tr_c[i]):>4} / {int(va_c[i]):>3}{flag}")

    tr=loader(train_ids,shuffle=True); va=loader(val_ids); te=loader(test_ids)

    model=Clf(len(LABELS)).to(device)

    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)

    lossf=torch.nn.BCEWithLogitsLoss(); scaler=torch.cuda.amp.GradScaler(enabled=use_amp)

    best=-1; best_state=None; bad=0

    for ep in range(EPOCHS):

        model.train(); tot=0.0

        for ids,mask,y in tr:

            ids=ids.to(device);mask=mask.to(device);y=y.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):

                loss=lossf(model(ids,mask),y)

            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot+=loss.item()

        Pv,Yv=collect(model,va); _,_,vf1=micro_prf(Yv,(Pv>=FIXED_THR).astype(int))

        print(f"  epoch {ep+1:02d}: loss={tot/len(tr):.4f} val_microF1@{FIXED_THR}={vf1:.4f}")

        if vf1>best: best=vf1; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0

        else:

            bad+=1

            if bad>=PATIENCE: print(f"  early stop @ epoch {ep+1}"); break

    if best_state is not None: model.load_state_dict(best_state)

    Pt,Yt=collect(model,te)

    for tag,thr in [("thr08(primary)",FIXED_THR),("thr05(sensitivity)",0.5)]:

        p,r,f=micro_prf(Yt,(Pt>=thr).astype(int))

        print(f"  [{tag}] test micro P={p:.3f} R={r:.3f} F1={f:.3f}")

    def write_preds(pm,path):

        with open(path,"w",encoding="utf-8") as f:

            for j,r in enumerate(test_rows):

                on=[LABELS[k] for k in np.where(pm[j]==1)[0]]

                pts=[l.split("::",1)[1] for l in on if l.startswith("PT::")]

                pfs=[l.split("::",1)[1] for l in on if l.startswith("PF::")]

                f.write(json.dumps({"id":r.get("id",f"m{j}"),"message":r.get("message",""),

                        "source":r.get("source","unknown"),

                        "pts":sorted(pts),"pfs":sorted(pfs)},ensure_ascii=False)+"\n")

    write_preds((Pt>=FIXED_THR).astype(int),"bert_batch2_all_pred_thr08.jsonl")

    write_preds((Pt>=0.5).astype(int),"bert_batch2_all_pred_thr05.jsonl")

    print("\nWrote bert_batch2_all_pred_thr08.jsonl (primary) and bert_batch2_all_pred_thr05.jsonl (sensitivity)")

if __name__=="__main__":

    main()

