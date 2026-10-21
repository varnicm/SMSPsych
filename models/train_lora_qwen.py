
#!/usr/bin/env python

"""

LoRA fine-tuning of Qwen3-4B-Instruct for PT/PF extraction (generative).

Identical protocol to train_lora_gemma.py; only the base model differs (and the

Gemma #3130 monkey-patch is removed, since Qwen uses standard nn.Linear proj).

Run (GPU node, A100):

  conda activate slm && unset PYTHONPATH

  export HF_HOME=./hf_cache

  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  python train_lora_qwen.py

"""

import json, random, re

from pathlib import Path

TRAIN_FILE = "gold_final_batch2.jsonl"

TEST_FILE  = "gold_final.jsonl"

BASE_MODEL = "Qwen/Qwen3.5-4B"   # set to the id you downloaded

PT_TAX = "pts-receiver.txt"

PF_TAX = "pfs-sender.txt"

LORA_R = 4

LORA_ALPHA = 16

LORA_DROPOUT = 0.05

LR = 5e-5

EPOCHS = 5

BATCH = 2

GRAD_ACCUM = 8

MAX_LEN = 384

VAL_FRAC = 0.1

SEED = 42

PATIENCE = 2

OUT_PRED = "lora_qwen_pred.jsonl"

USE_CHECKPOINTING = True

MAX_GRAD_NORM = 1.0

random.seed(SEED)

def load_tax(path):

    out=[]

    for line in open(path,encoding="utf-8"):

        line=line.strip()

        if line and not line.startswith("#"):

            out.append(line.split(":",1)[0].strip())

    return out

def build_prompt(msg, pt_list, pf_list):

    return (

        "You extract persuasion signals from an SMS message.\n"

        "Return ONLY a JSON object with keys \"pts\" and \"pfs\", each a list of "

        "labels drawn from the allowed sets. Use a label only if the mechanism is "

        "genuinely present in the text.\n\n"

        f"Allowed PTs: {', '.join(pt_list)}\n"

        f"Allowed PFs: {', '.join(pf_list)}\n\n"

        f"Message: {msg}\n\n"

        "JSON:"

    )

def target_json(rec):

    return json.dumps({"pts":[x.lower() for x in rec.get("pts",[])],

                       "pfs":[x.lower() for x in rec.get("pfs",[])]},

                      ensure_ascii=False)

def parse_output(text, pt_set, pf_set):

    m=re.search(r"\{.*?\}", text, re.DOTALL)

    pts, pfs = [], []

    if m:

        try:

            obj=json.loads(m.group(0))

            pts=[x.lower() for x in obj.get("pts",[]) if x.lower() in pt_set]

            pfs=[x.lower() for x in obj.get("pfs",[]) if x.lower() in pf_set]

        except Exception:

            pass

    return sorted(set(pts)), sorted(set(pfs))

def main():

    import torch

    from transformers import AutoTokenizer, AutoModelForCausalLM

    from peft import LoraConfig, get_peft_model

    from torch.utils.data import Dataset, DataLoader

    device="cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {device}")

    if device=="cuda": print("GPU:", torch.cuda.get_device_name(0))

    pt_list=load_tax(PT_TAX); pf_list=load_tax(PF_TAX)

    pt_set={x.lower() for x in pt_list}; pf_set={x.lower() for x in pf_list}

    print(f"taxonomy: {len(pt_list)} PTs, {len(pf_list)} PFs")

    train_rows=[json.loads(l) for l in open(TRAIN_FILE,encoding="utf-8") if l.strip()]

    test_rows =[json.loads(l) for l in open(TEST_FILE, encoding="utf-8") if l.strip()]

    random.shuffle(train_rows)

    n_val=int(len(train_rows)*VAL_FRAC)

    val_rows=train_rows[:n_val]; tr_rows=train_rows[n_val:]

    print(f"train {len(tr_rows)}  val {len(val_rows)}  test {len(test_rows)}")

    tok=AutoTokenizer.from_pretrained(BASE_MODEL)

    if tok.pad_token is None: tok.pad_token=tok.eos_token

    def make_example(rec):

        prompt=build_prompt(rec["message"], pt_list, pf_list)

        answer=" "+target_json(rec)+tok.eos_token

        p_ids=tok(prompt, add_special_tokens=False)["input_ids"]

        a_ids=tok(answer, add_special_tokens=False)["input_ids"]

        ids=(p_ids+a_ids)[:MAX_LEN]

        labels=([-100]*len(p_ids)+a_ids)[:MAX_LEN]

        return ids, labels

    class DS(Dataset):

        def __init__(self, rows): self.rows=rows

        def __len__(self): return len(self.rows)

        def __getitem__(self, i):

            ids,labels=make_example(self.rows[i]); return ids, labels

    def collate(batch):

        maxlen=max(len(x[0]) for x in batch)

        input_ids=[]; attn=[]; labels=[]

        for ids,lab in batch:

            pad=maxlen-len(ids)

            input_ids.append(ids+[tok.pad_token_id]*pad)

            attn.append([1]*len(ids)+[0]*pad)

            labels.append(lab+[-100]*pad)

        return (torch.tensor(input_ids),torch.tensor(attn),torch.tensor(labels))

    model=AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)

    if USE_CHECKPOINTING:

        model.gradient_checkpointing_enable()

        model.enable_input_require_grads()

        model.config.use_cache = False

    else:

        model.config.use_cache = True

    lora=LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,

                    target_modules=["q_proj","k_proj","v_proj","o_proj"],

                    task_type="CAUSAL_LM")

    model=get_peft_model(model, lora)

    model.to(device)

    model.print_trainable_parameters()

    n_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"parameters requiring grad: {n_grad:,}")

    if n_grad == 0:

        raise RuntimeError("No parameters require grad -- LoRA not trainable.")

    tr_loader=DataLoader(DS(tr_rows),batch_size=BATCH,shuffle=True,collate_fn=collate)

    va_loader=DataLoader(DS(val_rows),batch_size=BATCH,shuffle=False,collate_fn=collate)

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],

                          lr=LR, weight_decay=0.01)

    def val_loss():

        model.eval(); tot=0; n=0

        with torch.no_grad():

            for ids,attn,lab in va_loader:

                out=model(input_ids=ids.to(device),attention_mask=attn.to(device),

                          labels=lab.to(device))

                tot+=out.loss.item(); n+=1

        return tot/max(1,n)

    best=1e9; best_state=None; bad=0

    for ep in range(EPOCHS):

        model.train(); opt.zero_grad()

        for i,(ids,attn,lab) in enumerate(tr_loader):

            out=model(input_ids=ids.to(device),attention_mask=attn.to(device),

                      labels=lab.to(device))

            loss=out.loss

            if torch.isnan(loss) or torch.isinf(loss):

                opt.zero_grad(); continue

            (loss/GRAD_ACCUM).backward()

            if (i+1)%GRAD_ACCUM==0:

                torch.nn.utils.clip_grad_norm_(

                    [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)

                opt.step(); opt.zero_grad()

        vl=val_loss()

        print(f"epoch {ep+1}: val_loss={vl:.4f}")

        if vl<best:

            best=vl; bad=0

            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if "lora" in k}

        else:

            bad+=1

            if bad>=PATIENCE: print("early stop"); break

    if best_state:

        sd=model.state_dict()

        for k,v in best_state.items(): sd[k]=v.to(device)

        model.load_state_dict(sd)

    print("\ngenerating test predictions...")

    model.config.use_cache = True

    model.eval()

    with open(OUT_PRED,"w",encoding="utf-8") as f:

        for j,rec in enumerate(test_rows):

            prompt=build_prompt(rec["message"], pt_list, pf_list)

            ids=tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)

            with torch.no_grad():

                gen=model.generate(**ids, max_new_tokens=120, do_sample=False,

                                   pad_token_id=tok.pad_token_id)

            text=tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

            pts,pfs=parse_output(text, pt_set, pf_set)

            f.write(json.dumps({"id":rec.get("id",f"m{j}"),"message":rec["message"],

                                "source":rec.get("source","unknown"),

                                "pts":pts,"pfs":pfs},ensure_ascii=False)+"\n")

            if (j+1)%50==0: print(f"  {j+1}/{len(test_rows)}")

    print(f"\nWrote {OUT_PRED}")

    print(f"Score with:  python3 score_prf_v2.py {OUT_PRED}")

if __name__=="__main__":

    main()

