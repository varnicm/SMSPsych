import sys, json
from pathlib import Path
GOLD="gold_final.jsonl"; THRS=[0.5,0.6,0.7,0.8,0.9,1.0]
def load(p):
    d={}
    for l in open(p,encoding="utf-8"):
        l=l.strip()
        if not l: continue
        r=json.loads(l)
        d[r["id"]]={"lab":set(x.strip().lower() for x in r.get("pts",[]))|set(x.strip().lower() for x in r.get("pfs",[])),
                    "source":r.get("source","unknown")}
    return d
def spr(gold,pred,ids,thr):
    s=0
    for i in ids:
        g=gold[i]["lab"]; p=pred[i]["lab"]
        if not p or len(p&g)/len(p)>=thr: s+=1
    return s/(len(ids) or 1)
def nm(f):
    f=f.lower()
    if "bert_batch2_all" in f: return "BERT(all-label)"
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
    gold=load(GOLD)
    cols=[f"SM{int(t*100)}" for t in THRS]+[f"BN{int(t*100)}" for t in THRS]
    print(f"{'model':<17}"+"".join(f"{c:>7}" for c in cols))
    for f in sys.argv[1:]:
        pred=load(f)
        sm=[i for i in gold if i in pred and gold[i]["source"]=="smishing"]
        bn=[i for i in gold if i in pred and gold[i]["source"]=="benign"]
        vals=[spr(gold,pred,sm,t) for t in THRS]+[spr(gold,pred,bn,t) for t in THRS]
        print(f"{nm(f):<17}"+"".join(f"{v:>7.2f}" for v in vals))
if __name__=="__main__":
    main()
