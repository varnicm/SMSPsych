import json
from collections import defaultdict
pt=defaultdict(lambda:{'smishing':0,'benign':0})
pf=defaultdict(lambda:{'smishing':0,'benign':0})
nS=nB=0
for fn in ['gold_final.jsonl','gold_final_batch2.jsonl']:
    for l in open(fn):
        r=json.loads(l); s=r.get('source','unknown')
        if s=='smishing': nS+=1
        elif s=='benign': nB+=1
        for x in set(v.lower() for v in r.get('pts',[])): pt[x][s]+=1
        for x in set(v.lower() for v in r.get('pfs',[])): pf[x][s]+=1
print(f'# {nS} smishing, {nB} benign')
print('## PT'); 
for k in sorted(pt,key=lambda k:-(pt[k]['smishing']+pt[k]['benign'])):
    print(f\"{k}|{pt[k]['smishing']}|{pt[k]['benign']}\")
print('## PF')
for k in sorted(pf,key=lambda k:-(pf[k]['smishing']+pf[k]['benign'])):
    print(f\"{k}|{pf[k]['smishing']}|{pf[k]['benign']}\")