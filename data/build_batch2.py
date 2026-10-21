#!/usr/bin/env python
"""
Build the next annotation batch: all SmishTank smishing NOT already in the 396
gold set, plus an equal number of fresh benign messages (also not in the 396).

Exclusion here is EXACT-match only (near-duplicate handling to be done later).

Inputs:
  analysisdataset-2.csv   (smishing, col MainText, latin-1)
  smish-mandely.csv       (benign,   col TEXT, label ham, utf-8-sig)
  gold_final.jsonl        (the already-annotated 396, to exclude)

Output:
  batch2_to_annotate.jsonl   {id, message, source, pts:[], pfs:[]}
  batch2_answer_key.jsonl    {id, source}   (private reference)

Run:  python build_batch2.py
"""

import json, csv, random
from pathlib import Path

SMISH  = ("analysisdataset-2.csv", "MainText", "latin-1", None, None)
BENIGN = ("smish-mandely.csv", "TEXT", "utf-8-sig", "LABEL", "ham")
GOLD   = "gold_final.jsonl"
MIN_CHARS = 10
SEED = 202
ID_PREFIX = "batch2"

OUT_ANNOT = "batch2_to_annotate.jsonl"
OUT_KEY   = "batch2_answer_key.jsonl"


def load_col(path, col, enc, lcol, lval):
    out = []
    with open(path, encoding=enc, errors="replace", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if lcol and str(r.get(lcol, "")).strip().lower() != lval:
                continue
            t = (r.get(col) or "").strip()
            if len(t) >= MIN_CHARS:
                out.append(t)
    # dedup within source (exact)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq


def main():
    # already-annotated messages (exact text) to exclude
    gold_msgs = set()
    for line in open(GOLD, encoding="utf-8"):
        line = line.strip()
        if line:
            gold_msgs.add(json.loads(line)["message"].strip())
    print(f"Gold messages to exclude: {len(gold_msgs)}")

    smish = load_col(*SMISH)
    benign = load_col(*BENIGN)
    print(f"SmishTank smishing (unique): {len(smish)}")
    print(f"Benign pool (unique): {len(benign)}")

    # smishing = everything NOT in gold
    smish_new = [t for t in smish if t not in gold_msgs]
    print(f"Smishing not in gold: {len(smish_new)}")

    # benign = equal number, not in gold, randomly sampled
    rng = random.Random(SEED)
    benign_pool = [t for t in benign if t not in gold_msgs]
    n = min(len(smish_new), len(benign_pool))
    if len(benign_pool) < len(smish_new):
        print(f"WARNING: only {len(benign_pool)} benign available, "
              f"capping batch to {n} each")
    benign_new = rng.sample(benign_pool, len(smish_new)) \
        if len(benign_pool) >= len(smish_new) else benign_pool

    # build records
    records = [{"_src": "smishing", "message": t} for t in smish_new] + \
              [{"_src": "benign",   "message": t} for t in benign_new]
    rng.shuffle(records)

    with open(OUT_ANNOT, "w", encoding="utf-8") as fa, \
         open(OUT_KEY, "w", encoding="utf-8") as fk:
        for i, r in enumerate(records, 1):
            gid = f"{ID_PREFIX}-{i:04d}"
            fa.write(json.dumps({"id": gid, "message": r["message"],
                                 "source": r["_src"],
                                 "pts": [], "pfs": []},
                                ensure_ascii=False) + "\n")
            fk.write(json.dumps({"id": gid, "source": r["_src"]},
                                ensure_ascii=False) + "\n")

    n_s = sum(1 for r in records if r["_src"] == "smishing")
    n_b = sum(1 for r in records if r["_src"] == "benign")
    print(f"\nWrote {len(records)} messages to {OUT_ANNOT}")
    print(f"  {n_s} smishing + {n_b} benign")
    print(f"Wrote key to {OUT_KEY}")


if __name__ == "__main__":
    main()
