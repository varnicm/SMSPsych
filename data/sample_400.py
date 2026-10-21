#!/usr/bin/env python
"""
Sample a fresh, blinded 400-message set (200 smishing + 200 benign) using
stratified random sampling.

Sampling method (for the paper):
  Stratified random sampling. Messages are partitioned into strata defined by
  the crossing of message length (short / medium / long, by char-count
  terciles) and link presence (URL present / absent). Within each stratum,
  messages are drawn uniformly at random without replacement, allocated
  proportionally to stratum size, with a fixed random seed for reproducibility.
  Sampling is balanced across the two classes (smishing, benign) and is
  independent of any model output.

Outputs:
  sample_400_to_annotate.jsonl   blinded: id + message only
  sample_400_answer_key.jsonl    private: id -> source

Run:  python sample_400.py
"""

import json
import re
import random
import pandas as pd

# ============================ CONFIG ============================
SMISH_CSV       = "analysisdataset-2.csv"
SMISH_COL       = "MainText"
SMISH_ENCODING  = "latin-1"

BENIGN_CSV      = "smish-mandely.csv"
BENIGN_COL      = "TEXT"
BENIGN_LABEL    = "LABEL"
BENIGN_VAL      = "ham"
BENIGN_ENCODING = "utf-8-sig"

N_SMISHING = 200
N_BENIGN   = 200
MIN_CHARS  = 10
SEED       = 2024

OUT_ANNOTATE = "sample_400_to_annotate.jsonl"
OUT_KEY      = "sample_400_answer_key.jsonl"
# ================================================================


def load_pool(path, col, enc, label_col=None, label_val=None):
    df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
    df = df[df[col].notna()].copy()
    if label_col is not None:
        df = df[df[label_col].astype(str).str.lower() == str(label_val).lower()]
    df["_text"] = df[col].astype(str).str.strip()
    df = df[df["_text"].str.len() >= MIN_CHARS].drop_duplicates(subset="_text")
    return df


def has_link(t):
    return bool(re.search(r"https?://|www\.|\b\w+\.(com|info|net|us|cc|online|xyz|link|io)\b",
                          t, re.I))


def add_strata(df, length_edges):
    df = df.copy()
    df["_len"] = df["_text"].str.len()
    def lbin(n):
        if n <= length_edges[0]: return "short"
        if n <= length_edges[1]: return "medium"
        return "long"
    df["_lbin"] = df["_len"].apply(lbin)
    df["_link"] = df["_text"].apply(lambda t: "link" if has_link(t) else "nolink")
    df["_stratum"] = df["_lbin"] + "|" + df["_link"]
    return df


def stratified_sample(df, n, rng):
    if len(df) < n:
        raise ValueError(f"Pool has only {len(df)} messages but {n} requested. "
                         f"Reduce N or use a larger corpus.")
    strata = list(df.groupby("_stratum"))
    total = len(df)
    # proportional allocation
    alloc = []
    for key, g in strata:
        take = round(n * len(g) / total)
        take = min(take, len(g))
        alloc.append([key, g, take])
    # fix rounding to hit exactly n (bounded iterations)
    def total_alloc():
        return sum(a[2] for a in alloc)
    guard = 0
    while total_alloc() > n and guard < 100000:
        for a in sorted(alloc, key=lambda a: -a[2]):
            if total_alloc() <= n:
                break
            if a[2] > 0:
                a[2] -= 1
        guard += 1
    guard = 0
    while total_alloc() < n and guard < 100000:
        for a in sorted(alloc, key=lambda a: a[2]):
            if total_alloc() >= n:
                break
            if a[2] < len(a[1]):
                a[2] += 1
        guard += 1
    parts = []
    for key, g, take in alloc:
        if take > 0:
            parts.append(g.sample(n=take, random_state=rng.randint(0, 10**6)))
    return pd.concat(parts, ignore_index=True)


def main():
    rng = random.Random(SEED)

    smish = load_pool(SMISH_CSV, SMISH_COL, SMISH_ENCODING)
    benign = load_pool(BENIGN_CSV, BENIGN_COL, BENIGN_ENCODING,
                       BENIGN_LABEL, BENIGN_VAL)
    print(f"Pool: {len(smish)} smishing, {len(benign)} benign available")

    # length terciles computed on the smishing pool (the class of interest)
    q = smish["_text"].str.len().quantile([1/3, 2/3]).tolist()
    length_edges = (int(q[0]), int(q[1]))
    print(f"Length tercile edges (chars): <= {length_edges[0]} short, "
          f"<= {length_edges[1]} medium, else long")

    smish = add_strata(smish, length_edges)
    benign = add_strata(benign, length_edges)

    print(f"\nSampling {N_SMISHING} smishing (stratified) ...")
    s_sample = stratified_sample(smish, N_SMISHING, rng)
    print(f"  strata counts: {s_sample['_stratum'].value_counts().to_dict()}")

    print(f"Sampling {N_BENIGN} benign (stratified) ...")
    b_sample = stratified_sample(benign, N_BENIGN, rng)
    print(f"  strata counts: {b_sample['_stratum'].value_counts().to_dict()}")

    # build blinded records
    records = [{"_source": "smishing", "message": r["_text"]} for _, r in s_sample.iterrows()]
    records += [{"_source": "benign", "message": r["_text"]} for _, r in b_sample.iterrows()]
    rng.shuffle(records)

    ann, key = [], []
    for i, r in enumerate(records, 1):
        gid = f"s400-{i:03d}"
        ann.append({"id": gid, "message": r["message"], "pts": [], "pfs": []})
        key.append({"id": gid, "source": r["_source"]})

    with open(OUT_ANNOTATE, "w", encoding="utf-8") as f:
        for r in ann:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_KEY, "w", encoding="utf-8") as f:
        for r in key:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_s = sum(1 for r in key if r["source"] == "smishing")
    print(f"\nWrote {len(ann)} blinded messages to {OUT_ANNOTATE} "
          f"({n_s} smishing + {len(key)-n_s} benign)")
    print(f"Wrote private key to {OUT_KEY}")


if __name__ == "__main__":
    main()
