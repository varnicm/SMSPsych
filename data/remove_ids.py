#!/usr/bin/env python
"""
Remove a list of message IDs from all candidate files + answer key, so every
file stays aligned by id. Original ids are kept (gaps are fine).

Edit REMOVE_IDS below (or point REMOVE_FILE at a text file with one id per
line). Then run:

    python remove_ids.py

Writes cleaned copies with a _clean suffix; originals are left untouched.
"""

import json
import glob
from pathlib import Path

# ---- which ids to drop --------------------------------------------------
# Option A: list them here directly
REMOVE_IDS = [
     "s400-002",
     "s400-026",
     "s400-110",
     "s400-324",

]

# Option B: or put one id per line in a file and set its name here
REMOVE_FILE = "ids_to_remove.txt"   # ignored if it doesn't exist
# -------------------------------------------------------------------------

# files to clean: all candidate model files + the answer key
CANDIDATE_GLOB = "candidates/*.jsonl"
ANSWER_KEY     = "sample_400_answer_key.jsonl"
ALSO_CLEAN     = ["sample_400_to_annotate.jsonl"]   # add any others you use


def load_remove_set():
    ids = set(x.strip() for x in REMOVE_IDS if x.strip())
    p = Path(REMOVE_FILE)
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids


def clean_file(path, remove):
    kept, dropped = [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("id") in remove:
            dropped += 1
        else:
            kept.append(line)
    out = path.replace(".jsonl", "_clean.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    print(f"  {Path(path).name}: kept {len(kept)}, removed {dropped} -> {Path(out).name}")
    return dropped


def main():
    remove = load_remove_set()
    if not remove:
        print("No ids to remove. Edit REMOVE_IDS or create ids_to_remove.txt")
        return
    print(f"Removing {len(remove)} ids: {sorted(remove)[:10]}"
          f"{'...' if len(remove) > 10 else ''}\n")

    targets = sorted(glob.glob(CANDIDATE_GLOB))
    for extra in [ANSWER_KEY] + ALSO_CLEAN:
        if Path(extra).exists() and extra not in targets:
            targets.append(extra)

    print(f"Cleaning {len(targets)} files:")
    total = 0
    for t in targets:
        total += clean_file(t, remove)

    print(f"\nDone. Each file now has *_clean.jsonl with the ids removed.")
    print("Point your agreement script at the *_clean.jsonl files "
          "(or rename them over the originals once you've checked).")


if __name__ == "__main__":
    main()
