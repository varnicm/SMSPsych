#!/usr/bin/env python
"""
Detect non-English messages in a JSONL file (one record per line with a
"message" field).

Setup:
    pip install langdetect
Run:
    python check_language.py sample_400_to_annotate.jsonl
    (defaults to sample_400_to_annotate.jsonl if no argument given)

NOTE: SMS text is short and full of abbreviations, so detection is noisy.
Treat results as a flag for review, not ground truth. Very short messages
(< MIN_CHARS) are reported separately as "too short to judge".
"""

import sys
import json
from langdetect import detect_langs, DetectorFactory

DetectorFactory.seed = 0          # deterministic
MIN_CHARS = 15                    # below this, detection is unreliable
CONF_THRESHOLD = 0.70             # min probability to trust the top language

INPUT = sys.argv[1] if len(sys.argv) > 1 else "sample_400_to_annotate.jsonl"


def main():
    rows = []
    for line in open(INPUT, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} messages from {INPUT}\n")

    english, non_english, too_short, uncertain = [], [], [], []

    for r in rows:
        msg = (r.get("message") or "").strip()
        if len(msg) < MIN_CHARS:
            too_short.append(r)
            continue
        try:
            langs = detect_langs(msg)        # sorted, most likely first
            top = langs[0]
            if top.lang == "en" and top.prob >= CONF_THRESHOLD:
                english.append(r)
            elif top.lang != "en" and top.prob >= CONF_THRESHOLD:
                non_english.append((r, top.lang, top.prob))
            else:
                uncertain.append((r, top.lang, top.prob))
        except Exception:
            uncertain.append((r, "?", 0.0))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  English (>= {CONF_THRESHOLD:.0%} conf): {len(english)}")
    print(f"  Non-English (>= {CONF_THRESHOLD:.0%} conf): {len(non_english)}")
    print(f"  Uncertain (low conf): {len(uncertain)}")
    print(f"  Too short to judge (< {MIN_CHARS} chars): {len(too_short)}")

    if non_english:
        print("\n" + "=" * 60)
        print("LIKELY NON-ENGLISH")
        print("=" * 60)
        # group by detected language
        from collections import Counter
        by_lang = Counter(lang for _, lang, _ in non_english)
        print("By language:", dict(by_lang.most_common()))
        print()
        for r, lang, prob in non_english:
            print(f"[{r.get('id','?')}] {lang} ({prob:.2f})  {r['message'][:90]}")

    if uncertain:
        print("\n" + "=" * 60)
        print("UNCERTAIN (review manually)")
        print("=" * 60)
        for r, lang, prob in uncertain[:30]:
            print(f"[{r.get('id','?')}] guess={lang} ({prob:.2f})  {r['message'][:80]}")
        if len(uncertain) > 30:
            print(f"  ... and {len(uncertain)-30} more")


if __name__ == "__main__":
    main()
