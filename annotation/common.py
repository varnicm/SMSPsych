#!/usr/bin/env python
"""
common.py - shared helpers for frontier-model candidate annotation.

All per-provider scripts import from here so the prompt, taxonomy, and
parsing are IDENTICAL across providers. Only the API call differs.

These outputs are CANDIDATE labels for building the human gold set,
not final ground truth.
"""

import json
import re
import time
from pathlib import Path

# ============================ CONFIG ============================
#INPUT_FILE  = "sample_400_to_annotate.jsonl"   # the 400 blinded messages
INPUT_FILE = "batch2_to_annotate.jsonl"   #rest of messages 1710

TAXONOMY_PT = "pts-receiver.txt"
TAXONOMY_PF = "pfs-sender.txt"
OUTPUT_DIR  = "candidates"                   # separate from SLM predictions
OUTPUT_DIR = "candidates_batch2"      # separate folder so it doesn't mix with the 400 candidates

MAX_RETRIES = 4
RETRY_BASE  = 2.0                            # seconds, exponential backoff
# ================================================================


def load_taxonomy(path):
    """Read 'label: definition' lines -> list of (label, definition)."""
    pairs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            lab, defn = line.split(":", 1)
            pairs.append((lab.strip().lower(), defn.strip()))
        else:
            pairs.append((line.strip().lower(), ""))
    return pairs


def load_data(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_system_prompt(pt_pairs, pf_pairs):
    """The shared sender/receiver annotation prompt.
    NOTE: for candidate generation we cast a WIDE net - include even
    lower-confidence labels so the human reviewer can prune, not miss."""
    def fmt(pairs):
        return "\n".join(f"- {lab}: {defn}" if defn else f"- {lab}"
                         for lab, defn in pairs)
    return (
        "You are an expert annotator identifying psychological techniques "
        "(PTs) and psychological factors (PFs) in SMS messages"
        "PTs are influence strategies the sender uses "
        "(how the message tries to influence the receiver). PFs are human "
        "attributes the message appeals to or targets in the receiver (what "
        "the message draws on). A single message may contain multiple PTs and "
        "multiple PFs. Label any PT or PF genuinely present, whether the "
        "message is malicious or benign. Include a label even if you are only "
        "moderately confident it applies, so that no plausible technique or "
        "factor is missed; it is better to surface a borderline label than to "
        "omit it. "
        "Respond with ONLY a JSON object of the form: "
        '{"pts": [...], "pfs": [...]}. '
        "Use names copied EXACTLY from the provided lists; do not invent "
        "labels. Use [] for either list if none apply. Add no explanation."
        f"\n\nPsychological Techniques (PTs):\n{fmt(pt_pairs)}"
        f"\n\nPsychological Factors (PFs):\n{fmt(pf_pairs)}"
        "\n\nOUTPUT RULES:\n"
        "1. Respond with ONLY a JSON object, starting with '{' and ending "
        "with '}'. No explanation or commentary.\n"
        "2. Exactly two keys: \"pts\" and \"pfs\", each an array of strings.\n"
        "3. Use ONLY labels from the lists above, copied exactly.\n"
        "4. Put PT labels in \"pts\" and PF labels in \"pfs\".\n"
        '5. If none apply, return {"pts": [], "pfs": []}.'
    )


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_output(text):
    """Extract pts/pfs lists from model text. Returns (pts, pfs)."""
    if text is None:
        return [], []
    # strip any thinking traces
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    m = JSON_RE.search(text)
    if not m:
        return [], []
    # try progressively shorter spans in case of trailing junk
    candidate = m.group(0)
    for attempt in (candidate, candidate[:candidate.rfind("}") + 1]):
        try:
            obj = json.loads(attempt)
            pts = [str(x).strip().lower() for x in obj.get("pts", [])
                   if isinstance(x, str)]
            pfs = [str(x).strip().lower() for x in obj.get("pfs", [])
                   if isinstance(x, str)]
            return pts, pfs
        except Exception:
            continue
    return [], []


def with_retry(fn, *args, **kwargs):
    """Call fn with exponential-backoff retries. Returns fn result or raises."""
    last = None
    for i in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            wait = RETRY_BASE * (2 ** i)
            print(f"    retry {i+1}/{MAX_RETRIES} after error: {e} "
                  f"(waiting {wait:.0f}s)")
            time.sleep(wait)
    raise last


def run_provider(provider_name, call_model):
    """Shared driver. `call_model(system, user)` must return raw text.

    Writes candidates/<provider>_<timestamp>.jsonl with one record per
    message: id, message, pts, pfs, and the raw response.
    """
    pt_pairs = load_taxonomy(TAXONOMY_PT)
    pf_pairs = load_taxonomy(TAXONOMY_PF)
    system = build_system_prompt(pt_pairs, pf_pairs)
    records = load_data(INPUT_FILE)

    print(f"Provider: {provider_name}")
    print(f"Taxonomy: {len(pt_pairs)} PTs, {len(pf_pairs)} PFs")
    print(f"Messages: {len(records)}")

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(OUTPUT_DIR) / f"{provider_name}_{ts}.jsonl"
    print(f"Output: {out_path}\n")

    n_done, n_fail = 0, 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in records:
            try:
                raw = with_retry(call_model, system, rec["message"])
            except Exception as e:
                raw = None
                n_fail += 1
                print(f"  [{rec['id']}] FAILED after retries: {e}")

            pts, pfs = parse_output(raw)
            fout.write(json.dumps({
                "id":      rec["id"],
                "message": rec["message"],
                "pts":     pts,
                "pfs":     pfs,
                "raw":     raw,
            }, ensure_ascii=False) + "\n")
            fout.flush()

            n_done += 1
            if n_done % 20 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(records) - n_done) / rate
                print(f"  [{n_done}/{len(records)}] {rate:.2f}/s, "
                      f"ETA {eta/60:.1f} min")

    dt = (time.time() - t0) / 60
    print(f"\nDone. {n_done} written ({n_fail} failed) in {dt:.1f} min.")
    print(f"Output: {out_path}")
