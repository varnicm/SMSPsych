#!/usr/bin/env python
'''
Run one SLM zero-shot on the 396-message test set and save PT/PF
predictions to predictions_396/<model>_zero_shot_<time>.jsonl
'''

import os
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# ============================ CONFIG ============================

# CHANGE THIS LINE PER RUN
# MODEL_ID = "google/gemma-4-E4B-it"
# MODEL_ID = "microsoft/Phi-4-mini-instruct"
#MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_ID = "Qwen/Qwen3.5-4B"
# MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_ID = "Qwen/Qwen3.5-9B"   
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"



INPUT_FILE   = "gold_final.jsonl"
TAXONOMY_PT  = "pts.txt"
TAXONOMY_PF  = "pfs.txt"

OUTPUT_DIR   = "predictions_396"
USE_FEW_SHOT = False
USE_DEFINITIONS = False

MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.0       # deterministic
DO_SAMPLE      = False

# Models that support a reasoning/thinking toggle in their chat template.
# We disable thinking so the model outputs JSON directly without <think>...
THINKING_MODELS = {
    "Qwen/Qwen3.5-4B",
    "google/gemma-4-E4B-it",
    "openbmb/MiniCPM5-1B",
    "Qwen/Qwen3.5-9B",

}

# ================================================================


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def load_taxonomy(path: str) -> list[tuple[str, str]]:
    """Read 'label: definition' lines, return list of (label, definition) pairs.
    Skips comment lines starting with #."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                label, definition = line.split(":", 1)
                label = label.strip().lower()
                definition = definition.strip()
            else:
                label = line.strip().lower()
                definition = ""
            if label:
                pairs.append((label, definition))
    return pairs


def load_data(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------- few-shot examples (LOCKED, all from real dataset) ---------------------------
FEW_SHOT = [
    {
        # smishing-1: Chase Alert (hand-reviewed, msg-148 in original corpus)
        "message": ("Chase Alert (SM): We noticed unusual activities on your "
                    "account. Visit chaselocked.com/web to verify the "
                    "activities. Reply STOP to stop. Msg & data rates may apply."),
        "pts": ["pretexting", "impersonation", "urgency",
                "trusted relationship", "loss aversion"],
        "pfs": ["trust", "fear", "authority"],
    },
    {
        # smishing-2: Amazon raffle (hand-reviewed, msg-480 in original corpus)
        "message": ("Amazon: Congratulations Aashish, you came in 1st in this "
                    "week's Amazon Earpods raffle! Click the link to arrange "
                    "shipment: u5fxc.info/ZEJTTZmY3Z"),
        "pts": ["pretexting", "impersonation", "incentive and motivator",
                "urgency"],
        "pfs": ["trust", "greed"],
    },
    {
        # benign-1: hand-reviewed conversational ham (msg-1112)
        "message": ("Can you please ask macho what his price range is, does "
                    "he want something new or used plus it he only interfued "
                    "in the blackberry bold <#> or any bb"),
        "pts": [],
        "pfs": [],
    },
    {
        # benign-2: real ham from Mendeley (seed=7 pick, candidate 2)
        "message": ("Yo you guys ever figure out how much we need for alcohol? "
                    "Jay and I are trying to figure out how much we can safely "
                    "spend on weed"),
        "pts": [],
        "pfs": [],
    },
]


def build_messages2(message: str, pt_pairs: list[tuple[str, str]],
                   pf_pairs: list[tuple[str, str]],
                   few_shot: bool) -> list[dict]:
    def fmt(pairs):
        # one per line, "label: definition"
        return "\n".join(
            f"- {lab}: {defn}" if defn else f"- {lab}"
            for lab, defn in pairs
        )
    # def fmt(pairs):
    # if USE_DEFINITIONS:
    #     return "\n".join(
    #         f"- {lab}: {defn}" if defn else f"- {lab}"
    #         for lab, defn in pairs
    #     )
    # else:
    #     return ", ".join(lab for lab, _ in pairs)

    system = (
        "You are an expert in social engineering and SMS phishing (smishing) "
        "analysis. Given an SMS message, identify which Psychological "
        "Techniques (PTs) and Psychological Factors (PFs) a smishing attacker "
        "uses to deceive the recipient. "
        "Only label techniques and factors that are part of a deceptive or "
        "manipulative attempt. "
        "Legitimate (benign) messages — including personal conversations, "
        "transactional notifications, or messages expressing real urgency, "
        "social bonds, or emotion in good faith — should return empty PT and "
        "PF lists, even if they superficially contain words like 'hurry', "
        "'trust', or 'authority'. "
        "Output ONLY a JSON object of the form: "
        '{"pts": [...], "pfs": [...]}. '
        f"\n\nAllowed Psychological Techniques (PTs):\n{fmt(pt_pairs)}"
        f"\n\nAllowed Psychological Factors (PFs):\n{fmt(pf_pairs)}"
        "\n\nSTRICT OUTPUT RULES:\n"
        "1. Respond with ONLY a JSON object. Your entire response must start "
        "with '{' and end with '}'. No explanation, preamble, or commentary.\n"
        "2. The JSON must have exactly two keys: \"pts\" and \"pfs\", each "
        "containing an array of strings.\n"
        "3. Use ONLY labels from the lists above. Do not invent or paraphrase "
        "labels. Each label must appear EXACTLY as written (lowercase, "
        "matching spelling).\n"
        "4. Put PT labels in \"pts\" and PF labels in \"pfs\". Never mix them.\n"
        "5. If no PT or PF applies, return empty arrays: "
        '{"pts": [], "pfs": []}'        
    )

    msgs = [{"role": "system", "content": system}]

    if few_shot:
        for ex in FEW_SHOT:
            msgs.append({"role": "user", "content": ex["message"]})
            msgs.append({
                "role": "assistant",
                "content": json.dumps({"pts": ex["pts"], "pfs": ex["pfs"]}),
            })

    msgs.append({"role": "user", "content": message})
    return msgs

def build_messages(message: str, pt_pairs: list[tuple[str, str]],
                   pf_pairs: list[tuple[str, str]],
                   few_shot: bool) -> list[dict]:
    def fmt(pairs):
        return "\n".join(
            f"- {lab}: {defn}" if defn else f"- {lab}"
            for lab, defn in pairs
        )
    system = (
        "You are an expert annotator identifying psychological techniques "
        "(PTs) and psychological factors (PFs) in SMS messages, following "
        "Longtchi et al. (2024). PTs are influence strategies the sender uses "
        "(how the message tries to influence the receiver). PFs are human "
        "attributes the message appeals to or targets in the receiver (what "
        "the message draws on). A single message may contain multiple PTs and "
        "multiple PFs. Label any PT or PF genuinely present, whether the "
        "message is malicious or benign. "
        "Respond with ONLY a JSON object of the form: "
        '{"pts": [...], "pfs": [...]}. '
        "Use names copied EXACTLY from the provided lists; do not invent "
        "labels. Use [] for either list if none apply. Add no explanation."
        f"\n\nPsychological Techniques (PTs):\n{fmt(pt_pairs)}"
        f"\n\nPsychological Factors (PFs):\n{fmt(pf_pairs)}"
        "\n\nSTRICT OUTPUT RULES:\n"
        "1. Respond with ONLY a JSON object. Your entire response must start "
        "with '{' and end with '}'. No explanation, preamble, or commentary.\n"
        "2. The JSON must have exactly two keys: \"pts\" and \"pfs\", each "
        "containing an array of strings.\n"
        "3. Use ONLY labels from the lists above, copied exactly (lowercase, "
        "matching spelling). Do not invent or paraphrase labels.\n"
        "4. Put PT labels in \"pts\" and PF labels in \"pfs\". Never mix them.\n"
        "5. If no PT or PF applies, return empty arrays: "
        '{"pts": [], "pfs": []}'
    )

    msgs = [{"role": "system", "content": system}]

    if few_shot:
        for ex in FEW_SHOT:
            msgs.append({"role": "user", "content": ex["message"]})
            msgs.append({
                "role": "assistant",
                "content": json.dumps({"pts": ex["pts"], "pfs": ex["pfs"]}),
            })

    msgs.append({"role": "user", "content": message})
    return msgs
# --------------------------- JSON parsing ---------------------------
JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

def parse_output(text: str) -> tuple[list[str], list[str], str]:
    """Extract PT and PF lists from the model output. Returns (pts, pfs, raw)."""
    # If a chain-of-thought slipped through, strip it
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if "<think>" in text and "</think>" not in text:
        text = ""   # malformed, give up

    # Find the JSON object
    for match in JSON_RE.finditer(text):
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            pts = [str(x).strip().lower() for x in obj.get("pts", [])
                   if isinstance(x, str)]
            pfs = [str(x).strip().lower() for x in obj.get("pfs", [])
                   if isinstance(x, str)]
            return pts, pfs, candidate
        except Exception:
            continue
    return [], [], text.strip()


# --------------------------- main ---------------------------
def main():
    print(f"Model: {MODEL_ID}")
    print(f"Input: {INPUT_FILE}")
    print(f"Few-shot: {USE_FEW_SHOT}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Taxonomy
    pts = load_taxonomy(TAXONOMY_PT)
    pfs = load_taxonomy(TAXONOMY_PF)
    print(f"Loaded taxonomy: {len(pts)} PTs, {len(pfs)} PFs (with definitions)")

    # Data
    records = load_data(INPUT_FILE)
    print(f"Loaded {len(records)} messages")
    
    
    records = load_data(INPUT_FILE)
    print(f"Loaded {len(records)} messages")
    
    SAMPLE_LIMIT = None   # set to None for full data
    if SAMPLE_LIMIT:
        import random as _r
        _r.seed(42)
        records = _r.sample(records, SAMPLE_LIMIT)
        print(f"Sampled {len(records)} for testing")



    # Output path
    # Path(OUTPUT_DIR).mkdir(exist_ok=True)
    # suffix = "few_shot" if USE_FEW_SHOT else "zero_shot"
    # out_path = Path(OUTPUT_DIR) / f"{short_name(MODEL_ID)}_{suffix}.jsonl"
    # print(f"Output: {out_path}")

    # Output path
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    suffix = "few_shot" if USE_FEW_SHOT else "zero_shot"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(OUTPUT_DIR) / f"{short_name(MODEL_ID)}_{suffix}_{timestamp}.jsonl"
    print(f"Output: {out_path}")

    # Pre-download model to HF cache
    print(f"\nDownloading model weights ...")
    snapshot_download(repo_id=MODEL_ID)

    # Load tokenizer + model
    print("Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)

    print("Loading model (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()

    # Chat template arguments — disable thinking where supported
    apply_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if MODEL_ID in THINKING_MODELS:
        apply_kwargs["enable_thinking"] = False

    # Inference loop
    n_done = 0
    t_start = time.time()
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in records:
            msgs = build_messages(rec["message"], pts, pfs, USE_FEW_SHOT)
            try:
                prompt = tok.apply_chat_template(msgs, **apply_kwargs)
            except TypeError:
                # older tokenizers don't accept enable_thinking
                apply_kwargs.pop("enable_thinking", None)
                prompt = tok.apply_chat_template(msgs, **apply_kwargs)

            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE if DO_SAMPLE else None,
                    pad_token_id=tok.eos_token_id,
                )
            raw = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)

            pt_pred, pf_pred, parsed = parse_output(raw)
            rec_out = {
                "id":       rec["id"],
                "message":  rec["message"],
                #"source":   rec["source"],
                "source":   rec.get("source", "unknown"),
                "length":   rec.get("length"),
                "pts":      pt_pred,
                "pfs":      pf_pred,
                "raw":      raw,
            }
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            fout.flush()

            n_done += 1
            if n_done % 20 == 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed
                eta = (len(records) - n_done) / rate
                print(f"  [{n_done:>4}/{len(records)}] "
                      f"{rate:.2f} msgs/s, ETA {eta/60:.1f} min")

    elapsed = time.time() - t_start
    print(f"\nDone. Wrote {n_done} predictions to {out_path} in "
          f"{elapsed/60:.1f} min.")


    # # at the bottom of main(), change:
    # out_path = Path(OUTPUT_DIR) / f"{short_name(MODEL_ID)}_{suffix}.jsonl"
    # # to:
    # defn_tag = "withdefs" if USE_DEFINITIONS else "nodefs"
    # out_path = Path(OUTPUT_DIR) / f"{short_name(MODEL_ID)}_{suffix}_{defn_tag}.jsonl"


if __name__ == "__main__":
    main()