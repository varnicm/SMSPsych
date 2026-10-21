# evaluation/

Scripts that compute the scores and tables reported in the paper.

## Scripts

| Script | Paper table | Prediction files needed |
|---|---|---|
| `label-distribution-table.py` | Table II: label counts in smishing and benign messages | none (gold files only) |
| `make_per_label_f1.py` | Table III: per-label F1 on smishing messages | given on the command line |
| `make_micro_macro.py` | Table IV, upper panel: all messages | given on the command line |
| `make_micro_macro_smish.py` | Table IV, lower panel: smishing messages | given on the command line |
| `make_spr_by_class.py` | Table V: safe-prediction rate (SPR) by message class | given on the command line |
| `score_table4_30.py` | Table VI: encoder 5-fold cross-validation | fixed inside the script |
| `make_per_label_all.py` | Extra: per-label precision, recall, and F1 (not in the paper) | given on the command line |

## How to run

The scripts read all files from the folder they run in. Copy these files into one folder first:

- from `data/`: `gold_final.jsonl`, `gold_final_batch2.jsonl`
- from `predictions/prompted/`, `predictions/lora/`, `predictions/encoder/`: all prediction files
- from `evaluation/`: the scripts

Then run:

```bash
P="gemma-4-e4b-it_zero_shot_20260808_210832.jsonl qwen3.5-9b_zero_shot_20260808_213216.jsonl qwen3.5-4b_zero_shot_20260808_211903.jsonl llama-3.1-8b-instruct_zero_shot_20260808_214558.jsonl phi-3-mini-4k-instruct_zero_shot_20260808_215450.jsonl"
L="lora_gemma_pred.jsonl lora_qwen_pred.jsonl lora_qwen9b_pred.jsonl lora_llama_pred.jsonl lora_phi_pred.jsonl"
B="bert_batch2_all_pred_thr08.jsonl"

python label-distribution-table.py
python make_per_label_f1.py      $P $L $B
python make_micro_macro.py       $P $L $B
python make_micro_macro_smish.py $P $L $B
python make_spr_by_class.py      $P $L $B
python score_table4_30.py
python make_per_label_all.py     $P $L $B
```

The scripts detect each model from its file name, so keep the original file names.

## Notes

- Per-label, micro, and macro scores use the 25 labels with at least 30 occurrences in the full corpus (`TAU=30`).
- SPR uses each model's full predicted label set, not only the 25 evaluated labels.
- `make_spr_by_class.py` does not remove out-of-taxonomy labels before scoring. For the prompted models, its benign SPR can be up to 0.01 lower than in Table V of the paper.
- `score_table4_30.py` reads `bert_expanded_all_pred_thr08.jsonl` (the 5-fold cross-validation predictions).
- The saved outputs of all these scripts are in `results/`.
