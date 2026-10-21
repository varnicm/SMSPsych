# results/

Saved outputs of the scripts in `evaluation/`. Each file is the plain-text output of one script, so the numbers can be read without running anything.

## Files

| File | Paper table | Script |
|---|---|---|
| `table2_label_distribution.txt` | Table II: label counts in smishing and benign messages | `label-distribution-table.py` |
| `table3_per_label_f1.txt` | Table III: per-label F1 on smishing messages | `make_per_label_f1.py` |
| `table4_all_messages.txt` | Table IV, upper panel: all 396 test messages | `make_micro_macro.py` |
| `table4_smishing.txt` | Table IV, lower panel: 198 smishing test messages | `make_micro_macro_smish.py` |
| `table5_spr_by_class.txt` | Table V: SPR by message class | `make_spr_by_class.py` |
| `table6_bert_cv5.txt` | Table VI: encoder 5-fold cross-validation over 2,110 messages | `score_table4_30.py` |
| `extra_per_label_prf.txt` | Not in the paper: per-label precision, recall, and F1 | `make_per_label_all.py` |

## Column names

**`table4_*.txt`**
- `PT` / `PF`: psychological techniques / psychological factors
- `mi` / `ma`: micro-average / macro-average
- `P`, `R`, `F`: precision, recall, F1 (for example, `PTmiP` = technique micro-precision)
- `S50` to `S100`: SPR at reference-support thresholds from 50% to 100% (`S100` = strict SPR)

**`table5_spr_by_class.txt`**
- `SM50` to `SM100`: SPR on smishing messages
- `BN50` to `BN100`: SPR on benign messages

**`table2_label_distribution.txt`**
- Each line is `label|smishing count|benign count`.

## Model names

| Name in the files | Configuration |
|---|---|
| `Gemma-4B(prompt)`, `Qwen-9B(prompt)`, `Qwen-4B(prompt)`, `Llama-8B(prompt)`, `Phi-3(prompt)` | zero-shot prompted models |
| `LoRA Gemma-4B`, `LoRA Qwen-4B`, `LoRA Qwen-9B`, `LoRA Llama-8B`, `LoRA Phi-3` | LoRA-adapted models |
| `BERT(all-label)` | fine-tuned encoder (`bert-base-uncased`), threshold 0.8 |

## Notes

- Per-label, micro, and macro scores use the 25 labels with at least 30 occurrences in the full corpus.
- The numbers here are the direct script outputs. Two small differences from the paper are known:
  - `table5_spr_by_class.txt`: for the prompted models, benign SPR is up to 0.01 lower than in Table V, because the script does not remove out-of-taxonomy labels before scoring.
  - `table4_smishing.txt`: LoRA Qwen-4B has factor micro-precision 0.79 and micro-recall 0.84 (the paper lists 0.78 and 0.83).
