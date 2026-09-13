# Behavioral Robustness of Small Language Models for Smishing Detection

This repository contains the reproducible pipeline for evaluating six instruction-tuned small language models on SMS smishing detection under textual perturbations and prompt-configuration changes.

The study evaluates three observable dimensions:

1. **Prediction robustness** — changes in predicted labels under character-, word-, sentence-, and multi-level perturbations.
2. **Prompt-configuration sensitivity** — variation across four prompt configurations formed by neutral/expert framing and label-only/explanation-required output.
3. **Explanation stability and compliance** — similarity between non-empty rationales generated for clean and perturbed messages, conditioned on both predictions being correct.

The central analysis separates three distinct outcomes:

- overall output change,
- smishing-to-ham attack success rate (ASR),
- ham-to-smishing conditional benign-flip rate.

This separation is important because low evasion can coexist with severe instability on legitimate messages.

---

## Repository layout

```text
smishing_behavioral_robustness/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── configs/
│   ├── experiment.yaml
│   ├── models.yaml
│   └── prompts.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manual_audit/
├── scripts/
│   ├── 01_prepare_data/
│   │   ├── build_balanced_corpus.py
│   │   └── validate_corpus.py
│   ├── 02_generate_perturbations/
│   │   ├── perturbations.py
│   │   ├── run_perturbations.py
│   │   ├── validate_perturbations.py
│   │   └── deduplicate_sentence_variants.py
│   ├── 03_run_inference/
│   │   ├── run_inference_batched.py
│   │   ├── rerun_explanation_prompts.py
│   │   └── parse_model_outputs.py
│   ├── 04_postprocess/
│   │   ├── combine_predictions.py
│   │   ├── normalize_template_names.py
│   │   ├── dedup_sentence_predictions.py
│   │   └── integrity_checks.py
│   ├── 05_analyze_prediction/
│   │   ├── bootstrap_analysis.py
│   │   ├── transition_analysis.py
│   │   └── clean_accuracy.py
│   ├── 06_analyze_prompt/
│   │   └── prompt_agreement.py
│   ├── 07_analyze_explanations/
│   │   ├── analyze_explanation_stability.py
│   │   ├── validate_nli_metric.py
│   │   └── rationale_compliance.py
│   └── 08_make_tables_figures/
│       ├── make_tables.py
│       ├── plot_asr_vs_benign_flip.py
│       └── plot_attack_profiles.py
├── slurm/
│   ├── 01_perturbations.slurm
│   ├── 02_inference.slurm
│   ├── 03_explanation_inference.slurm
│   ├── 04_bootstrap.slurm
│   └── 05_explanation_stability.slurm
├── outputs/
│   ├── perturbations/
│   ├── predictions/
│   ├── analysis/
│   │   ├── prediction/
│   │   ├── prompt/
│   │   └── explanations/
│   ├── tables/
│   └── figures/
├── tests/
│   ├── test_perturbations.py
│   ├── test_deduplication.py
│   ├── test_bootstrap_metrics.py
│   └── test_explanation_filtering.py
└── docs/
    ├── DATA_CARD.md
    ├── MODEL_CARD.md
    ├── REPRODUCIBILITY.md
    └── MANUAL_AUDIT_PROTOCOL.md
```

---

## Data flow

### 1. Build the evaluated corpus

The source corpus contains:

- 1,055 smishing messages from Smishtank,
- 1,055 ham messages sampled from the Mendeley SMS dataset.

Three ham messages produced no valid perturbation under any attack/seed and are excluded from the evaluated set. The final evaluated corpus therefore contains:

- 1,055 smishing,
- 1,052 ham,
- **2,107 total messages**.

Expected clean prediction rows per model:

```text
2,107 messages × 4 prompt configurations = 8,428 rows
```

Run:

```bash
python scripts/01_prepare_data/build_balanced_corpus.py \
  --smishing data/raw/smishtank.csv \
  --ham data/raw/mendeley_sms.csv \
  --output data/interim/balanced_source.csv

python scripts/01_prepare_data/validate_corpus.py \
  --input data/interim/balanced_source.csv
```

---

### 2. Generate perturbations

Four perturbation classes are used:

- **Character:** homoglyphs, intra-word spacing, and misspellings.
- **Word:** synonym substitution using counter-fitted embeddings.
- **Sentence:** deterministic PEGASUS beam-search paraphrasing.
- **Multi-level:** a sequential chain of word, character, and sentence transformations.

For character and word attacks, approximately 30% of eligible tokens are selected without replacement. A token is eligible when it:

- is not a masked URL or structured placeholder,
- contains at least two alphabetic characters.

The selected count is rounded to the nearest integer with a minimum of one token.

Three seeds are used for character, word, and multi-level attacks. Sentence-level decoding is deterministic, so only one distinct paraphrase is retained per source message.

A semantic-similarity gate retains only variants with:

```text
cosine(original, perturbed) ≥ 0.65
```

using `sentence-transformers/all-MiniLM-L6-v2`.

Run:

```bash
python scripts/02_generate_perturbations/run_perturbations.py \
  --input data/processed/evaluated_corpus.csv \
  --output outputs/perturbations/all_variants.jsonl \
  --counter-fitted data/raw/counter-fitted-vectors.txt \
  --frac 0.30 \
  --seeds 0 1 2 \
  --sim-threshold 0.65 \
  --attacks character word sentence multi \
  --device cuda
```

Validate:

```bash
python scripts/02_generate_perturbations/validate_perturbations.py \
  --input outputs/perturbations/all_variants.jsonl
```

Deduplicate deterministic sentence-level rows:

```bash
python scripts/02_generate_perturbations/deduplicate_sentence_variants.py \
  --input outputs/perturbations/all_variants.jsonl \
  --output outputs/perturbations/valid_variants_dedup.jsonl
```

Expected valid perturbed variants after deduplication:

| Attack | Valid variants |
|---|---:|
| Character | 5,339 |
| Word | 5,611 |
| Sentence | 1,637 |
| Multi-level | 2,265 |
| **Total** | **14,852** |

---

### 3. Run model inference

Each clean message and valid perturbed variant is evaluated using:

- six models,
- four prompt configurations.

The four prompts form a 2×2 design:

| Prompt | Role framing | Output mode |
|---|---|---|
| t1 | neutral | label only |
| t2 | expert | label only |
| t3 | neutral | explanation required |
| t4 | expert | explanation required |

The required final line is:

```text
FINAL: smishing
```

or

```text
FINAL: ham
```

Run:

```bash
python scripts/03_run_inference/run_inference_batched.py \
  --clean data/processed/evaluated_corpus.csv \
  --perturbed outputs/perturbations/valid_variants_dedup.jsonl \
  --models-config configs/models.yaml \
  --prompts-config configs/prompts.yaml \
  --output-dir outputs/predictions/by_model
```

Expected final prediction count:

```text
(2,107 clean + 14,852 perturbed) × 4 prompts × 6 models
= 407,016 prediction rows
```

---

### 4. Combine and validate predictions

```bash
python scripts/04_postprocess/combine_predictions.py \
  --input-dir outputs/predictions/by_model \
  --output outputs/predictions/preds_all.jsonl

python scripts/04_postprocess/normalize_template_names.py \
  --input outputs/predictions/preds_all.jsonl \
  --output outputs/predictions/preds_normalized.jsonl

python scripts/04_postprocess/dedup_sentence_predictions.py \
  --input outputs/predictions/preds_normalized.jsonl \
  --output outputs/predictions/preds_dedup.jsonl

python scripts/04_postprocess/integrity_checks.py \
  --preds outputs/predictions/preds_dedup.jsonl
```

Required integrity conditions:

```text
dup_clean = 0
dup_pert = 0
label_conflicts = 0
missing_clean_cells = 0
```

The integrity script should terminate with a non-zero exit code if any condition fails.

---

### 5. Prediction robustness analysis

The primary analysis uses a message-level cluster bootstrap:

- resampling unit: original message,
- stratified by true class,
- 1,000 bootstrap replicates,
- 95% percentile confidence intervals.

Run once per attack:

```bash
python scripts/05_analyze_prediction/bootstrap_analysis.py \
  --preds outputs/predictions/preds_dedup.jsonl \
  --out-dir outputs/analysis/prediction/character \
  --B 1000 \
  --attack character
```

Repeat with `word`, `sentence`, and `multi`.

Report:

- overall clean accuracy,
- matched clean and perturbed accuracy,
- matched ΔAcc,
- strict label-flip rate,
- output-change rate,
- conditional ASR,
- conditional benign-flip rate,
- unparsed-output rate.

Conditional denominators:

```text
ASR:
clean-correct smishing messages

Benign flip:
clean-correct ham messages
```

Message-weighted values are primary. Variant-weighted values may be retained as secondary diagnostics.

Run transition analysis:

```bash
python scripts/05_analyze_prediction/transition_analysis.py \
  --preds outputs/predictions/preds_dedup.jsonl \
  --output-dir outputs/analysis/prediction/transitions
```

---

### 6. Prompt-configuration sensitivity

Compute:

- PAR-majority,
- PAR-unanimous,
- pairwise disagreement,
- results by clean condition and attack class.

```bash
python scripts/06_analyze_prompt/prompt_agreement.py \
  --preds outputs/predictions/preds_dedup.jsonl \
  --output-dir outputs/analysis/prompt
```

Interpretation:

```text
1 − PAR-unanimous
```

is the fraction of inputs for which at least one of the four prompt configurations disagrees.

Do not interpret `1 − PAR-majority` as a disagreement rate.

---

### 7. Explanation compliance and stability

Explanation analysis uses only t3 and t4.

A pair is eligible only when:

1. the model predicts the correct label on the clean message,
2. the model predicts the correct label on the perturbed message,
3. both outputs contain non-empty rationales.

Empty rationales are excluded and reported as instruction-compliance failures.

Run:

```bash
python scripts/07_analyze_explanations/analyze_explanation_stability.py \
  --preds outputs/predictions/explanations_all.jsonl \
  --out-dir outputs/analysis/explanations/stability \
  --bertscore \
  --sentcos \
  --batch-size 32
```

Primary metric:

- BERTScore F1.

Secondary metrics:

- sentence-embedding cosine,
- ROUGE-L,
- token Jaccard.

Important interpretation:

- these metrics measure similarity,
- they do not establish faithfulness, factual correctness, grounding, or causal reasoning.

Validate the rejected NLI metric:

```bash
python scripts/07_analyze_explanations/validate_nli_metric.py \
  --preds outputs/predictions/explanations_all.jsonl \
  --messages outputs/perturbations/valid_variants_dedup.jsonl \
  --output outputs/analysis/explanations/nli_validation.json
```

The shuffled-message control is retained as a methodological negative result.

---

### 8. Generate paper tables and figures

```bash
python scripts/08_make_tables_figures/make_tables.py \
  --analysis-root outputs/analysis \
  --output-dir outputs/tables

python scripts/08_make_tables_figures/plot_asr_vs_benign_flip.py \
  --input outputs/analysis/prediction/character/bootstrap_summary.json \
  --output outputs/figures/asr_vs_benign_flip.pdf

python scripts/08_make_tables_figures/plot_attack_profiles.py \
  --analysis-root outputs/analysis/prediction \
  --output outputs/figures/attack_profiles.pdf
```

Recommended main artifacts:

1. clean accuracy table,
2. ASR and benign-flip table,
3. prompt-agreement table,
4. explanation-compliance and stability table,
5. ASR-versus-benign-flip figure,
6. attack-profile figure.

---

## Reproducibility notes

- Use the exact Hugging Face model revisions recorded in `configs/models.yaml`.
- Greedy decoding should be used for classification.
- Record package versions in `environment.yml` and `requirements.txt`.
- Store large model weights and raw datasets outside Git.
- Do not commit generated predictions or model caches.
- Keep older or invalid analyses outside the main output directories.
- Treat `outputs/predictions/preds_dedup.jsonl` as the authoritative prediction file.
- Treat the corrected `B=1000` bootstrap outputs as the authoritative prediction-analysis results.
- Treat the empty-rationale-filtered explanation output as the authoritative explanation analysis.

---

## Manual perturbation audit

A manual audit is recommended before publication.

Suggested sample:

- 25 variants per attack,
- balanced across smishing and ham,
- 100 variants total.

For each variant, annotate:

- Is the original label preserved?
- Is the message understandable?
- Is the payload preserved?
- Is malicious or benign intent preserved?
- Does the perturbation introduce new meaning?

See `docs/MANUAL_AUDIT_PROTOCOL.md`.

---

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate smishing-robustness
```

Or pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Citation

Add the final paper citation here after publication.

---

## License

Code can be released under the MIT License. Dataset redistribution must follow the licenses and terms of the original Smishtank and Mendeley sources.

