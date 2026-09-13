# SMSPsych

**Psychological technique and factor extraction from SMS messages**

SMSPsych is the research artifact for a study of psychological manipulation in smishing and benign SMS messages. The project examines how compact language models identify psychological techniques and psychological factors and how the extracted labels can support deterministic explanations.

## Project scope

SMSPsych focuses on two complementary dimensions of message-level manipulation.

- **Psychological techniques (PTs)** describe how a sender constructs or delivers a persuasive or deceptive message.
- **Psychological factors (PFs)** describe what psychological mechanism the message attempts to activate in a recipient.

The current study evaluates PT and PF extraction. It does not evaluate binary smishing detection. Message class is provided as a known input rather than predicted by the extractor.

## Research questions

1. How are PTs and PFs distributed across annotated smishing and benign SMS messages?
2. How accurately can compact language models extract PTs and PFs, and how does fine-tuning affect performance?
3. How frequently do the evaluated models produce fully reference-supported label sets for deterministic explanations?

## Dataset

The study uses a balanced corpus of 2,110 English SMS messages.

| Message class | Messages |
| --- | ---: |
| Smishing | 1,055 |
| Benign | 1,055 |
| **Total** | **2,110** |

The annotation taxonomy contains 16 PTs and 46 PFs. Fifty-one labels occur in the annotated corpus. The reported per-label evaluation covers 25 labels with at least 30 occurrences, which include 14 PTs and 11 PFs.

The primary evaluation uses a fixed test set of 396 messages with 198 messages from each class. The remaining 1,714 messages form the development pool. A separate five-fold cross-validation analysis is conducted for the encoder.

## Reference annotation

Three language models independently propose candidate labels for each message. A human reviewer examines the message, proposed labels, and taxonomy definitions before finalizing the reference annotation. The review follows a mechanism-only principle. A label is retained only when the message provides evidence for the corresponding mechanism.

## Evaluated approaches

SMSPsych compares 11 model configurations.

### Zero-shot models

- Gemma 4 E4B
- Qwen3.5 9B
- Qwen3.5 4B
- Llama 3.1 8B Instruct
- Phi 3 Mini

### Fine-tuned models

- LoRA-adapted versions of the five generative models
- A 110M-parameter `bert-base-uncased` multilabel encoder

These configurations support a comparison of zero-shot prompting, parameter-efficient adaptation, and encoder-based fine-tuning.

## Explanation pipeline

SMSPsych uses deterministic explanation assembly rather than free-form explanation generation. Each taxonomy label maps to a fixed clause. The system combines only the clauses associated with the extracted labels.

This design prevents the explanation module from introducing labels or psychological mechanisms that the extractor did not return. It does not correct extraction errors. An incorrect extracted label will still produce its corresponding clause.

## Evaluation

The study reports the following measures.

- Micro-averaged and macro-averaged precision, recall, and F1
- Per-label extraction performance
- Safe-Prediction Rate (SPR)
- Empty-Prediction Rate (EPR)
- Invalid-label rate

SPR evaluates whether a complete taxonomy-restricted predicted label set is supported by the reference annotation. Strict SPR requires full reference support. Lower support thresholds are examined through a sensitivity analysis.

## Repository contents

This repository will provide the materials required to reproduce the SMSPsych study, subject to the redistribution terms of the original data sources.

- PT and PF taxonomy definitions
- Annotation and preprocessing resources
- Zero-shot inference configurations
- LoRA training configurations
- Encoder training and evaluation code
- Deterministic explanation templates
- Metric implementation and analysis scripts
- Reproducibility instructions

The directory structure and execution commands will be documented as the research artifact is finalized.

## Responsible use

SMSPsych is intended for cybersecurity research and education. PF labels represent mechanisms inferred from message content. They do not measure a recipient's actual psychological state, susceptibility, intent, or behavior. The outputs should not be treated as psychological diagnoses or used as the sole basis for security decisions.

## Citation

If you use SMSPsych, please cite the repository. The citation for the accompanying paper will be added after publication.

```bibtex
@software{smspsych2026,
  author = {Sanjari Pirmahalleh, Seyed Mohammad and Pritom, Mir Mehedi Ahsan},
  title = {{SMSPsych}},
  year = {2026},
  url = {https://github.com/varnicm/SMSPsych}
}
```

## Contact

For questions or research collaboration, open an issue in this repository or contact the maintainer through the [GitHub profile](https://github.com/varnicm).

## License

License information will be added before the complete research artifact is released. Any released data will remain subject to the terms of the original data sources.
