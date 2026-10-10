# BERT-clean (primary) — fine-tuned weights

**Size:** ~387 MB zipped · **Owner:** z5571377

> Download: https://unsw-my.sharepoint.com/:u:/g/personal/z5571377_ad_unsw_edu_au/IQDPXBSNdMrJTbXlPOgxuSWTAaRJYj-bTu3seYprf82PLc4?e=68GNUp

This is the **primary BERT model** reported in the main results table —
trained purely on `train.csv` (60 k Jigsaw rows) with no manual-set
augmentation, so the full 232-row `manual_annotation_200.csv` can be
used as an honest OOD evaluation.

After downloading `bert_clean_weights.zip`, unzip so the contents land
**directly inside this folder** (the zip is flat — no wrapping folder):

```
MISC/saved_model/bert_clean/
├── README.md                 (this file)
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

This is the path `cli.py`, `app.py`, and `evaluate_all.py` look for by
default when `--model bert-clean` is selected. To override, pass
`--model-dir /some/other/path`.
