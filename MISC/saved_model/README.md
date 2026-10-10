# saved_model/ — model weights

Naive Bayes weights are small enough to ship directly inside this folder
(`naive_bayes_model.pkl` + `tfidf_vectorizer.pkl`, ≈ 26 MB total — below
the 200 MB Moodle limit).
The four transformer checkpoints each exceed Moodle's 200 MB upload
limit, so they are hosted on OneDrive instead. This README is the single
place to find every download link + where to put the files.

```
saved_model/
├── README.md                       (this file — all OneDrive links below)
├── naive_bayes/
│   ├── naive_bayes_model.pkl       (bundled)
│   └── tfidf_vectorizer.pkl        (bundled)
├── roberta_base/   <- download from OneDrive, unzip here
├── distilbert/     <- download from OneDrive, unzip here
├── bert_clean/     <- download from OneDrive, unzip here
└── bert_aug/       <- download from OneDrive, unzip here
```

---

## OneDrive download links

| Model | Size | Link |
|---|---:|---|
| RoBERTa-base (owner: z5571377) | ~430 MB | https://unsw-my.sharepoint.com/:u:/g/personal/z5571377_ad_unsw_edu_au/IQDESbOAmlH-SZriP3aOIpnkAY0r-48ZiMNcH_o3rxAbqbs?e=y1ZCuq |
| DistilBERT (owner: z5571022) | ~250 MB | https://unsw-my.sharepoint.com/:u:/g/personal/z5571022_ad_unsw_edu_au/IQDrlt6CZ0h-RoNWbmepSs2KAX5DMChV9hHSwIPqsSUg1F0?e=i0yfXh |
| BERT-clean (owner: z5571377) | ~430 MB | https://unsw-my.sharepoint.com/:u:/g/personal/z5571377_ad_unsw_edu_au/IQDPXBSNdMrJTbXlPOgxuSWTAaRJYj-bTu3seYprf82PLc4?e=68GNUp |
| BERT-aug+weighted (owner: z5564055) | ~430 MB | https://unsw-my.sharepoint.com/:u:/g/personal/z5564055_ad_unsw_edu_au/IQAjhNA5_BBESKj8kolTazQZAZCxRi6AfZ8J1OJaPQ3UUYc?e=oXFxJC |

> If any link has expired, ask the listed owner to re-share as
> "Anyone with the link can view".

---

## Expected contents per model folder (after unzipping)

| File | Size |
|---|---:|
| `config.json` | ~1 KB |
| `model.safetensors` (or `pytorch_model.bin`) | 250 – 430 MB |
| `tokenizer.json` / `tokenizer_config.json` | ~2 MB |
| `vocab.*`, `merges.txt`, `special_tokens_map.json` | ~1 MB combined |
| `training_args.bin` | ~6 KB |

All checkpoints have `id2label = {"0": "non-toxic", "1": "toxic"}` baked
into `config.json`, so HuggingFace `pipeline("text-classification", ...)`
returns those strings directly with no `LABEL_0` / `LABEL_1` translation
needed.

## Smoke test after downloading any transformer

From inside `../../CODE/`:

```bash
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
           m = AutoModelForSequenceClassification.from_pretrained('../MISC/saved_model/roberta_base'); \
           t = AutoTokenizer.from_pretrained('../MISC/saved_model/roberta_base'); \
           print(m.num_labels, m.config.id2label)"
# Expected: 2 {0: 'non-toxic', 1: 'toxic'}
```

Then run `python evaluate_all.py` to reproduce the Part D quantitative
numbers in the report.
