# BERT-aug+weighted (ablation) — fine-tuned weights

**Size:** ~430 MB · **Owner:** Yinghao Wang (z5564055)

> Download: https://unsw-my.sharepoint.com/:u:/g/personal/z5564055_ad_unsw_edu_au/IQAjhNA5_BBESKj8kolTazQZAZCxRi6AfZ8J1OJaPQ3UUYc?e=oXFxJC
>
> The zip unpacks as a folder named `bert_augmented_weighted_model/`.
> Copy the **contents** of that folder (not the folder itself) into
> `MISC/saved_model/bert_aug/`.
> The zip unpacks as a folder named `bert_augmented_weighted_model/`.
> Copy the **contents** of that folder (not the folder itself) into
> `MISC/saved_model/bert_aug/`.

This is the **ablation variant** of BERT — same hyperparameters as
BERT-clean but with training augmented by 174 of the 232 rows from
`manual_annotation_200.csv`. The remaining 58 manual rows were held out
for evaluation in the original training notebook.

**Kept for ablation transparency only** — the primary BERT number in the
main results table comes from `bert_clean/`. See the top-level README §4
for why the two variants are not directly comparable on the manual set.

After downloading, unzip so the contents land **directly inside this
folder**:

```
MISC/saved_model/bert_aug/
├── README.md                 (this file)
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```
