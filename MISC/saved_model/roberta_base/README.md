# RoBERTa-base — fine-tuned weights

**Size:** ~430 MB · **Owner:** Hanyu Wang (z5571377)

> Download: https://unsw-my.sharepoint.com/:u:/g/personal/z5571377_ad_unsw_edu_au/IQDESbOAmlH-SZriP3aOIpnkAY0r-48ZiMNcH_o3rxAbqbs?e=y1ZCuq

After downloading, unzip so the contents land **directly inside this
folder** (not inside another `model/` subfolder):

```
MISC/saved_model/roberta_base/
├── README.md                 (this file)
├── config.json
├── model.safetensors
├── tokenizer.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
└── tokenizer_config.json
```

This is the path every script in `../../../CODE/` looks for by default
(`../MISC/saved_model/roberta_base/`). To override, pass
`--model-dir /some/other/path` to `cli.py`, `app.py`, or
`evaluate_all.py`.
