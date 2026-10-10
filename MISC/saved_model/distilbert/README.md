# DistilBERT — fine-tuned weights

**Size:** ~250 MB · **Owner:** Kanglong Teng (z5571022)

> Download: https://unsw-my.sharepoint.com/:u:/g/personal/z5571022_ad_unsw_edu_au/IQDrlt6CZ0h-RoNWbmepSs2KAX5DMChV9hHSwIPqsSUg1F0?e=i0yfXh

After downloading (the archive is named `best_model.zip`), unzip so the
contents land **directly inside this folder**:

```
MISC/saved_model/distilbert/
├── README.md                 (this file)
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

Note: the archive's inner folder is called `best_model/` — when
unzipping, strip that top-level folder so the files live flat in
`MISC/saved_model/distilbert/`. On Windows PowerShell:

```powershell
Expand-Archive best_model.zip -DestinationPath .
Move-Item .\best_model\* .\
Remove-Item .\best_model
```

This is the path `cli.py`, `app.py`, and `evaluate_all.py` look for by
default. To override, pass `--model-dir /some/other/path`.
