# CODE/ — runnable code for all four models

This folder contains every notebook and script needed to **train**,
**evaluate**, **test via CLI**, and **demo via Gradio** each of the four
models in the submission.

```
CODE/
├── README.md                       (this file)
├── requirements.txt
├── app.py                          unified Gradio demo (all 4 models)
├── cli.py                          unified command-line tester
├── evaluate_all.py                 universal Part D quantitative evaluator
├── sample_inputs.txt               example inputs for cli.py --file
├── roberta_base/
│   └── roberta_base_finetune.ipynb
├── distilbert/
│   └── train_distilbert.ipynb
├── bert/
│   ├── bert_clean_finetune.ipynb        primary BERT model (no manual aug)
│   └── bert_augmented_weighted.ipynb    ablation (with 174 manual rows)
└── naive_bayes/
    ├── baseline_model.ipynb
    ├── data_pipeline.py
    └── predict.py
```

---

## 1. Environment setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt

# One-off NLTK corpora (for the Naive Bayes deep-cleaning pipeline)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

Tested with Python 3.10 and PyTorch 2.1 (CPU works for inference; GPU
required for fine-tuning from scratch).

---

## 2. Model weights

Each model expects its weights in a specific location. Naive Bayes is
already bundled; the three transformers must be downloaded from OneDrive
— see `../MISC/saved_model/README.md` for the links and target paths.

| Model | Expected path (default) | Size |
|---|---|---:|
| Naive Bayes | `../MISC/saved_model/naive_bayes/{naive_bayes_model,tfidf_vectorizer}.pkl` | ~26 MB (bundled) |
| DistilBERT | `../MISC/saved_model/distilbert/` | ~250 MB (OneDrive) |
| BERT-clean | `../MISC/saved_model/bert_clean/` | ~430 MB (OneDrive) |
| BERT-aug   | `../MISC/saved_model/bert_aug/` | ~430 MB (OneDrive) |
| RoBERTa-base | `../MISC/saved_model/roberta_base/` | ~430 MB (OneDrive) |

All scripts accept `--model-dir` to override.

---

## 3. Part D deliverables

### 3.1 Quantitative evaluation (10 cr.) — `evaluate_all.py`

Runs all four models on validation / test / manual_200 and emits a single
merged CSV in the report's format:

```bash
python evaluate_all.py
# -> ../MISC/outputs/comparison/partD_quantitative.csv

python evaluate_all.py --models roberta,bert-clean
python evaluate_all.py --threshold 0.7
```

### 3.2 Qualitative evaluation (5 cr.)

Per-model top-10 false-positive / false-negative CSVs written by each
training notebook:

- `../MISC/outputs/roberta_base/roberta_base_error_samples.csv`
- `../MISC/outputs/distilbert/manual_200_errors.csv`
- `../MISC/outputs/bert_clean/` (regenerated when the notebook is re-run)
- `../MISC/outputs/naive_bayes/` (regenerated when the notebook is re-run)

The written error-pattern taxonomy (implicit toxicity, sarcasm,
profanity-without-attack, …) is in the report §Qualitative.

### 3.3 Command-line testing (5 cr.) — `cli.py`

Unified across all four models via `--model`:

```bash
python cli.py --model roberta --text "you are an idiot and should shut up"
# [TOXIC]    p=0.991 | you are an idiot and should shut up

python cli.py --model bert-clean --file sample_inputs.txt
python cli.py --model distilbert --text "..." --threshold 0.7
python cli.py --model naive-bayes --text "..."

# batch + save predictions as CSV
python cli.py --model roberta --file sample_inputs.txt --csv-out cli_predictions.csv

# interactive — prompts "Enter a comment >" when run with no --text / --file
python cli.py --model roberta
```

Accepted model ids: `naive-bayes`, `distilbert`, `bert-clean`, `bert-aug`,
`roberta`.

### 3.4 Gradio demo (10 cr.) — `app.py`

```bash
python app.py                 # http://127.0.0.1:7860
python app.py --share         # also exposes a public link
```

The UI provides a **model dropdown** (so the grader can A/B any two
models live), a toxic-probability gauge, a tunable threshold slider, and
one-click example sentences (including sarcasm / implicit-toxic cases
that all four models are known to miss).

---

## 4. Training each model from scratch

All four notebooks auto-detect Colab vs. local and resolve data paths
relative to `../MISC/data/` locally or
`/content/drive/MyDrive/6713groupwork/MISC/data/` in Colab.

| Model | Notebook | ~Train time (T4) |
|---|---|---:|
| Naive Bayes | `naive_bayes/baseline_model.ipynb` | <1 min (CPU) |
| DistilBERT | `distilbert/train_distilbert.ipynb` | ~32 min |
| BERT-clean | `bert/bert_clean_finetune.ipynb` | ~40 min |
| BERT-aug   | `bert/bert_augmented_weighted.ipynb` | ~40 min |
| RoBERTa-base | `roberta_base/roberta_base_finetune.ipynb` | ~22 min |

Each notebook is configured to save model weights + summary metrics +
qualitative error CSVs to the corresponding `../MISC/outputs/<model>/`
subfolder.

---

## 5. Reproducibility

- Single random seed = 42 across every transformer
  (`set_seed`, `TrainingArguments(seed, data_seed)`).
- Class-weighted cross-entropy
  (`[0.551, 5.407]` computed via `sklearn.compute_class_weight("balanced")`)
  is used for every transformer to handle the 91/9 class imbalance.
- Decision threshold = 0.5 for the main results table; threshold-tuned
  numbers are reported only in the respective notebook for completeness.
- `max_length = 128` covers > 95 % of tokens.
- fp16 enabled on GPU, disabled on CPU.

---

## 6. Notes on the two BERT variants

`bert/bert_clean_finetune.ipynb` is the **primary** BERT model — it trains
purely on `train.csv` (60 k Jigsaw rows) and is evaluated on the full
232-row manual set. `bert/bert_augmented_weighted.ipynb` is the
**ablation** — it augments training with 174 of the 232 manual rows and
evaluates on the remaining 58. The ablation demonstrates how much
in-domain labelled data helps; the primary run is what the main-table
numbers compare against.

See the report §BERT ablation for the full discussion.
