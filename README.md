# COMP6713 Group Project — Toxic Comment Classification

**Team:** bravo11234
**Task:** binary toxic comment classification on Jigsaw + Civil Comments, with
an out-of-distribution (OOD) evaluation on 232 manually-annotated social-media
comments.

## Team members

| Name | zID |
|---|---|
| Yinghao Wang  | z5564055 |
| Runzhuo Bai   | z5547602 |
| Hanyu Wang    | z5571377 |
| Kanglong Teng | z5571022 |
| Longjun Zhao  | z5531354 |

---

## Quick Start for markers

Three commands. All paths are relative to this folder (`bravo11234/`).

```bash
cd bravo11234
pip install -r CODE/requirements.txt
# Download model weights once — see MISC/saved_model/README.md for OneDrive links.
```

Then pick what you want to test:

| Task | Command | Notes |
|---|---|---|
| Single-input CLI (rubric: "input argument") | `python CODE/cli.py --text "you are an idiot"` | prints `[TOXIC]` / `[non-tox]` label |
| Batch CLI (rubric: "input file") | `python CODE/cli.py --file CODE/sample_inputs.txt` | one prediction per line |
| Gradio demo | `python CODE/app.py` | then open <http://127.0.0.1:7860> |

**Windows users without a terminal**: double-click `run_cli.bat` or `run_demo.bat` at the project root.
**macOS / Linux**: `bash run_cli.sh` or `bash run_demo.sh`.

Default model is RoBERTa; switch with `--model {roberta, bert-clean, bert-aug, distilbert, naive-bayes}`.
Paths inside `cli.py` and `app.py` resolve weights via `__file__`, so you can invoke the scripts from any working directory.

---

## 1. What this submission contains

A single zip with **four fine-tuned classifiers** sharing one evaluation
protocol, plus a BERT ablation study:

| # | Model | Type | Training data |
|---|---|---|---|
| 1 | **Naive Bayes + TF-IDF** | Baseline | `train_deepclean.csv` (heavy cleaning) |
| 2 | **DistilBERT** | Transformer | `train.csv` (light cleaning, 60 000 samples) |
| 3 | **BERT-clean** (primary) | Transformer | `train.csv` (60 000 samples) |
| 3b | **BERT-aug+weighted** (ablation) | Transformer | `train.csv` + 174/232 manual rows |
| 4 | **RoBERTa-base** | Transformer | `train.csv` (60 000 samples) |

Models 2, 3, and 4 all use identical evaluation protocol: default threshold
0.5, full 232-sample manual set. Model 3b is kept as an **ablation of training
data** (not as the primary BERT result) because it saw 174 of the 232 manual
samples during training.

---

## 2. Directory layout

```
bravo11234/
├── README.md                     (this file — top entry point)
├── CONTRIBUTION.md               per-member contribution statement (fill in TODOs)
├── REPORT.pdf                    the written report            (to add)
├── PRESENTATION.pdf              the presentation              (to add)
├── CODE/                         runnable code only
│   ├── README.md                 how to train / evaluate / demo each model
│   ├── requirements.txt
│   ├── app.py                    unified Gradio demo (all 4 models)
│   ├── cli.py                    unified command-line tester
│   ├── evaluate_all.py           universal quantitative evaluator
│   ├── sample_inputs.txt
│   ├── roberta_base/             fine-tuning notebook
│   ├── distilbert/               fine-tuning notebook
│   ├── bert/                     BERT-clean + BERT-aug notebooks
│   └── naive_bayes/              notebook + data_pipeline + predict
└── MISC/                         data, outputs, model weights
    ├── README.md
    ├── data/                     train / val / test / manual + deepclean
    ├── outputs/                  one subfolder per model (metrics + figures)
    └── saved_model/              pkl files directly + OneDrive links for big weights
```

---

## 3. Quick start

### 3.1 Install dependencies

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r CODE/requirements.txt
```

### 3.2 Fetch the large model weights

Naive Bayes weights (~26 MB, well under the Moodle 200 MB limit) are already
bundled inside `MISC/saved_model/naive_bayes/`. The four transformer
checkpoints are hosted on OneDrive because each is >200 MB. See
`MISC/saved_model/README.md` for the per-model download links and the
expected extraction paths.

### 3.3 Run the unified quantitative evaluation (Part D, 10 cr.)

```bash
cd CODE
python evaluate_all.py
# -> writes ../MISC/outputs/comparison/partD_quantitative.csv
```

### 3.4 Command-line tester (Part D, 5 cr.)

```bash
python cli.py --model roberta --text "you are an idiot"
python cli.py --model bert-clean --file sample_inputs.txt
python cli.py --model distilbert --threshold 0.7 --text "..."
python cli.py --model naive-bayes --file sample_inputs.txt
```

### 3.5 Gradio demo (Part D, 10 cr.)

```bash
python app.py
# open http://127.0.0.1:7860
# dropdown lets the user switch between the four models live
```

---

## 4. Part D deliverables — what is delivered, and where

The assignment brief awards Part D a minimum of 20 credits split across four
sub-items. This submission delivers all four, for a headline total of **30
credits**:

| # | Part D sub-item | Credits | Where it lives | Verification command |
|---|---|---:|---|---|
| 1 | **Quantitative Evaluation** — Accuracy / Precision / Recall / F1 on the test set | 10 | `CODE/evaluate_all.py` (unified evaluator across 4 models × 3 splits) → `MISC/outputs/comparison/partD_quantitative.csv`; per-model numbers in `MISC/outputs/<model>/*.csv` | `python evaluate_all.py` |
| 2 | **Qualitative Evaluation** — analysis of misclassified instances and common error types | 5 | Per-model error samples in `MISC/outputs/<model>/*_error_samples.csv` (RoBERTa, DistilBERT, BERT) + the written analysis in §Qualitative of `REPORT.pdf` | (manual inspection) |
| 3 | **Command Line Testing** — interface that accepts an input argument *or* an input file | 5 | `CODE/cli.py` with `--text`, `--file`, interactive, `--threshold`, `--csv-out`; five models selectable via `--model` | `python cli.py --model roberta --text "..."` or `--file sample_inputs.txt` |
| 4 | **Demo** — Gradio interface for interactive prediction | 10 | `CODE/app.py` with a model dropdown (all four classifiers), threshold slider, live probability gauge | `python app.py` → open `http://127.0.0.1:7860` |

All four items reuse a **single `MODEL_REGISTRY`** defined in `cli.py`, so
adding or removing a model (for example, swapping BERT-clean for BERT-aug in
the ablation section) updates CLI, demo, and the evaluator simultaneously.

Minimal end-to-end verification (~2 minutes once the transformer weights are
in place):

```bash
cd CODE
# item 3 — CLI, both calling conventions required by the spec
python cli.py --model roberta     --text "you are an idiot"
python cli.py --model naive-bayes --file sample_inputs.txt
# item 1 — unified quantitative sweep
python evaluate_all.py --models roberta,naive-bayes --splits test_6w
# item 4 — Gradio demo
python app.py
```

---

## 5. Headline results

All numbers below are produced by the **unified** `CODE/evaluate_all.py`
script: default threshold 0.5, identical data loader, identical metrics
function, run on a Colab T4 on 2026-04-21 (re-run with AUC enabled). Raw
CSV at `MISC/outputs/comparison/partD_quantitative.csv`. Toxic-class
Precision / Recall / F1 are reported for the minority class (label = 1).

### 5.1 In-distribution (Jigsaw test_6w, n = 6 391)

| Model | Accuracy | Precision-tox | Recall-tox | **F1-tox** | F1-macro | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes + TF-IDF  | 0.9288 | 0.9658 | 0.2386 | 0.3826 | 0.6724 | 0.9301 |
| DistilBERT            | 0.9537 | 0.7245 | 0.8054 | 0.7628 | 0.8686 | 0.9745 |
| BERT-clean            | 0.9598 | 0.7756 | 0.7953 | **0.7853** | 0.8816 | **0.9782** |
| BERT-aug+weighted     | 0.9582 | 0.7547 | 0.8122 | 0.7824 | 0.8796 | 0.9756 |
| RoBERTa-base          | 0.9567 | 0.7336 | 0.8342 | 0.7807 | 0.8783 | 0.9758 |

### 5.2 Out-of-distribution (manual_annotation_200, n = 232)

| Model | Accuracy | Precision-tox | Recall-tox | **F1-tox** | F1-macro | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes + TF-IDF  | 0.6164 | 0.4000 | 0.0227 | 0.0430 | 0.4016 | 0.6674 |
| DistilBERT            | 0.6897 | 0.6600 | 0.3750 | 0.4783 | 0.6287 | 0.6870 |
| BERT-clean            | 0.7069 | 0.7273 | 0.3636 | 0.4848 | 0.6400 | 0.6788 |
| RoBERTa-base          | 0.6983 | 0.6607 | 0.4205 | **0.5139** | 0.6476 | 0.6880 |
| BERT-aug+weighted\*   | 0.7931 | 0.8125 | 0.5909 | *0.6842\** | 0.7652 | **0.8326** |

\* **BERT-aug saw 174 of these 232 samples during training** (see §5.3
below). Its OOD number is therefore not a fair generalisation estimate;
it is kept in the same table only so the pattern is visible at a glance.

### 5.3 Ablation: effect of training on manual data (BERT only)

Both BERT variants share hyperparameters (learning rate, batch size,
max length, epochs, class-weighted cross-entropy `[0.5515, 5.3506]`).
The **only** difference is whether `manual_annotation_200.csv` rows were
mixed into the training set.

| BERT variant | Test F1-tox | Manual F1-tox (n = 232) | Manual set usage |
|---|---:|---:|---|
| BERT-clean (primary result)     | 0.7853 | 0.4848 | evaluation only — **no manual rows seen in training** |
| BERT-aug+weighted (ablation)    | 0.7824 | **0.6842** | 174 / 232 manual rows **mixed into training**; same 174 then re-evaluated together with the 58 held-out in this row |

> The +0.20 OOD F1-tox gap between the two variants is **partly a
> genuine generalisation gain from target-domain fine-tuning, and
> partly evaluation leakage**: the 174 training rows are counted again
> inside the 232-row evaluation, so the model is partially being scored
> on what it memorised. The teammate notebook originally reported
> **0.7368 on the 58 held-out rows only**; even that smaller, leakage-free
> estimate beats BERT-clean's 0.4848, which is the informative ablation
> signal — target-domain fine-tuning helps, but the raw +0.20 should not
> be taken at face value. A second, fully fresh manual batch would be
> required for an apples-to-apples comparison (discussed in the report).

### 5.4 Take-aways

1. **All transformers dramatically beat the NB baseline on F1-tox** — the
   gap on the Jigsaw test set is ~0.40 absolute F1. BoW simply cannot
   capture the contextual cues needed for implicit toxicity, and NB
   suffers especially on recall (0.24 on test, 0.02 on OOD).
2. **BERT-clean edges RoBERTa on the Jigsaw in-distribution test set**
   (+0.005 F1-tox), but **RoBERTa wins the OOD manual set** (+0.03
   F1-tox over BERT-clean). Practical implication: the "best" model
   depends on where your deployment distribution actually is.
3. **~60 % of manual-set toxic comments are missed by every in-domain
   transformer trained only on Jigsaw** (recall-tox ~0.36–0.42). The
   fine-tuning objective and the training distribution both
   under-represent emoji-laden, all-caps, abbreviated social-media
   toxicity.
4. **Adding 174 manually-labelled target-domain rows boosts OOD recall
   dramatically** (ablation §5.3) — target-domain fine-tuning is the
   right lever to pull, _provided_ held-out evaluation really is held
   out from training.

### 5.5 ROC-AUC note

- ROC-AUC is now computed directly in `CODE/evaluate_all.py` from model
  probabilities (`roc_auc_score`), independent of threshold.
- Values above come from the same unified run as Accuracy/Precision/Recall/F1
  and are stored in `MISC/outputs/comparison/partD_quantitative.csv`.

---

## 6. Reproducibility

- Python 3.10+
- Single seed = 42 (numpy / torch / HF `set_seed`) for all transformer runs.
- All transformer models fine-tuned on a single T4 GPU (Colab), 2–3 epochs,
  batch 16, learning rate 2e-5, `max_length = 128`, fp16 where supported.
- Class-weighted cross-entropy with
  weights = `[0.551, 5.407]` computed via `sklearn.compute_class_weight`
  on `train.csv`.
- Per-notebook reproduction steps are in `CODE/README.md`.
