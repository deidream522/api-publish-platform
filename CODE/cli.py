"""Unified command-line tester for all four models.

COMP6713 bravo11234 - Part D (5 credits).

Examples
--------
    python cli.py --text "you are an idiot"              # default model: roberta
    python cli.py --model bert-clean --text "..."
    python cli.py --model distilbert --file sample_inputs.txt
    python cli.py --model naive-bayes --text "..."
    python cli.py --file sample_inputs.txt --csv-out out.csv

Accepted --model values: roberta (default), bert-clean, bert-aug,
distilbert, naive-bayes.

If --text and --file are both omitted, the script prompts interactively.

Model weights are resolved from ../MISC/saved_model/<model>/ by default; see
`../MISC/saved_model/README.md` for the per-model download links. Override
the weight location with --model-dir.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


MODELS = {
    "roberta":     {"type": "hf",  "dir": "roberta_base"},
    "bert-clean":  {"type": "hf",  "dir": "bert_clean"},
    "bert-aug":    {"type": "hf",  "dir": "bert_aug"},
    "distilbert":  {"type": "hf",  "dir": "distilbert"},
    "naive-bayes": {"type": "nb",  "dir": "naive_bayes"},
}
DEFAULT_MODEL = "roberta"


# ---------------------------------------------------------------------------
# Weight-dir resolution
# ---------------------------------------------------------------------------
def resolve_model_dir(model_id: str, override: Path | None = None) -> Path:
    if override is not None:
        return override
    here = Path(__file__).resolve().parent
    return here.parent / "MISC" / "saved_model" / MODELS[model_id]["dir"]


# ---------------------------------------------------------------------------
# HuggingFace transformer backend (roberta / bert-* / distilbert)
# ---------------------------------------------------------------------------
def build_hf_predictor(model_dir: Path, max_length: int = 128, batch_size: int = 32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (model_dir / "config.json").exists():
        sys.exit(
            f"[error] no config.json in {model_dir}\n"
            f"        Download the weights from the OneDrive link in\n"
            f"        ../MISC/saved_model/README.md and extract so that\n"
            f"        config.json + model.safetensors sit directly in {model_dir}."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    mdl.eval()

    import inspect
    allowed_kwargs = set(inspect.signature(mdl.forward).parameters)

    @torch.no_grad()
    def predict(texts):
        texts = list(texts)
        all_probs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items() if k in allowed_kwargs}
            logits = mdl(**enc).logits
            all_probs.extend(logits.softmax(dim=-1)[:, 1].cpu().numpy().tolist())
            del enc, logits
        return all_probs

    return predict


# ---------------------------------------------------------------------------
# Naive Bayes backend (joblib-pickled TF-IDF + MultinomialNB)
# ---------------------------------------------------------------------------
def build_nb_predictor(model_dir: Path):
    import warnings

    import joblib

    nb_path = model_dir / "naive_bayes_model.pkl"
    tfidf_path = model_dir / "tfidf_vectorizer.pkl"
    if not nb_path.exists() or not tfidf_path.exists():
        sys.exit(
            f"[error] Naive Bayes weights missing at {model_dir}\n"
            f"        Expected: naive_bayes_model.pkl + tfidf_vectorizer.pkl"
        )
    # The pickles were trained with scikit-learn 1.3.2; silencing the
    # cross-version load warning is safe here because the estimator API we
    # use (`predict_proba` on TfidfVectorizer + MultinomialNB) is stable.
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:  # pragma: no cover - older sklearn
        InconsistentVersionWarning = UserWarning  # type: ignore
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        nb = joblib.load(nb_path)
        tfidf = joblib.load(tfidf_path)

    # Re-use the exact deep-cleaning the training data was put through.
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here / "naive_bayes"))
    try:
        from data_pipeline import clean_text  # type: ignore
    except Exception:
        def clean_text(text, is_baseline=True):
            return (text or "").lower()

    def predict(texts):
        cleaned = [clean_text(t, is_baseline=True) for t in texts]
        X = tfidf.transform(cleaned)
        probs = nb.predict_proba(X)[:, 1]
        return probs.tolist()

    return predict


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def build_predictor(model_id: str, model_dir: Path, max_length: int = 128):
    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id!r} (expected one of {list(MODELS)})")
    kind = MODELS[model_id]["type"]
    if kind == "hf":
        return build_hf_predictor(model_dir, max_length=max_length)
    if kind == "nb":
        return build_nb_predictor(model_dir)
    raise RuntimeError(f"unhandled backend {kind!r}")


def classify(predict_fn, texts, threshold: float = 0.5):
    probs = predict_fn(texts)
    out = []
    for text, p in zip(texts, probs):
        out.append({
            "text": text,
            "prob_toxic": float(p),
            "label": "toxic" if p >= threshold else "non-toxic",
        })
    return out


def _read_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.strip():
                yield line


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Unified toxic comment classifier CLI (Part D, all four models)."
    )
    ap.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL,
                    help=f"Model to use (default: {DEFAULT_MODEL}).")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--text", type=str, help="A single comment to classify.")
    grp.add_argument("--file", type=Path, help="Plain text file, one comment per line.")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="Override weight directory (default: ../MISC/saved_model/<model>/).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Toxic probability cutoff (default: 0.5).")
    ap.add_argument("--max-length", type=int, default=128,
                    help="Transformer tokenizer max_length (default: 128).")
    ap.add_argument("--csv-out", type=Path, help="Optional path to dump predictions as CSV.")
    args = ap.parse_args(argv)

    if args.text is None and args.file is None:
        try:
            text = input("Enter a comment > ").strip()
        except EOFError:
            text = ""
        if not text:
            ap.error("either --text, --file, or an interactive comment is required")
        texts = [text]
    elif args.text is not None:
        texts = [args.text]
    else:
        texts = list(_read_lines(args.file))
        if not texts:
            ap.error(f"{args.file} is empty")

    model_dir = resolve_model_dir(args.model, override=args.model_dir)
    print(f"[info] model = {args.model}   weights = {model_dir}")
    predict_fn = build_predictor(args.model, model_dir, max_length=args.max_length)

    results = classify(predict_fn, texts, threshold=args.threshold)
    for r in results:
        tag = "[TOXIC]   " if r["label"] == "toxic" else "[non-tox] "
        preview = r["text"] if len(r["text"]) <= 120 else r["text"][:117] + "..."
        print(f"{tag} p={r['prob_toxic']:.3f} | {preview}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["text", "label", "prob_toxic"])
            w.writeheader()
            w.writerows(results)
        print(f"\n[saved] predictions -> {args.csv_out}")


if __name__ == "__main__":
    main()
