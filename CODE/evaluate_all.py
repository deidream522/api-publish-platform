"""Universal quantitative evaluator — runs any or all of the four models.

COMP6713 bravo11234 - Part D (10 credits) / quantitative evaluation.

Produces a SINGLE CSV in the team report format:

    Model,Dataset,Accuracy,Precision,Recall,F1-score,F1-toxic,F1-macro

`Precision`, `Recall`, `F1-score` / `F1-toxic` are all computed for the
toxic class (positive = 1). `F1-macro` is the unweighted average over both
classes.  `Accuracy` is overall accuracy.

Usage
-----
    python evaluate_all.py                            # all available models, all splits
    python evaluate_all.py --models roberta,bert-clean
    python evaluate_all.py --splits val,test_6w       # skip manual_200
    python evaluate_all.py --out custom.csv

Models whose weights are not installed in ../MISC/saved_model/<model>/ are
skipped with a warning (never an error) so the script still produces
partial comparison CSVs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from cli import MODELS, resolve_model_dir


SPLIT_FILES = {
    "val":        "val.csv",
    "test_6w":    "test.csv",
    "manual_200": "manual_annotation_200.csv",
}
DEFAULT_SPLITS = list(SPLIT_FILES.keys())

MODEL_DISPLAY = {
    "roberta":     "RoBERTa-base (weighted)",
    "bert-clean":  "BERT-clean",
    "bert-aug":    "BERT-aug+weighted",
    "distilbert":  "DistilBERT",
    "naive-bayes": "Naive Bayes + TF-IDF",
}

_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]


def _load_split(csv_path: Path) -> pd.DataFrame:
    last_err = None
    df = None
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
    if df is None:
        raise RuntimeError(f"cannot decode {csv_path}: {last_err}")
    text_col = (
        "cleaned_text" if "cleaned_text" in df.columns
        else "comment" if "comment" in df.columns
        else df.columns[0]
    )
    label_col = (
        "toxic" if "toxic" in df.columns
        else "label" if "label" in df.columns
        else df.columns[-1]
    )
    df = df[[text_col, label_col]].dropna().copy()
    df.columns = ["text", "toxic"]
    df["text"] = df["text"].astype(str)
    df["toxic"] = df["toxic"].astype(int)
    return df


def _metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    out = {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(p[0]), 4),
        "Recall": round(float(r[0]), 4),
        "F1-score": round(float(f1[0]), 4),
        "F1-toxic": round(float(f1[0]), 4),
        "F1-macro": round(float(f1_macro), 4),
    }
    if y_prob is not None and len(set(np.asarray(y_true).tolist())) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
            out["ROC-AUC"] = round(float(auc), 4)
        except ValueError:
            out["ROC-AUC"] = None
    else:
        out["ROC-AUC"] = None
    return out


def _predict_probs(predict_fn, texts):
    return np.asarray(predict_fn(texts), dtype=float)


def _available_models(wanted: list[str]) -> dict[str, Path]:
    from cli import MODELS as _M
    avail: dict[str, Path] = {}
    for k in wanted:
        if k not in _M:
            print(f"[warn ] unknown model {k!r}, skipping")
            continue
        d = resolve_model_dir(k)
        has_hf = (d / "config.json").exists()
        has_nb = (k == "naive-bayes" and (d / "naive_bayes_model.pkl").exists())
        if has_hf or has_nb:
            avail[k] = d
        else:
            print(f"[warn ] {k}: weights not found at {d}, skipping")
    return avail


def main(argv=None):
    ap = argparse.ArgumentParser(description="Part D quantitative evaluation across all four models.")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="Comma-separated subset (default: all five).")
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                    help="Comma-separated subset of val,test_6w,manual_200.")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Default: ../MISC/data/")
    ap.add_argument("--manual-path", type=Path, default=None,
                    help="Override manual_annotation_200.csv location.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Toxic-probability cutoff (default 0.5).")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: ../MISC/outputs/comparison/partD_quantitative.csv")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    data_dir = args.data_dir or (here.parent / "MISC" / "data")
    manual_path = args.manual_path or (data_dir / "manual_annotation_200.csv")
    out_path = args.out or (here.parent / "MISC" / "outputs" / "comparison" / "partD_quantitative.csv")

    wanted_models = [m.strip() for m in args.models.split(",") if m.strip()]
    wanted_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    avail = _available_models(wanted_models)
    if not avail:
        sys.exit("[error] no available models; install at least one set of weights "
                 "under ../MISC/saved_model/<model>/")

    # Load splits once.
    splits_to_run = {}
    for s in wanted_splits:
        if s not in SPLIT_FILES:
            print(f"[warn ] unknown split {s!r}, skipping")
            continue
        p = manual_path if s == "manual_200" else data_dir / SPLIT_FILES[s]
        if not p.exists():
            print(f"[warn ] {s}: file not found at {p}, skipping")
            continue
        try:
            df = _load_split(p)
            splits_to_run[s] = df
            print(f"[info] loaded {s}: n={len(df)}, toxic={df['toxic'].mean():.1%}")
        except Exception as e:
            print(f"[error] {s}: {e}")

    if not splits_to_run:
        sys.exit("[error] no splits loaded")

    from cli import build_predictor  # lazy import

    rows = []
    for mid, mdir in avail.items():
        print(f"\n[eval ] {mid} ({mdir})")
        try:
            predict_fn = build_predictor(mid, mdir, max_length=args.max_length)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[error] {mid}: failed to load: {e}")
            continue
        for sname, df in splits_to_run.items():
            try:
                y_prob = _predict_probs(predict_fn, df["text"].tolist())
                y_pred = (y_prob >= args.threshold).astype(int)
            except Exception as e:
                print(f"[error] {mid} on {sname}: {e}")
                continue
            m = _metrics(df["toxic"].values, y_pred, y_prob)
            row = {"Model": MODEL_DISPLAY.get(mid, mid), "Dataset": sname, "n": len(df), **m}
            rows.append(row)
            auc_str = f"{m['ROC-AUC']:.4f}" if m.get('ROC-AUC') is not None else "  n/a "
            print(f"       {sname:10s}  acc={m['Accuracy']:.4f}  "
                  f"P={m['Precision']:.4f}  R={m['Recall']:.4f}  "
                  f"F1-tox={m['F1-toxic']:.4f}  F1-macro={m['F1-macro']:.4f}  "
                  f"AUC={auc_str}")
            _flush(rows, out_path)
        del predict_fn
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    out_df = pd.DataFrame(rows, columns=["Model", "Dataset", "n", "Accuracy",
                                         "Precision", "Recall", "F1-score",
                                         "F1-toxic", "F1-macro", "ROC-AUC"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 86)
    print(out_df.to_string(index=False))
    print("=" * 86)
    print(f"[saved] -> {out_path}")


def _flush(rows, out_path: Path):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
