"""Unified Gradio demo across all four models.

COMP6713 bravo11234 - Part D (10 credits).

Usage
-----
    python app.py                  # launches on http://127.0.0.1:7860
    python app.py --share          # also create a public Gradio link
    python app.py --only roberta   # restrict the dropdown to one model

The page exposes a model dropdown (only models whose weights are actually
installed are listed), a textbox, a threshold slider, a probability gauge
and five one-click example sentences.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from cli import MODELS, build_predictor, classify, resolve_model_dir

DEFAULT_THRESHOLD = 0.5

EXAMPLES = [
    ["I love this community, thanks for the kind response!", 0.5],
    ["you are an absolute idiot and should shut up", 0.5],
    ["This stupid war, stupid tariffs, stupid everything.", 0.5],
    ["I genuinely believe this was written in crayon.", 0.5],
    ["Not bad, actually pretty interesting take.", 0.5],
]

DESCRIPTION = """# Toxic comment classifier — bravo11234 (COMP6713)

Four models share this demo: two **transformers** (RoBERTa, DistilBERT) and
two **BERT** variants (clean + aug ablation), plus a **Naive Bayes + TF-IDF**
baseline. Pick a model from the dropdown, paste or type a comment on the
left, and click **Classify**.

The model returns the probability that the comment is toxic; anything at or
above the threshold is flagged as toxic.
"""


def _detect_available_models(only: list[str] | None = None):
    keys = list(MODELS) if not only else [k for k in only if k in MODELS]
    available = {}
    for k in keys:
        weight_dir = resolve_model_dir(k)
        if (weight_dir / "config.json").exists() or (
            k == "naive-bayes"
            and (weight_dir / "naive_bayes_model.pkl").exists()
        ):
            available[k] = weight_dir
    return available


def build_ui(available: dict, max_length: int = 128):
    # lazily instantiate predictors on first use so startup is fast.
    cached: dict = {}

    def _get(model_id: str):
        if model_id not in cached:
            cached[model_id] = build_predictor(
                model_id, available[model_id], max_length=max_length
            )
        return cached[model_id]

    def predict_fn(model_id: str, text: str, threshold: float):
        text = (text or "").strip()
        if not text:
            return {"toxic": 0.0, "non-toxic": 0.0}, "Please enter a comment."
        r = classify(_get(model_id), [text], threshold=threshold)[0]
        label_scores = {"toxic": r["prob_toxic"], "non-toxic": 1.0 - r["prob_toxic"]}
        verdict = (
            f"**{r['label'].upper()}**  ·  p_toxic = {r['prob_toxic']:.3f}  ·  "
            f"threshold = {threshold:.2f}  ·  model = `{model_id}`"
        )
        return label_scores, verdict

    with gr.Blocks(title="bravo11234 · Toxic Classifier") as demo:
        gr.Markdown(DESCRIPTION)

        if not available:
            gr.Markdown(
                "### No model weights found\n"
                "Download at least one model's weights into\n"
                "`../MISC/saved_model/<model>/` — see `../MISC/saved_model/README.md`."
            )
            return demo

        with gr.Row():
            with gr.Column(scale=3):
                model_pick = gr.Dropdown(
                    choices=list(available.keys()),
                    value=list(available.keys())[0],
                    label="Model",
                )
                inp = gr.Textbox(
                    label="Comment",
                    placeholder="Type or paste a comment here...",
                    lines=4,
                )
                thr = gr.Slider(
                    minimum=0.1, maximum=0.9, step=0.05,
                    value=DEFAULT_THRESHOLD,
                    label="Toxic threshold",
                    info="Raise for stricter precision, lower for higher recall.",
                )
                btn = gr.Button("Classify", variant="primary")
            with gr.Column(scale=2):
                label_out = gr.Label(label="Probability", num_top_classes=2)
                verdict_out = gr.Markdown()

        gr.Examples(examples=EXAMPLES, inputs=[inp, thr])

        btn.click(fn=predict_fn, inputs=[model_pick, inp, thr], outputs=[label_out, verdict_out])
        inp.submit(fn=predict_fn, inputs=[model_pick, inp, thr], outputs=[label_out, verdict_out])

    return demo


def main(argv=None):
    ap = argparse.ArgumentParser(description="Unified Gradio demo for all four toxic classifiers.")
    ap.add_argument("--only", nargs="*", help="Restrict dropdown to these model ids.")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    ap.add_argument("--server-name", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=7860)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args(argv)

    available = _detect_available_models(only=args.only)
    if not available:
        print("[warn ] no model weights detected under ../MISC/saved_model/<model>/")
        print("        The UI will still launch but classification will be disabled.")
    else:
        print("[info] available models: " + ", ".join(available.keys()))

    demo = build_ui(available, max_length=args.max_length)
    demo.launch(share=args.share,
                server_name=args.server_name,
                server_port=args.server_port)


if __name__ == "__main__":
    main()
