from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Path setup
# =========================
# Assume this script is run inside the CODE/ directory
PROJECT_ROOT = Path.cwd()
MODEL_DIR = PROJECT_ROOT.parent / "MISC" / "saved_model" / "bert_augmented_weighted_model"

# =========================
# Load tokenizer and model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================
# Prediction function
# =========================
def predict_toxicity(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    non_toxic_prob = float(probs[0])
    toxic_prob = float(probs[1])
    pred_label = "Toxic" if toxic_prob > non_toxic_prob else "Non-toxic"

    return pred_label, non_toxic_prob, toxic_prob

# =========================
# Main CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Predict whether a comment is toxic or non-toxic using the fine-tuned BERT model."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input comment text for prediction."
    )

    args = parser.parse_args()

    if not args.text:
        print("Please provide input text using: python predict.py --text \"your comment here\"")
        return

    label, non_toxic_prob, toxic_prob = predict_toxicity(args.text)

    print("Input text:", args.text)
    print("Predicted label:", label)
    print(f"Non-toxic probability: {non_toxic_prob:.4f}")
    print(f"Toxic probability: {toxic_prob:.4f}")

if __name__ == "__main__":
    main()