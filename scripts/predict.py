"""
predict.py — Test interactif des modèles SVM et Transformer.

Usage:
    python scripts/predict.py                      # Mode interactif
    python scripts/predict.py "شنو هو le problème"  # Prédiction unique
    python scripts/predict.py --svm-only            # SVM seul (rapide)
    python scripts/predict.py --transformer-only    # Transformer seul
"""
import os
import sys
import argparse
import re
import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.normalizer import normalize
from src.utils import load_config

# Nécessaire pour désérialiser le pipeline SVM (FunctionTransformer référence cette fonction)
def compute_script_ratios(texts):
    """Calcule arab_ratio, latin_ratio, cs_ratio pour chaque texte."""
    arab_pat = re.compile(r'[\u0600-\u06FF]+')
    latin_pat = re.compile(r'[a-zA-Z]+')
    features = []
    for text in texts:
        tokens = str(text).split()
        if not tokens:
            features.append([0.0, 0.0, 0.0])
            continue
        n = len(tokens)
        arab = sum(1 for t in tokens if arab_pat.search(t))
        latin = sum(1 for t in tokens if latin_pat.search(t))
        arab_r = arab / n
        latin_r = latin / n
        cs_r = min(arab_r, latin_r)
        features.append([arab_r, latin_r, cs_r])
    return np.array(features)

# ── Couleurs terminal ──────────────────────────────────────────────────────
COLORS = {
    "AR_DAR": "\033[93m",   # Jaune
    "AR_MSA": "\033[92m",   # Vert
    "FR":     "\033[94m",   # Bleu
    "EN":     "\033[95m",   # Magenta
    "MIX":    "\033[91m",   # Rouge
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def colorize(label):
    c = COLORS.get(label, "")
    return f"{c}{BOLD}{label}{RESET}"


def load_svm(config):
    path = config["paths"]["models"]["baseline_svm"]
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_transformer(config):
    model_path = os.path.join(config["paths"]["models"]["transformer"], "best")
    if not os.path.exists(model_path):
        return None, None, None

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    id2label = model.config.id2label
    return model, tokenizer, id2label


def predict_svm(pipeline, text_norm):
    proba = None
    label = pipeline.predict([text_norm])[0]
    # LinearSVC n'a pas de predict_proba, on utilise decision_function
    try:
        scores = pipeline.decision_function([text_norm])[0]
        classes = pipeline.classes_
        idx = list(classes).index(label)
        confidence = scores[idx] if len(scores) > 1 else abs(scores)
    except Exception:
        confidence = None
    return label, confidence


def predict_transformer(model, tokenizer, id2label, text_norm):
    import torch

    inputs = tokenizer(text_norm, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax().item()
    label = id2label[pred_idx]
    confidence = probs[pred_idx].item()

    # Top-3 prédictions
    top3_idx = probs.argsort(descending=True)[:3]
    top3 = [(id2label[i.item()], probs[i].item()) for i in top3_idx]
    return label, confidence, top3


def predict_text(text, svm_pipeline, transformer_data, config):
    """Prédit la langue d'un texte avec les deux modèles."""
    text_norm = normalize(text)

    print(f"\n{DIM}Texte normalisé : {text_norm}{RESET}")
    print("─" * 60)

    # SVM
    if svm_pipeline:
        label_svm, conf_svm = predict_svm(svm_pipeline, text_norm)
        conf_str = f" (score: {conf_svm:.2f})" if conf_svm is not None else ""
        print(f"  {'SVM':>12}  →  {colorize(label_svm)}{conf_str}")

    # Transformer
    if transformer_data[0] is not None:
        model, tokenizer, id2label = transformer_data
        label_tf, conf_tf, top3 = predict_transformer(model, tokenizer, id2label, text_norm)
        print(f"  {'Transformer':>12}  →  {colorize(label_tf)} ({conf_tf:.1%})")
        top3_str = " | ".join(f"{colorize(l)} {p:.1%}" for l, p in top3)
        print(f"  {'Top-3':>12}  :  {top3_str}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Test interactif des modèles")
    parser.add_argument("text", nargs="?", default=None, help="Texte à classifier")
    parser.add_argument("--svm-only", action="store_true", help="SVM uniquement")
    parser.add_argument("--transformer-only", action="store_true", help="Transformer uniquement")
    args = parser.parse_args()

    config = load_config()

    # Chargement des modèles
    svm_pipeline = None
    transformer_data = (None, None, None)

    if not args.transformer_only:
        print("Chargement du SVM...", end=" ", flush=True)
        svm_pipeline = load_svm(config)
        print("✓" if svm_pipeline else "✗ (introuvable)")

    if not args.svm_only:
        print("Chargement du Transformer...", end=" ", flush=True)
        model, tokenizer, id2label = load_transformer(config)
        transformer_data = (model, tokenizer, id2label)
        print("✓" if model else "✗ (introuvable)")

    if svm_pipeline is None and transformer_data[0] is None:
        print("\n⚠ Aucun modèle disponible. Lancez les étapes 6 et 7 d'abord.")
        return

    print()
    print(f"{BOLD}═══ Test des Modèles — Darija CS Detection ═══{RESET}")
    print(f"  Classes : {' | '.join(colorize(l) for l in ['AR_DAR','AR_MSA','EN','FR','MIX'])}")
    print(f"  Tapez 'q' pour quitter.\n")

    # Mode prédiction unique
    if args.text:
        predict_text(args.text, svm_pipeline, transformer_data, config)
        return

    # Mode interactif
    while True:
        try:
            text = input(f"{BOLD}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not text or text.lower() in ("q", "quit", "exit"):
            print("Au revoir !")
            break

        predict_text(text, svm_pipeline, transformer_data, config)


if __name__ == "__main__":
    main()
