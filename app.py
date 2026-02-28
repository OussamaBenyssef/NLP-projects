"""
app.py — Flask Chat Multilingue avec panneau de détails techniques.

Routes :
    GET  /           → Interface chat
    POST /api/chat   → Détection + Génération → JSON

Prérequis : Le serveur Qwen3.5 doit tourner AVANT de lancer cette app.
"""

import os
import sys
import traceback
import urllib.request
import urllib.error
from flask import Flask, render_template, request, jsonify

# Ajouter le dossier projet au PATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.detector_service import DetectorService
from services.generator_service import GeneratorService


app = Flask(__name__)

# ── Vérification du serveur Qwen3.5 ──
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
QWEN_HEALTH_URL = QWEN_BASE_URL.replace("/v1", "/health")  # essai /health
QWEN_MODELS_URL = f"{QWEN_BASE_URL}/models"                 # fallback /v1/models


def check_qwen_server():
    """Vérifie que le serveur de génération Qwen3.5 est accessible."""
    for url in [QWEN_MODELS_URL, QWEN_HEALTH_URL]:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            continue
    return False


print("⏳ Vérification du serveur Qwen3.5...")
if not check_qwen_server():
    print("\n" + "=" * 65)
    print("  ❌  ERREUR : Le serveur Qwen3.5 n'est pas accessible !")
    print("=" * 65)
    print()
    print("  Lancez d'abord le serveur de génération dans un autre terminal :")
    print()
    print("    transformers serve \\")
    print("        --force-model Qwen/Qwen3.5-397B-A17B \\")
    print("        --port 8000 \\")
    print("        --continuous-batching")
    print()
    print(f"  URL attendue : {QWEN_BASE_URL}")
    print("  (configurable via la variable QWEN_BASE_URL)")
    print("=" * 65 + "\n")
    sys.exit(1)
print("✓ Serveur Qwen3.5 accessible")

# ── Initialisation des services ──
print("⏳ Chargement du détecteur...")
detector = DetectorService()
print("✓ Détecteur chargé")

generator = GeneratorService()
print("✓ Générateur prêt (Qwen3.5)")


@app.route("/")
def index():
    """Page principale du chat."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint principal : détecte la langue puis génère une réponse.

    Input:  {"text": "..."}
    Output: {
        "answer": "...",
        "label": "FR",
        "scores": {"FR": 0.88, "EN": 0.06, "AR_MSA": 0.04, "AR_DAR": 0.02},
        "timings": {"detect_ms": 12, "generate_ms": 840},
        "model": {"detector": "transformer_final_v2", "generator": "qwen2.5:7b-instruct"},
        "script": "arabic" | "arabizi" | null
    }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Le champ 'text' est requis."}), 400

    user_text = data["text"].strip()

    try:
        # 1. Détection
        detection = detector.detect(user_text)

        # 2. Génération
        generation = generator.generate(
            user_text=user_text,
            label=detection["label"],
            script=detection["script"],
        )

        # 3. Réponse JSON standardisée
        return jsonify({
            "answer": generation["answer"],
            "label": detection["label"],
            "scores": detection["scores"],
            "confidence": detection["confidence"],
            "timings": {
                "detect_ms": detection["detect_ms"],
                "generate_ms": generation["generate_ms"],
            },
            "model": {
                "detector": detection["detector_name"],
                "generator": generation["generator_name"],
            },
            "script": detection["script"],
            "prompt_key": generation["prompt_key"],
        })

    except ConnectionError as e:
        return jsonify({"error": f"Serveur Qwen3.5 inaccessible : {e}"}), 503

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erreur serveur : {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
