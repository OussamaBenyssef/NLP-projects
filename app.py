"""
app.py — Flask Chat Multilingue avec panneau de détails techniques.

Routes :
    GET  /           → Interface chat
    POST /api/chat   → Détection + Génération → JSON
"""

import os
import sys
import traceback
from flask import Flask, render_template, request, jsonify

# Ajouter le dossier projet au PATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.detector_service import DetectorService
from services.generator_service import GeneratorService


app = Flask(__name__)

# ── Initialisation des services (au démarrage) ──
print("⏳ Chargement du détecteur...")
detector = DetectorService()
print("✓ Détecteur chargé")

generator = GeneratorService()
print("✓ Générateur prêt (Qwen3.5 / Ollama)")


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
        return jsonify({"error": f"Ollama inaccessible : {e}"}), 503

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erreur serveur : {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
