"""
detector_service.py — Wrapper autour du détecteur de langue existant.

Expose les scores (probabilités) par classe et le temps d'inférence.
"""

import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from src.normalizer import normalize


PAT_ARABIC = re.compile(r'[\u0600-\u06FF]')
PAT_LATIN = re.compile(r'[a-zA-Z]')
ARABIZI_DIGITS = re.compile(r'[2378569]')


class DetectorService:
    """Service de détection de langue avec scores détaillés."""

    LABELS = ["AR_DAR", "AR_MSA", "EN", "FR"]

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = os.path.join(_project_root, "models", "transformer_final_v2")

        self.model_name = "XLM-RoBERTa (fine-tuned)"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def detect(self, text: str) -> dict:
        """
        Détecte la langue et retourne les détails complets.

        Returns:
            {
                "label": "FR",
                "confidence": 0.94,
                "scores": {"AR_DAR": 0.02, "AR_MSA": 0.04, "EN": 0.06, "FR": 0.88},
                "script": "arabic" | "arabizi" | null,
                "detect_ms": 12.3,
                "detector_name": "transformer_final_v2"
            }
        """
        text_norm = normalize(text)

        t0 = time.perf_counter()

        inputs = self.tokenizer(
            text_norm, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)

        t1 = time.perf_counter()
        detect_ms = round((t1 - t0) * 1000, 1)

        confidence, pred_id = torch.max(probs, dim=-1)
        label = self.model.config.id2label[pred_id.item()]

        # Scores par classe
        scores = {}
        for idx, prob in enumerate(probs.tolist()):
            class_label = self.model.config.id2label[idx]
            scores[class_label] = round(prob, 4)

        # Script pour AR_DAR
        script = None
        if label == "AR_DAR":
            script = "arabizi" if self._is_arabizi(text) else "arabic"

        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "scores": scores,
            "script": script,
            "detect_ms": detect_ms,
            "detector_name": self.model_name,
        }

    @staticmethod
    def _is_arabizi(text: str) -> bool:
        """Détecte si le texte est en arabizi (darija en lettres latines)."""
        arab_chars = len(PAT_ARABIC.findall(text))
        latin_chars = len(PAT_LATIN.findall(text))
        total = arab_chars + latin_chars
        if total == 0:
            return False
        latin_ratio = latin_chars / total
        if latin_ratio > 0.5:
            return True
        if latin_ratio > 0.3:
            tokens = text.lower().split()
            for t in tokens:
                if ARABIZI_DIGITS.search(t) and PAT_LATIN.search(t):
                    return True
        return False
