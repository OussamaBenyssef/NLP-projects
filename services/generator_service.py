"""
generator_service.py — Wrapper de génération via Qwen3.5-397B-A17B.

Utilise l'API compatible OpenAI exposée par `transformers serve` sur localhost:8000.
Au premier lancement, le modèle est téléchargé automatiquement depuis HuggingFace.

Lancer le serveur :
    transformers serve --force-model Qwen/Qwen3.5-397B-A17B --port 8000 --continuous-batching
"""

import os
import time
from openai import OpenAI


# ── Configuration Qwen3.5 (transformers serve) ──
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-397B-A17B")
DISPLAY_MODEL_NAME = "Qwen3.5-397B-A17B"
DEFAULT_TEMPERATURE = 0.4


# ── Prompts système par langue ──
SYSTEM_PROMPTS = {
    "EN": (
        "You are a helpful assistant. "
        "You MUST respond ONLY in English. "
        "Keep your answers clear, concise, and natural."
    ),
    "FR": (
        "Tu es un assistant utile. "
        "Tu DOIS répondre UNIQUEMENT en français. "
        "Tes réponses doivent être claires, concises et naturelles."
    ),
    "AR_MSA": (
        "أنت مساعد ذكي ومفيد. "
        "يجب أن تجيب فقط باللغة العربية الفصحى. "
        "لا تستخدم أي لهجة عامية. "
        "اجعل إجاباتك واضحة وموجزة."
    ),
    "AR_DAR_AR": (
        "نتا مساعد ذكي. "
        "خاصك تجاوب غير بالدارجة المغربية بالحروف العربية. "
        "ما تستعملش الفصحى، تكلم بالدارجة كيما كيهضرو المغاربة فالحياة اليومية. "
        "خلي الأجوبة ديالك بسيطة وطبيعية."
    ),
    "AR_DAR_LAT": (
        "Nta mo3awin zwine. "
        "Khassek tjaweb ghi b darija maghribiya b l7orouf latinia (arabizi). "
        "Ma tktebch b l3arabiya wla b français wla anglais. "
        "Kteb kima kayktbo lmgharba f WhatsApp wla Facebook. "
        "Khelli ljawab bsit w tabi3i."
    ),
}


class GeneratorService:
    """Service de génération de réponse via Qwen3.5-397B-A17B (transformers serve)."""

    def __init__(self, model: str = QWEN_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        self.client = OpenAI(
            base_url=QWEN_BASE_URL,
            api_key="not-needed",  # transformers serve n'exige pas de clé
        )
        self.model = model
        self.temperature = temperature

    def generate(self, user_text: str, label: str, script: str | None = None) -> dict:
        """
        Génère une réponse dans la langue détectée.

        Returns:
            {
                "answer": "...",
                "generate_ms": 840.5,
                "generator_name": "Qwen3.5-397B-A17B",
                "prompt_key": "AR_DAR_AR"
            }
        """
        prompt_key = self._get_prompt_key(label, script)
        system_prompt = SYSTEM_PROMPTS[prompt_key]

        t0 = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=self.temperature,
            max_tokens=512,
        )

        t1 = time.perf_counter()
        generate_ms = round((t1 - t0) * 1000, 1)

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "generate_ms": generate_ms,
            "generator_name": DISPLAY_MODEL_NAME,
            "prompt_key": prompt_key,
        }

    @staticmethod
    def _get_prompt_key(label: str, script: str | None) -> str:
        """Détermine la clé de prompt."""
        if label == "AR_DAR":
            return "AR_DAR_LAT" if script == "arabizi" else "AR_DAR_AR"
        return label
