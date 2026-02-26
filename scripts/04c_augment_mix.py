"""
04c_augment_mix.py — Augmentation de données MIX (code-switching).

Génère des exemples synthétiques de MIX en injectant des mots/phrases
FR ou EN dans des textes AR_DAR existants, suivant les patterns observés
dans les vrais exemples MIX :
  - Termes tech/marques en latin dans le flux arabe
  - Fragments de phrases FR insérés (la résidence, le problème, etc.)
  - Mots d'emprunt courants (internet, facebook, youtube, etc.)

Injecte les données augmentées dans train_silver.parquet.
"""
import os
import sys
import random
import re
import uuid
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger, ensure_dir

# Fragments français courants au Maroc (code-switching naturel)
FR_FRAGMENTS = [
    "le problème", "la solution", "en fait", "c'est normal",
    "pas de souci", "la résidence", "le quartier", "la ville",
    "le marché", "le projet", "le travail", "la question",
    "du coup", "par exemple", "en tout cas", "c'est bien",
    "la famille", "le weekend", "le restaurant", "le café",
    "la pharmacie", "l'hôpital", "la gare", "le bus",
    "le programme", "la formation", "le stage", "le bureau",
    "la réunion", "le directeur", "le prix", "le budget",
    "la qualité", "le service", "le client", "le produit",
    "bonne chance", "pas mal", "très bien", "comme même",
    "la licence", "le master", "la faculté", "le diplôme",
    "le loyer", "la location", "l'appartement", "la chambre",
]

# Fragments anglais courants (tech, pop culture)
EN_FRAGMENTS = [
    "the problem", "the solution", "by the way", "the same thing",
    "the website", "the application", "the update", "the system",
    "machine learning", "deep learning", "data science", "artificial intelligence",
    "social media", "online", "the internet", "the link",
    "the game", "the match", "the score", "the team",
    "the movie", "the series", "the song", "the album",
    "good luck", "no problem", "for real", "the best",
    "the meeting", "the deadline", "the project", "the plan",
    "the phone", "the laptop", "the screen", "the camera",
    "full time", "part time", "freelance", "remote work",
]



# ── Templates Arabizi+FR/EN (code-switching tout en script latin) ──
# {FR} = sera remplacé par un fragment français
# {EN} = sera remplacé par un fragment anglais
ARABIZI_TEMPLATES = [
    "Wach derti l'{FR} wla mazal?",
    "Khassni n'envoyer had le {FR} avant le {EN}",
    "Ana bghit {FR} bach ndير had le {EN}",
    "Had l'{FR} fiha un bug f'l'{FR}, khass n'dirou liha {EN}",
    "Wach kayn chi {FR} f'had le {EN}?",
    "Chhal dial le {FR} bqaw? Khassni nchouf le {EN}",
    "Safi ana ghadi n'preparer les {FR} dyal had le {EN}",
    "Iwa sir chof l'{FR} dyalek, kayn chi {EN}",
    "Mazal ma dert l'{FR}, khassni {EN} aussi",
    "Bghit nchouf le {FR} avant la {FR}",
    "Derti l'{FR} f'la fac wla mazal? Khassek t'pripari les {FR}",
    "Wach had le {FR} working? Ana ma3reftch",
    "Khassni n'verifier le {FR} avant le {EN} dyal ghda",
    "Ana f'le {FR} daba, ghadi n'envoyer le {FR}",
    "Kolchi ready? Khassna n'finaliser le {FR}",
    "Wach n9dro n'changer le {FR} dyal had l'{FR}?",
    "Bghit n'reserver une {FR} f'had le {FR}",
    "Khoya dir le {EN} dyal had le {FR} safi",
    "Ma3reftch kifach n'configurer had l'{FR}, 3tini le {EN}",
    "Ila bghiti n'planifier le {EN}, khassek t'preparer les {FR}",
    "Daba ghadi n'checker les {FR} avant le {EN}",
    "Wach kayna la {FR} dyal had le {FR}?",
    "Safi n'khedam 3la le {FR} daba, aprés le {EN}",
    "Ana ghadi n'finaliser le {FR} had le weekend",
    "Bghit n'acheter un {FR} jdid, chhal le {FR}?",
    "Sir chouf le {EN} dyalek, 3andi wahed le {FR}",
    "Mazal khassni n'preparer la {FR} avant la {FR}",
    "Wach t9der t'envoyer le {FR} par le {EN}?",
    "Ana mrid, ghadi n'appeler le {FR} daba",
    "Khassni n'imprimer les {FR} avant la {FR} dyal ghda",
]

FR_NOUNS = [
    "inscription", "documents", "rapport", "réunion", "bureau",
    "formation", "application", "interface", "projet", "résidence",
    "quartier", "ville", "marché", "programme", "licence",
    "facture", "commande", "livraison", "pharmacie", "chambre",
    "présentation", "conférence", "entreprise", "contrat", "certificat",
    "loyer", "location", "appartement", "diplôme", "administration",
]

EN_NOUNS = [
    "deadline", "meeting", "update", "report", "email",
    "password", "website", "feedback", "schedule", "budget",
    "database", "server", "account", "invoice", "delivery",
    "backup", "download", "upload", "refund", "checkout",
]

# Patterns d'insertion du code-switching (uniquement les patterns naturels)
INSERTION_PATTERNS = [
    # Pattern 1: Fragment FR/EN au début puis arabe
    "fragment_start",
    # Pattern 2: Arabe puis fragment FR/EN à la fin
    "fragment_end",
    # Pattern 3: Arabizi+FR/EN (tout latin, mélange de langues)
    "arabizi_fr_en",
]





def inject_fragment_start(tokens: list, rng: random.Random, lang="fr") -> list:
    """Ajoute un fragment FR/EN au début."""
    fragments = FR_FRAGMENTS if lang == "fr" else EN_FRAGMENTS
    fragment = rng.choice(fragments)
    return fragment.split() + tokens


def inject_fragment_end(tokens: list, rng: random.Random, lang="fr") -> list:
    """Ajoute un fragment FR/EN à la fin."""
    fragments = FR_FRAGMENTS if lang == "fr" else EN_FRAGMENTS
    fragment = rng.choice(fragments)
    return tokens + fragment.split()








def generate_arabizi_mix(rng: random.Random) -> str:
    """Génère un exemple MIX 100% latin (Arabizi+FR/EN) à partir d'un template."""
    template = rng.choice(ARABIZI_TEMPLATES)
    
    # Remplacer les placeholders {FR} et {EN}
    result = template
    while "{FR}" in result:
        result = result.replace("{FR}", rng.choice(FR_NOUNS), 1)
    while "{EN}" in result:
        result = result.replace("{EN}", rng.choice(EN_NOUNS), 1)
    
    return result.lower()


def generate_mix_sample(text: str, rng: random.Random) -> str:
    """Génère un exemple MIX à partir d'un texte arabe source."""
    tokens = text.split()
    if len(tokens) < 3:
        return None

    # Choisir le pattern d'insertion (uniquement les patterns naturels)
    pattern = rng.choice(INSERTION_PATTERNS)
    lang = rng.choice(["fr", "en"])

    if pattern == "fragment_start":
        new_tokens = inject_fragment_start(tokens, rng, lang)
    elif pattern == "fragment_end":
        new_tokens = inject_fragment_end(tokens, rng, lang)
    elif pattern == "arabizi_fr_en":
        # Ce pattern ne dépend pas du texte source arabe
        return generate_arabizi_mix(rng)
    else:
        return None

    return " ".join(new_tokens)


def compute_cs_metrics(text: str) -> tuple:
    """Calcule cs_ratio, arab_ratio, latin_ratio."""
    tokens = text.split()
    if not tokens:
        return 0.0, 0.0, 0.0

    arab_pat = re.compile(r'[\u0600-\u06FF]+')
    latin_pat = re.compile(r'[a-zA-Z]+')

    arab_count = sum(1 for t in tokens if arab_pat.search(t))
    latin_count = sum(1 for t in tokens if latin_pat.search(t))

    arab_ratio = arab_count / len(tokens)
    latin_ratio = latin_count / len(tokens)
    cs_ratio = min(arab_ratio, latin_ratio)

    return cs_ratio, arab_ratio, latin_ratio


def main():
    config = load_config()
    logger = setup_logger("04c_mix_aug", log_level=config["project"]["log_level"])
    seed = config["project"]["seed"]
    rng = random.Random(seed)

    logger.info("Début de l'augmentation MIX")

    # 1. Charger les données train silver
    train_path = config["paths"]["data"]["train_silver"]
    df_train = pd.read_parquet(train_path)

    existing_mix = len(df_train[df_train["label_silver"] == "MIX"])
    logger.info(f"MIX existants dans train: {existing_mix}")

    # 2. Échantillonner des textes arabes comme base
    target_new = 5000  # Exemples avec fragment_start et fragment_end
    target_arabizi = 4000  # Exemples Arabizi+FR/EN (templates naturels)
    ar_dar = df_train[df_train["label_silver"] == "AR_DAR"]
    ar_msa = df_train[df_train["label_silver"] == "AR_MSA"]

    # Prendre 60% de DAR et 40% de MSA comme sources
    n_dar = int(target_new * 0.6)
    n_msa = target_new - n_dar

    source_dar = ar_dar.sample(n=min(n_dar, len(ar_dar)), random_state=seed)
    source_msa = ar_msa.sample(n=min(n_msa, len(ar_msa)), random_state=seed + 1)
    source_texts = pd.concat([source_dar, source_msa])

    logger.info(f"Textes sources sélectionnés: {len(source_texts)}")

    # 3. Générer les exemples MIX
    new_rows = []
    for _, row in source_texts.iterrows():
        text = row.get("text_norm", "")
        if not text or len(text.split()) < 3:
            continue

        mix_text = generate_mix_sample(text, rng)
        if mix_text is None:
            continue

        cs_ratio, arab_ratio, latin_ratio = compute_cs_metrics(mix_text)

        # Vérifier que c'est un vrai MIX (au moins 2 tokens de chaque script)
        if cs_ratio < 0.1:
            continue

        new_rows.append({
            "id": f"mix_aug_{uuid.uuid4().hex[:12]}",
            "text_raw": mix_text,
            "label_original": "MIX_AUG",
            "source": "augmentation",
            "text_norm": mix_text,
            "label_silver": "MIX",
            "confidence": 0.90,
            "cs_ratio": cs_ratio,
            "arab_ratio": arab_ratio,
            "latin_ratio": latin_ratio,
        })

    df_new = pd.DataFrame(new_rows)
    logger.info(f"Exemples MIX (script arabe+latin) générés: {len(df_new)}")

    # 3b. Générer des exemples MIX Arabizi+FR/EN (tout en latin)
    arabizi_rows = []
    for _ in range(target_arabizi):
        mix_text = generate_arabizi_mix(rng)
        if mix_text is None:
            continue
        arabizi_rows.append({
            "id": f"mix_arabizi_{uuid.uuid4().hex[:12]}",
            "text_raw": mix_text,
            "label_original": "MIX_AUG_ARABIZI",
            "source": "augmentation",
            "text_norm": mix_text,
            "label_silver": "MIX",
            "confidence": 0.90,
            "cs_ratio": 0.30,  # Estimation pour Arabizi-MIX
            "arab_ratio": 0.0,
            "latin_ratio": 1.0,
        })

    df_arabizi = pd.DataFrame(arabizi_rows)
    logger.info(f"Exemples MIX Arabizi+FR/EN générés: {len(df_arabizi)}")

    # Combiner les deux types de MIX augmentés
    df_all_new = pd.concat([df_new, df_arabizi], ignore_index=True)

    # 4. Fusionner avec le train existant
    df_augmented = pd.concat([df_train, df_all_new], ignore_index=True)

    # Bilan
    mix_total = len(df_augmented[df_augmented["label_silver"] == "MIX"])
    mix_pct = mix_total / len(df_augmented)
    logger.info(f"MIX total après augmentation: {mix_total} ({mix_pct:.1%} du train)")

    bilan = df_augmented["label_silver"].value_counts(normalize=True).to_dict()
    logger.info(f"Distribution finale: { {k: f'{v:.1%}' for k, v in bilan.items()} }")

    # 5. Sauvegarder
    ensure_dir(train_path)
    df_augmented.to_parquet(train_path, index=False)
    logger.info(f"Données augmentées sauvegardées dans {train_path}")

    # Montrer quelques exemples
    logger.info("Exemples générés:")
    for _, r in df_new.head(10).iterrows():
        logger.info(f"  cs={r['cs_ratio']:.2f} | {r['text_norm'][:100]}")

    logger.info("Augmentation MIX terminée.")


if __name__ == "__main__":
    main()
